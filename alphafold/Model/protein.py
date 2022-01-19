import torch
import torch.nn.functional as F
from alphafold.Model import affine
from alphafold.Common import residue_constants
from typing import Dict, Optional

import numpy as np


def torsion_angles_to_frames(aatype:torch.Tensor, backb_to_global:affine.Rigids, torsion_angles_sin_cos:torch.Tensor) -> affine.Rigids:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L445
	"""
	assert aatype.ndimension() == 1
	assert backb_to_global.rot.xx.ndimension() == 1
	assert torsion_angles_sin_cos.ndimension() == 3
	assert torsion_angles_sin_cos.size(1) == 7
	assert torsion_angles_sin_cos.size(2) == 2
	device = torsion_angles_sin_cos.device
	dtype = torsion_angles_sin_cos.dtype

	frames = torch.from_numpy(residue_constants.restype_rigid_group_default_frame).to(device=device, dtype=dtype)
	idxs = aatype.to(dtype=torch.long)[...,None,None,None].repeat(1, frames.size(1), frames.size(2), frames.size(3))
	m = torch.gather(frames, dim=0, index=idxs)	
	default_frames = affine.rigids_from_tensor4x4(m)
	

	sin_angles = torsion_angles_sin_cos[..., 0]
	cos_angles = torsion_angles_sin_cos[..., 1]
	num_res = aatype.size(0)
	sin_angles = torch.cat([torch.zeros(num_res, 1, dtype=sin_angles.dtype, device=sin_angles.device), sin_angles], dim=-1)
	cos_angles = torch.cat([torch.ones(num_res, 1, dtype=sin_angles.dtype, device=sin_angles.device), cos_angles], dim=-1)
	zeros = torch.zeros_like(sin_angles)
	ones = torch.ones_like(sin_angles)

	all_rots = affine.Rots(	ones, zeros, zeros,
							zeros, cos_angles, -sin_angles,
							zeros, sin_angles, cos_angles)

	all_frames = affine.rigids_mul_rots(default_frames, all_rots)

	chi2_frame_to_frame = affine.rigids_apply(lambda x: x[:, 5], all_frames)
	chi3_frame_to_frame = affine.rigids_apply(lambda x: x[:, 6], all_frames)
	chi4_frame_to_frame = affine.rigids_apply(lambda x: x[:, 7], all_frames)

	chi1_frame_to_backb = affine.rigids_apply(lambda x: x[:, 4], all_frames)
	chi2_frame_to_backb = affine.rigids_mul_rigids(chi1_frame_to_backb, chi2_frame_to_frame)
	chi3_frame_to_backb = affine.rigids_mul_rigids(chi2_frame_to_backb, chi3_frame_to_frame)
	chi4_frame_to_backb = affine.rigids_mul_rigids(chi3_frame_to_backb, chi4_frame_to_frame)
	
	def _concat_frames(xall, x5, x6, x7):
		return torch.cat([xall[:, :5], x5[:, None], x6[:, None], x7[:, None]], dim=-1)

	all_frames_to_backb = affine.rigids_apply(_concat_frames, all_frames, chi2_frame_to_backb, chi3_frame_to_backb, chi4_frame_to_backb)
	all_frames_to_global = affine.rigids_mul_rigids(
							affine.rigids_apply(lambda x: x[:, None], backb_to_global),
							all_frames_to_backb)
	return all_frames_to_global

def frames_and_literature_positions_to_atom14_pos(aatype:torch.Tensor, all_frames_to_global:affine.Rigids)->affine.Vecs:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L532
	"""
	aatype = aatype.to(dtype=torch.long)
	device = all_frames_to_global.trans.x.device
	dtype = all_frames_to_global.trans.x.dtype
	
	restype_atom14_to_rigid_group = torch.from_numpy(residue_constants.restype_atom14_to_rigid_group).to(device=device)
	restype_atom14_rigid_group_positions = torch.from_numpy(residue_constants.restype_atom14_rigid_group_positions).to(device=device, dtype=dtype)
	restype_atom14_mask = torch.from_numpy(residue_constants.restype_atom14_mask).to(device=device)

	residx_to_group_idx = torch.gather(restype_atom14_to_rigid_group, dim=0, index=aatype[:,None].repeat(1, 14))
	group_mask = F.one_hot(residx_to_group_idx, 8)
	map_atoms_to_global = affine.rigids_apply(lambda x: torch.sum(x[:, None, :]*group_mask, dim=-1), all_frames_to_global)
	
	lit_positions_tensor = torch.gather(restype_atom14_rigid_group_positions, dim=0, index=aatype[:,None,None].repeat(1, 14, 3))
	lit_positions = affine.vecs_from_tensor(lit_positions_tensor)
	pred_positions = affine.rigids_mul_vecs(map_atoms_to_global, lit_positions)

	mask = torch.gather(restype_atom14_mask, dim=0, index=aatype[:,None].repeat(1, 14))
	pred_positions = affine.vecs_apply(lambda x: x*mask, pred_positions)
	return pred_positions
	
def atom14_to_atom37(atom14_data:torch.Tensor, batch:Dict[str, torch.Tensor]) -> torch.Tensor:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L76
	"""
	assert atom14_data.ndimension() in [2, 3]
	assert 'residx_atom37_to_atom14' in batch
	assert 'atom37_atom_exists' in batch
	idxs = batch['residx_atom37_to_atom14'].to(dtype=torch.long).unsqueeze(dim=-1).repeat(1,1,3)
	atom37_data = torch.gather(atom14_data, dim=1, index=idxs)
	if atom14_data.ndimension() == 2:
		atom37_data *= batch['atom37_atom_exists']
	elif atom14_data.ndimension() == 3:
		atom37_data *= batch['atom37_atom_exists'][:, :, None].to(atom37_data.dtype)
	
	return atom37_data

def dist(x:torch.Tensor, y:torch.Tensor=None, eps:float=1e-10, dim:int=2):
	if y is None:
		if dim==2:
			d = x[:,None,:,None,:] - x[None,:,None,:,:]
		elif dim==1:
			d = x[:,:,None,:] - x[:,None,:,:]
		else:
			raise(NotImplementedError())
	else:
		d = x - y
	return torch.sqrt(eps + torch.sum(d*d, dim=-1))

def find_optimal_renaming(
		atom14_gt_positions:torch.Tensor, #(N, 14, 3)
		atom14_alt_gt_positions:torch.Tensor, #(N, 14, 3)
		atom14_atom_is_ambiguous:torch.Tensor, #(N,14)
		atom14_gt_exists:torch.Tensor, #(N, 14)
		atom14_pred_positions:torch.Tensor, #(N, 14, 3)
		atom14_pred_exists:torch.Tensor #(N, 14)
		) -> torch.Tensor: #(N,)
	assert atom14_gt_positions.ndimension() == 3
	assert atom14_alt_gt_positions.ndimension() == 3
	assert atom14_atom_is_ambiguous.ndimension() == 2
	assert atom14_gt_exists.ndimension() == 2
	assert atom14_pred_positions.ndimension() == 3
	assert atom14_pred_exists.ndimension() == 2

	pred_dists = dist(atom14_pred_positions)
	gt_dists = dist(atom14_gt_positions)
	alt_gt_dists = dist(atom14_alt_gt_positions)
	
	lddt = dist(pred_dists, gt_dists)
	alt_lddt = dist(pred_dists, alt_gt_dists)

	mask = 	atom14_gt_exists[:,None,:,None] * atom14_atom_is_ambiguous[:,None,:,None] * \
			atom14_gt_exists[None,:,None,:] * (1-atom14_atom_is_ambiguous[None,:,None,:])

	per_res_lddt = torch.sum(mask * lddt, dim=(1,2,3))
	alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(1,2,3))

	alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).to(dtype=torch.float32)
	return alt_naming_is_better

def between_residue_bond_loss(
		pred_atom_positions:torch.Tensor, pred_atom_mask:torch.Tensor, 
		residue_index:torch.Tensor, aatype:torch.Tensor,
		tolerance_factor_soft:float=12.0, tolerance_factor_hard:float=12.0
	) -> Dict[str, torch.Tensor]:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L609
	"""
	assert pred_atom_positions.ndimension() == 3
	assert pred_atom_mask.ndimension() == 2
	assert residue_index.ndimension() == 1
	assert aatype.ndimension() == 1

	this_ca_pos, this_ca_mask = pred_atom_positions[:-1, 1, :], pred_atom_mask[:-1, 1]
	this_c_pos, this_c_mask = pred_atom_positions[:-1, 2, :], pred_atom_mask[:-1, 2]
	next_n_pos, next_n_mask = pred_atom_positions[1:, 0, :], pred_atom_mask[1:, 0]
	next_ca_pos, next_ca_mask = pred_atom_positions[1:, 1, :], pred_atom_mask[1:, 1]
	has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).to(dtype=torch.float32)

	def compare(value, gt_value, gt_stddev, mask):
		value_error = torch.sqrt(1e-6 + torch.square(value - gt_value))
		loss_per_residue = F.relu(value_error - tolerance_factor_soft*gt_stddev)
		loss = torch.sum(mask * loss_per_residue) / (torch.sum(mask) + 1e-6)
		violation_mask = mask * (value_error > (tolerance_factor_hard*gt_stddev))
		return loss, violation_mask, loss_per_residue

	#distances			
	c_n_bond_length = dist(this_c_pos, next_n_pos, eps=1e-6)
	next_is_proline = (aatype[1:] == residue_constants.resname_to_idx['PRO']).to(dtype=torch.float32)
	
	c_n_loss, c_n_violation_mask, c_n_loss_per_residue = compare(	value=c_n_bond_length, 
					gt_value = (1.0 - next_is_proline)*residue_constants.between_res_bond_length_c_n[0] + \
						next_is_proline*residue_constants.between_res_bond_length_c_n[1], 
					gt_stddev = (1.0 - next_is_proline)*residue_constants.between_res_bond_length_stddev_c_n[0] + \
						next_is_proline*residue_constants.between_res_bond_length_stddev_c_n[1], 
					mask = this_c_mask * next_n_mask * has_no_gap_mask)
	
	#angles
	ca_c_bond_length = dist(this_ca_pos, this_c_pos, eps=1e-6)
	n_ca_bond_length = dist(next_n_pos, next_ca_pos, eps=1e-6)
	c_ca_unit_vec = (this_ca_pos - this_c_pos)/ca_c_bond_length[:,None]
	c_n_unit_vec = (next_n_pos - this_c_pos)/c_n_bond_length[:,None]
	n_ca_unit_vec = (next_ca_pos - next_n_pos)/n_ca_bond_length[:,None]

	#ca_c_n
	ca_c_n_loss, ca_c_n_violation_mask, ca_c_n_loss_per_residue = compare(value=torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1),
											gt_value=residue_constants.between_res_cos_angles_ca_c_n[0],
											gt_stddev=residue_constants.between_res_cos_angles_ca_c_n[1],
											mask=this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
											)
		
	c_n_ca_loss, c_n_ca_violation_mask, c_n_ca_loss_per_residue = compare(value=torch.sum((-c_n_unit_vec)*n_ca_unit_vec, dim=-1),
											gt_value=residue_constants.between_res_cos_angles_c_n_ca[0],
											gt_stddev=residue_constants.between_res_cos_angles_c_n_ca[1],
											mask=this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
											)
	per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
	per_residue_loss_sum = 0.5 * (F.pad(per_residue_loss_sum, [[0, 1]]) + F.pad(per_residue_loss_sum, [[1, 0]]))

	violation_mask = torch.max(torch.stack([c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask]), dim=0)
	violation_mask = 0.5 * (F.pad(violation_mask, [[0, 1]]) + F.pad(violation_mask, [[1, 0]]))

	return {'c_n_loss': c_n_loss, 'ca_c_n_loss': ca_c_n_loss, 'c_n_ca_loss':c_n_ca_loss,
			'per_residue_loss_sum': per_residue_loss_sum, 'per_residue_violation_mask': violation_mask}

def between_residue_clash_loss(
		atom14_pred_positions:torch.Tensor, atom14_atom_exists:torch.Tensor, atom14_atom_radius:torch.Tensor,
		residue_index:torch.Tensor,
		overlap_tolerance_soft:float=1.5, overlap_tolerance_hard:float=1.5
	) -> Dict[str, torch.Tensor]:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L744
	"""
	assert atom14_pred_positions.ndimension() == 3
	assert atom14_atom_exists.ndimension() == 2
	assert atom14_atom_radius.ndimension() == 2
	assert residue_index.ndimension() == 1	

	dists = dist(atom14_pred_positions, eps=1e-10)
	dists_mask = atom14_atom_exists[:,None,:,None,:] * atom14_atom_exists[None,:,None,:,:]
	dists_mask *= (residue_index[:,None,None,None]<residue_index[None,:,None,None])

	c_one_hot = F.one_hot(torch.Tensor([2], dtype=torch.long), num_classes=14)
	n_one_hot = F.one_hot(torch.Tensor([0], dtype=torch.long), num_classes=14)
	neighbour_mask = ((residue_index[:,None,None,None]+1) == residue_index[None,:,None,None])
	c_n_bonds = neighbour_mask * c_one_hot[None,None,:,None] * n_one_hot[None,None,None,:]
	dists_mask *= (1.0 - c_n_bonds)

	cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
	cys_sg_one_hot = F.one_hot(torch.Tensor([cys_sg_idx], dtype=torch.long), num_classes=14)
	disulfide_bonds = neighbour_mask * cys_sg_one_hot[None,None,:,None] * cys_sg_one_hot[None,None,None,:]
	dists_mask *= (1.0 - disulfide_bonds)

	dists_lower_bound = dists_mask * (atom14_atom_radius[:,None,:,None] + atom14_atom_radius[None,:,None,:])
	dists_to_low_error = dists_mask * F.relu(dists_lower_bound - overlap_tolerance_soft - dists)

	mean_loss = torch.sum(dists_to_low_error)/(1e-6 + torch.sum(dists_mask))
	per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(0,2)) + torch.sum(dists_to_low_error, dim=(1,3))

	clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))
	clash_mask1 = clash_mask.permute([1,3,0,2]).view(clash_mask.size(1), clash_mask.size(3), clash_mask.size(0)*clash_mask.size(2))
	clash_mask2 = clash_mask.permute([0,2,1,3]).view(clash_mask.size(0), clash_mask.size(2), clash_mask.size(1)*clash_mask.size(3))
	per_atom_clash_mask = torch.maximum(torch.max(clash_mask1, dim=-1).values, torch.max(clash_mask2, dim=-1).values)

	return {'mean_loss': mean_loss, 'per_atom_loss_sum':per_atom_loss_sum, 'per_atom_clash_mask':per_atom_clash_mask}

def within_residue_violations(
		atom14_pred_positions:torch.Tensor, atom14_atom_exists:torch.Tensor, 
		atom14_dists_lower_bound:torch.Tensor, atom14_dists_upper_bound:torch.Tensor, 
		tighten_bounds_for_loss = 0.0
	) -> Dict[str, torch.Tensor]:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L853
	"""
	assert atom14_pred_positions.ndimension() == 3
	assert atom14_atom_exists.ndimension() == 2
	assert atom14_dists_lower_bound.ndimension() == 3
	assert atom14_dists_upper_bound.ndimension() == 3

	dists_mask = (1.0 - torch.eye(14,14)[None])
	dists_mask *= (atom14_atom_exists[:, :, None] * atom14_atom_exists[:, None, :])
	dists = dist(atom14_pred_positions, dim=1)

	dists_to_low_error = F.relu(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
	dists_to_high_error = F.relu(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
	loss = dists_mask * (dists_to_low_error + dists_to_high_error)

	per_atom_loss_sum = torch.sum(loss, dim=1) + torch.sum(loss, dim=2)
	violations =  dists_mask*((dists > atom14_dists_upper_bound) | (dists < atom14_dists_lower_bound))
	per_atom_violations = torch.maximum(torch.max(violations, dim=1).values, torch.max(violations, dim=2).values)

	return {'per_atom_loss_sum': per_atom_loss_sum, 'per_atom_violations':per_atom_violations}

def extreme_ca_ca_distance_violations(
		pred_atom_positions:torch.Tensor, pred_atom_mask:torch.Tensor, residue_index:torch.Tensor,
		max_angstrom_tolerance=1.5
	) ->torch.Tensor :
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L575
	"""
	this_ca_pos, this_ca_mask = pred_atom_positions[:-1,1,:], pred_atom_mask[:-1,1]
	next_ca_pos, next_ca_mask = pred_atom_positions[1:,1,:], pred_atom_mask[1:,1]
	has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).to(dtype=torch.float32)
	ca_ca_distance = dist(this_ca_pos, next_ca_pos, eps=1e-6)
	violations = (ca_ca_distance - residue_constants.ca_ca) > max_angstrom_tolerance
	mask = this_ca_mask*next_ca_mask*has_no_gap_mask
	return torch.sum(violations*mask)/(torch.sum(mask)+1e-10)

def frame_aligned_point_error(
		pred_frames:affine.Rigids, target_frames:affine.Rigids, frames_mask:torch.Tensor,
		pred_positions:affine.Vecs, target_positions:affine.Vecs, positions_mask:torch.Tensor,
		length_scale:float, l1_clamp_distance:Optional[float]=None, epsilon=1e-4 
	) -> torch.Tensor:
	assert pred_frames.rot.xx.ndimension() == 1
	assert target_frames.rot.xx.ndimension() == 1
	assert frames_mask.ndimension() == 1
	assert pred_positions.x.ndimension() == 1
	assert target_positions.x.ndimension() == 1
	assert positions_mask.ndimension() == 1

	local_pred_pos = affine.rigids_mul_vecs(
							affine.rigids_apply(lambda r: r[:, None], affine.rigids_invert(pred_frames)),
							affine.rigids_apply(lambda x: x[None, :], pred_positions))

	local_target_pos = affine.rigids_mul_vecs(
							affine.rigids_apply(lambda r: r[:, None], affine.rigids_invert(target_frames)),
							affine.rigids_apply(lambda x: x[None, :], target_positions))

	error_dist = torch.sqrt(affine.vecs_squared_dist(local_pred_pos, local_target_pos) + epsilon)
	if l1_clamp_distance:
		error_dist = torch.clamp(error_dist, min=0.0, max=l1_clamp_distance)
	
	normed_error = (error_dist/length_scale)*frames_mask.unsqueeze(dim=-1)*positions_mask.unsqueeze(dim=-2)
	normalization_factor = torch.sum(frames_mask, dim=-1) * torch.sum(positions_mask, dim=-1)
	return torch.sum(normed_error, dim=(-2, -1)) / (epsilon + normalization_factor)


def atom37_to_frames(aatype:torch.Tensor, all_atom_positions:torch.Tensor, all_atom_mask:torch.Tensor) -> Dict[str, torch.Tensor]:
	"""https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L114"""
	aatype_in_shape = aatype.shape
	aatype = aatype.view(-1)
	all_atom_positions = all_atom_positions.view(-1, 37, 3)
	all_atom_mask = all_atom_mask.view(-1, 37)
	restype_rigidbody_base_atom_names = np.full([21, 8, 3], '', dtype=object)
	restype_rigidbody_base_atom_names[:, 0, :] = ['C', 'CA', 'N']
	restype_rigidbody_base_atom_names[:, 3, :] = ['CA', 'C', 'O']
	for restype, restype_letter in enumerate(residue_constants.restypes):
		resname = residue_constants.restype_1to3[restype_letter]
		for chi_idx in range(4):
			if residue_constants.chi_angles_mask[restype][chi_idx]:
				atom_names = residue_constants.chi_angles_atoms[resname][chi_idx]
				restype_rigidbody_base_atom_names[restype, chi_idx+4, :] = atom_names[1:]

	restype_rigidgroup_mask = torch.zeros(21, 8, dtype=torch.float32)
	restype_rigidgroup_mask[:, 0] = 1
	restype_rigidgroup_mask[:, 3] = 1
	restype_rigidgroup_mask[:20, 4:] = residue_constants.chi_angles_mask

	lookuptable = residue_constants.atom_order.copy()
	lookuptable[''] = 0
	restype_rigidbody_base_atom_names = np.vectorize(lambda x: lookuptable[x])(restype_rigidbody_base_atom_names)
	residx_regidgroup_base_atom37_idx = torch.gather(restype_rigidbody_base_atom_names, 0, aatype)
	base_atom_pos = torch.gather(all_atom_positions, 1, residx_regidgroup_base_atom37_idx)

	gt_frames = affine.rigids_from_3_points(
				point_on_neg_axis=affine.vecs_from_tensor(base_atom_pos[:, :, 0, :]),
				origin=affine.vecs_from_tensor(base_atom_pos[:, :, 1, :]),
				point_on_xy_plane=affine.vecs_from_tensor(base_atom_pos[:, :, 2, :]))
	group_exists = torch.gather(restype_rigidgroup_mask, aatype)
	# gt_atom_exists = torch.gather()
	



def atom37_to_torsion_angles():
	"""https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/all_atom.py#L271"""
	pass

