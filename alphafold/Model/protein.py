import torch
import torch.nn.functional as F
from alphafold.Model import affine
from alphafold.Common import residue_constants


def torsion_angles_to_frames(aatype:torch.Tensor, backb_to_global:affine.Rigids, torsion_angles_sin_cos:torch.Tensor) -> affine.Rigids:
	assert aatype.ndimension() == 1
	assert backb_to_global.rot.xx.ndimension() == 1
	assert torsion_angles_sin_cos.ndimension() == 3
	assert torsion_angles_sin_cos.size(1) == 7
	assert torsion_angles_sin_cos.size(2) == 2

	frames = torch.from_numpy(residue_constants.restype_rigid_group_default_frame)
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
	aatype = aatype.to(dtype=torch.long)
	
	restype_atom14_to_rigid_group = torch.from_numpy(residue_constants.restype_atom14_to_rigid_group)
	restype_atom14_rigid_group_positions = torch.from_numpy(residue_constants.restype_atom14_rigid_group_positions)
	restype_atom14_mask = torch.from_numpy(residue_constants.restype_atom14_mask)

	residx_to_group_idx = torch.gather(restype_atom14_to_rigid_group, dim=0, index=aatype[:,None].repeat(1, 14))
	group_mask = F.one_hot(residx_to_group_idx, 8)
	map_atoms_to_global = affine.rigids_apply(lambda x: torch.sum(x[:, None, :]*group_mask, dim=-1), all_frames_to_global)
	
	lit_positions_tensor = torch.gather(restype_atom14_rigid_group_positions, dim=0, index=aatype[:,None,None].repeat(1, 14, 3))
	lit_positions = affine.vecs_from_tensor(lit_positions_tensor)
	pred_positions = affine.rigids_mul_vecs(map_atoms_to_global, lit_positions)

	mask = torch.gather(restype_atom14_mask, dim=0, index=aatype[:,None].repeat(1, 14))
	pred_positions = affine.vecs_apply(lambda x: x*mask, pred_positions)
	return pred_positions
	
	