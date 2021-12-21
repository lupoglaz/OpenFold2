from numpy import broadcast
from numpy.lib.arraysetops import isin
import torch
from torch import nn
from typing import Sequence, Tuple, Dict
import numpy as np
from math import sqrt
from collections import namedtuple
from alphafold.Common import residue_constants

from alphafold.Model.affine import QuatAffine, vecs_to_tensor 
from alphafold.Model import protein

class InvariantPointAttention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L37
	"""
	def __init__(self, config, global_config, num_res:int, num_feat_2d:int, num_feat_1d:int, dist_epsilon:float=1e-8) -> None:
		super(InvariantPointAttention, self).__init__()
		self.config = config
		self.global_config = global_config
		self._dist_epsilon = dist_epsilon

		self.num_res = num_res
		self.num_head = self.config.num_head
		self.num_scalar_qk = self.config.num_scalar_qk
		self.num_scalar_v = self.config.num_scalar_v
		self.num_point_qk = self.config.num_point_qk
		self.num_point_v = self.config.num_point_v
			

		scalar_variance = max(self.num_scalar_qk, 1) * 1.0
		point_variance = max(self.num_point_qk, 1) * 9.0/2.0
		num_logit_terms = 3
		self.scalar_weights = sqrt(1.0/(num_logit_terms*scalar_variance))
		self.point_weights = sqrt(1.0/(num_logit_terms*point_variance))
		self.attention_2d_weights = sqrt(1.0/num_logit_terms)

		self.q_scalar = nn.Linear(num_feat_1d, self.num_head * self.num_scalar_qk)
		self.kv_scalar = nn.Linear(num_feat_1d, self.num_head*(self.num_scalar_v + self.num_scalar_qk))
		self.q_point_local = nn.Linear(num_feat_1d, self.num_head * 3 * self.num_point_qk)
		self.kv_point_local = nn.Linear(num_feat_1d, self.num_head * 3 * (self.num_point_qk + self.num_point_v))
		self.trainable_point_weights = nn.Parameter(torch.ones(self.num_head))
		self.attention_2d = nn.Linear(num_feat_2d, self.num_head)
		self.output_pojection = nn.Linear(self.num_head * (num_feat_2d + self.num_scalar_v + 4*self.num_point_v), self.config.num_channel)

		self.softplus = nn.Softplus()
		self.softmax = nn.Softmax(dim=-1)
	
	def load_weights_from_af2(self, data, rel_path: str='invariant_point_attention', ind:int=None):
		modules=[self.q_scalar, self.kv_scalar, self.q_point_local, self.kv_point_local, self.attention_2d,  self.output_pojection]
		names=['q_scalar', 'kv_scalar', 'q_point_local', 'kv_point_local', 'attention_2d', 'output_projection']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['weights'].transpose(-1,-2)
				b = data[f'{rel_path}/{name}']['bias']
			else:
				w = data[f'{rel_path}/{name}']['weights'][ind,...].transpose(-1,-2)
				b = data[f'{rel_path}/{name}']['bias'][ind,...]
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w))
			module.bias.data.copy_(torch.from_numpy(b))
		
		if ind is None:
			d = data[f'{rel_path}']['trainable_point_weights']
		else:
			d = data[f'{rel_path}']['trainable_point_weights'][ind,...]
		print(f'Loading trainable_point_weights: {d.shape} -> {self.trainable_point_weights.size()}')
		self.trainable_point_weights.data.copy_(torch.from_numpy(d))
		

	def forward(self, inputs_1d:torch.Tensor, inputs_2d:torch.Tensor, mask:torch.Tensor, affine) -> torch.Tensor:
		q_scalar = self.q_scalar(inputs_1d)
		q_scalar = q_scalar.view(self.num_res, self.num_head, self.num_scalar_qk)

		kv_scalar = self.kv_scalar(inputs_1d)
		kv_scalar = kv_scalar.view(self.num_res, self.num_head, self.num_scalar_v + self.num_scalar_qk)
		k_scalar, v_scalar = kv_scalar.split(self.num_scalar_qk, dim=-1)

		q_point_local = self.q_point_local(inputs_1d)
		q_point_local = q_point_local.split(self.num_head * self.num_point_qk, dim=-1)
		q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
		q_point = [x.view(self.num_res, self.num_head, self.num_point_qk) for x in q_point_global]
		
		kv_point_local = self.kv_point_local(inputs_1d)
		kv_point_local = kv_point_local.split(self.num_head * (self.num_point_qk + self.num_point_v), dim=-1)
		kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
		kv_point_global = [x.view(self.num_res, self.num_head, self.num_point_qk + self.num_point_v) for x in kv_point_global]
		k_point, v_point = list(zip(*[(x[:,:,:self.num_point_qk], x[:,:,self.num_point_qk:]) for x in kv_point_global]))
		
		point_weights = self.softplus(self.trainable_point_weights).unsqueeze(dim=1) * self.point_weights
		v_point = [x.transpose(-2, -3) for x in v_point]
		q_point = [x.transpose(-2, -3) for x in q_point]
		k_point = [x.transpose(-2, -3) for x in k_point]
		dist2 = [torch.pow(qx[:,:,None,:]-kx[:,None,:,:], 2) for qx, kx in zip(q_point, k_point)]
		dist2 = sum(dist2)
		
		attn_qk_point = -0.5 * torch.sum(point_weights[:,None,None,:] * dist2, dim=-1)
		
		v = v_scalar.transpose(-2, -3)
		q = (self.scalar_weights * q_scalar).transpose(-2, -3)
		k = k_scalar.transpose(-2, -3)

		attn_qk_scalar = torch.matmul(q, k.transpose(-2, -1))
		attn_logits = attn_qk_scalar + attn_qk_point
		attention_2d = self.attention_2d(inputs_2d)
		attn_logits += attention_2d.permute(2,0,1) * float(self.attention_2d_weights)
		
		mask_2d = mask * (mask.transpose(-1, -2))
		attn_logits -= 1e5 * (1.0 - mask_2d)

		attn = self.softmax(attn_logits)
		result_scalar = torch.matmul(attn, v).transpose(-2, -3)
		result_point_global = [torch.sum(attn[:,:,:,None]*vx[:,None,:,:], dim=-2).transpose(-2, -3) for vx in v_point]

		output_features = []
		result_scalar = result_scalar.reshape(self.num_res, self.num_head*self.num_scalar_v)
		output_features.append(result_scalar)
		
		result_point_global = [r.reshape(self.num_res, self.num_head*self.num_point_v) for r in result_point_global]
		result_point_local = affine.invert_point(result_point_global, extra_dims=1)
		output_features.extend(result_point_local)
		
		output_features.append(torch.sqrt(self._dist_epsilon 
										+ torch.pow(result_point_local[0], 2)
										+ torch.pow(result_point_local[1], 2)
										+ torch.pow(result_point_local[2], 2)))
		

		result_attention_over_2d = torch.einsum('hij,ijc->ihc', attn, inputs_2d)
		num_out = self.num_head * result_attention_over_2d.shape[-1]
		output_features.append(result_attention_over_2d.view(self.num_res, num_out))
		final_act = torch.cat(output_features, dim=-1)
		
		return self.output_pojection(final_act)

class MultiRigidSidechain(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L929
	"""
	def __init__(self, config, global_config, repr_dim:int, num_repr:int) -> None:
		super(MultiRigidSidechain, self).__init__()
		self.config = config
		self.global_config = global_config

		self.num_repr = num_repr

		self.input_projection = nn.ModuleList([nn.Linear(repr_dim, self.config.num_channel)
											for i in range(num_repr)])
		self.resblock1 = nn.ModuleList([nn.Linear(self.config.num_channel, self.config.num_channel) 
										for i in range(self.config.num_residual_block)])
		self.resblock2 = nn.ModuleList([nn.Linear(self.config.num_channel, self.config.num_channel) 
										for i in range(self.config.num_residual_block)])
		self.unnormalized_angles = nn.Linear(self.config.num_channel, 14)
		self.relu = nn.ReLU()

	def load_weights_from_af2(self, data, rel_path: str='rigid_sidechain', ind:int=None):
		modules=[self.resblock1, self.resblock2, self.input_projection, self.unnormalized_angles]
		names=['resblock1', 'resblock2', 'input_projection', 'unnormalized_angles']
		nums=[self.config.num_residual_block, self.config.num_residual_block, self.num_repr, 1]
		for module, name, num in zip(modules, names, nums):
			for i in range(num):
				if i==0:
					add_str = ''
				else:
					add_str = f'_{i}'
				if ind is None:
					w = data[f'{rel_path}/{name}{add_str}']['weights'].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias']
				else:
					w = data[f'{rel_path}/{name}{add_str}']['weights'][ind,...].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias'][ind,...]
				if isinstance(module, nn.ModuleList):
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module[i].weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module[i].bias.size()}')
					module[i].weight.data.copy_(torch.from_numpy(w))
					module[i].bias.data.copy_(torch.from_numpy(b))
				else:
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module.weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module.bias.size()}')
					module.weight.data.copy_(torch.from_numpy(w))
					module.bias.data.copy_(torch.from_numpy(b))

	def l2_normalize(self, x:torch.Tensor, dim:int=-1, eps:float=1e-12) -> torch.Tensor:
		return x / torch.sqrt(torch.sum(x*x, dim=dim, keepdim=True) + eps)

	def forward(self, affine:QuatAffine, representations_list: Sequence[torch.Tensor], aatype:torch.Tensor):
		act = [self.input_projection[i](self.relu(x)) for i, x in enumerate(representations_list)]
		act = sum(act)

		for i in range(self.config.num_residual_block):
			old_act = act
			act = self.resblock1[i](self.relu(act))
			act = self.resblock2[i](self.relu(act))
			act += old_act

		unnormalized_angles = self.unnormalized_angles(self.relu(act))
		unnormalized_angles = unnormalized_angles.view(act.size(0), 7, 2)
		angles = self.l2_normalize(unnormalized_angles, dim=-1)

		backb_to_global = affine.to_rigids()
		all_frames_to_global = protein.torsion_angles_to_frames(aatype, backb_to_global, angles)
		pred_positions = protein.frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global)
		outputs = {	'angles_sin_cos': angles,
					'unnormalized_angles_sin_cos': unnormalized_angles,
					'atom_pos': pred_positions,
					'frames': all_frames_to_global}
		return outputs

class FoldIteration(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L281
	"""
	affine_update_size = 6
	def __init__(self, config, global_config, num_res:int, num_feat_1d:int, num_feat_2d:int) -> None:
		super(FoldIteration, self).__init__()
		self.config = config
		self.global_config = global_config

		self.attention_module = InvariantPointAttention(config, global_config, 
								num_res=num_res, num_feat_1d=num_feat_1d, num_feat_2d=num_feat_2d)
		self.attention_layer_norm = nn.LayerNorm(num_feat_1d)
		self.transition = nn.ModuleList([nn.Linear(self.config.num_channel, self.config.num_channel) 
										for i in range(self.config.num_layer_in_transition)])
		self.transition_layer_norm = nn.LayerNorm(self.config.num_channel)
		self.relu = nn.ReLU()

		self.affine_update = nn.Linear(self.config.num_channel, self.affine_update_size)
		self.side_chain = MultiRigidSidechain(config.sidechain, global_config, num_repr=2, repr_dim=self.config.num_channel)

	def load_weights_from_af2(self, data, rel_path: str='fold_iteration', ind:int=None):
		modules=[self.transition, self.affine_update]
		names=['transition', 'affine_update']
		nums=[self.config.num_layer_in_transition, 1]
		for module, name, num in zip(modules, names, nums):
			for i in range(num):
				if i==0:
					add_str = ''
				else:
					add_str = f'_{i}'
				if ind is None:
					w = data[f'{rel_path}/{name}{add_str}']['weights'].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias']
				else:
					w = data[f'{rel_path}/{name}{add_str}']['weights'][ind,...].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias'][ind,...]
				if isinstance(module, nn.ModuleList):
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module[i].weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module[i].bias.size()}')
					module[i].weight.data.copy_(torch.from_numpy(w))
					module[i].bias.data.copy_(torch.from_numpy(b))
				else:
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module.weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module.bias.size()}')
					module.weight.data.copy_(torch.from_numpy(w))
					module.bias.data.copy_(torch.from_numpy(b))

		modules=[self.attention_layer_norm, self.transition_layer_norm]
		names=['attention_layer_norm', 'transition_layer_norm']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['scale']
				b = data[f'{rel_path}/{name}']['offset']
			else:
				w = data[f'{rel_path}/{name}']['scale'][ind,...]
				b = data[f'{rel_path}/{name}']['offset'][ind,...]
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w))
			module.bias.data.copy_(torch.from_numpy(b))

		self.side_chain.load_weights_from_af2(data, rel_path=f'{rel_path}/rigid_sidechain')
		self.attention_module.load_weights_from_af2(data, rel_path=f'{rel_path}/invariant_point_attention')

	def forward(self, activations:torch.Tensor, sequence_mask:torch.Tensor, update_affine:bool, initial_act:torch.Tensor, 
					is_training:bool=False, static_feat_2d:torch.Tensor=None, aatype:torch.Tensor=None):
		affine = QuatAffine.from_tensor(activations['affine'].to(dtype=activations['act'].dtype))
		act = activations['act']
		attn = self.attention_module(inputs_1d=act, inputs_2d=static_feat_2d, mask=sequence_mask, affine=affine)
		act += attn
		act = self.attention_layer_norm(act)

		input_act = act
		for i in range(self.config.num_layer_in_transition):
			act = self.transition[i](act)
			if i < self.config.num_layer_in_transition - 1:
				act = self.relu(act)
		act += input_act
		act = self.transition_layer_norm(act)

		if update_affine:
			affine_update = self.affine_update(act)
			affine = affine.pre_compose(affine_update)

		sc = self.side_chain(affine.scale_translation(self.config.position_scale), [act, initial_act], aatype)
		outputs = {'affine': affine.to_tensor(), 'sc': sc}
		new_activations = {'act': act,	'affine': affine.apply_rotation_tensor_fn(torch.detach).to_tensor()}

		return new_activations, outputs

def recursive_apply(func, *data, **params):
	if isinstance(data[0], dict):
		ret = {}
		for key in data[0].keys():
			new_data = tuple([dat[key] for dat in data])
			ret[key] = recursive_apply(func, *new_data, **params)
		return ret
	elif isinstance(data[0], list) or isinstance(data[0], tuple):
		ret = []
		for key in range(len(data[0])):
			new_data = tuple([dat[key] for dat in data])
			ret.append(recursive_apply(func, *new_data, **params))
		if isinstance(data[0], tuple):
			return data[0]._make(ret)
		else:
			return ret
	elif isinstance(data[0], torch.Tensor):
		return func(*data, **params)

class StructureModule(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L464
	"""
	def __init__(self, config, global_config, num_res:int, num_feat_1d:int, num_feat_2d:int, compute_loss:bool=False) -> None:
		super(StructureModule, self).__init__()
		self.config = config
		self.global_config = global_config
		self.compute_loss = compute_loss

		self.single_layer_norm = nn.LayerNorm(num_feat_1d)
		self.pair_layer_norm = nn.LayerNorm(num_feat_2d)
		self.initial_projection = nn.Linear(num_feat_1d, self.config.num_channel)

		self.fold_iteration = FoldIteration(config, global_config, num_res, num_feat_1d, num_feat_2d)

	def load_weights_from_af2(self, data, rel_path: str='structure_module', ind:int=None):
		modules=[self.initial_projection]
		names=['initial_projection']
		nums=[1]
		for module, name, num in zip(modules, names, nums):
			for i in range(num):
				if i==0:
					add_str = ''
				else:
					add_str = f'_{i}'
				if ind is None:
					w = data[f'{rel_path}/{name}{add_str}']['weights'].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias']
				else:
					w = data[f'{rel_path}/{name}{add_str}']['weights'][ind,...].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias'][ind,...]
				if isinstance(module, nn.ModuleList):
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module[i].weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module[i].bias.size()}')
					module[i].weight.data.copy_(torch.from_numpy(w))
					module[i].bias.data.copy_(torch.from_numpy(b))
				else:
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module.weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module.bias.size()}')
					module.weight.data.copy_(torch.from_numpy(w))
					module.bias.data.copy_(torch.from_numpy(b))

		modules=[self.single_layer_norm, self.pair_layer_norm]
		names=['single_layer_norm', 'pair_layer_norm']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['scale']
				b = data[f'{rel_path}/{name}']['offset']
			else:
				w = data[f'{rel_path}/{name}']['scale'][ind,...]
				b = data[f'{rel_path}/{name}']['offset'][ind,...]
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w))
			module.bias.data.copy_(torch.from_numpy(b))

		self.fold_iteration.load_weights_from_af2(data, rel_path=f'{rel_path}/fold_iteration')

	def generate_new_affine(self, sequence_mask:torch.Tensor):
		num_residues = sequence_mask.size(0)
		quaternion = torch.tile(torch.Tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 4), (num_residues, 1))
		translation = torch.zeros(num_residues, 3)
		return QuatAffine(quaternion, translation, unstack_inputs=True)

	def generate_affines(self, representations, batch, is_training:bool=False):
		assert batch['seq_mask'].ndimension() == 1
		sequence_mask = batch['seq_mask'][:, None]
		
		act = self.single_layer_norm(representations['single'])
		initial_act = act
		act = self.initial_projection(act)
		affine = self.generate_new_affine(sequence_mask)
		activations = {'act': act, 'affine':affine.to_tensor()}
		act_2d = self.pair_layer_norm(representations['pair'])

		outputs = []
		for i in range(self.config.num_layer):
			activations, output = self.fold_iteration(
				activations,
				initial_act=initial_act,
				static_feat_2d=act_2d,
				sequence_mask=sequence_mask,
				update_affine=True,
				is_training=is_training,
				aatype=batch['aatype']
			)
			outputs.append(output)

		output = recursive_apply(lambda *x: torch.stack(x), *outputs)
		output['act'] = activations['act']
		return output


	def forward(self, representations, batch, is_training:bool=False):
		output = self.generate_affines(representations, batch, is_training=is_training)
		atom14_pred_positions = vecs_to_tensor(output['sc']['atom_pos'])[-1]
		atom37_pred_positions = protein.atom14_to_atom37(atom14_pred_positions, batch)
		atom37_pred_positions *= batch['atom37_atom_exists'][:, :, None]
		traj = output['affine'] * torch.tensor([1.0]*4 + [self.config.position_scale]*3)
		ret = {
			'representations': {
				'structure_module': output['act']
				},
			'traj': traj,
			'sidechains': output['sc'],
			'final_atom14_positions': atom14_pred_positions,
			'final_atom14_mask': batch['atom14_atom_exists'],
			'final_atom_positions': atom37_pred_positions,
			'final_atom_mask': batch['atom37_atom_exists'],
			'final_affines': traj[-1]
		}
		if self.compute_loss:
			return ret
		else:
			return {k: ret[k] for k in {'final_atom_positions', 'final_atom_mask', 'representations'}}

	def loss(self, value:Dict[str, torch.Tensor], batch:Dict[str, torch.Tensor]):
		ret = { 'value': 0.0,
				'metrics': {}
				}
		if self.config.compute_in_graph_metrics:
			atom14_pred_positions = value['final_atom14_positions']
			value.update(self.compute_renamed_ground_truth(batch, atom14_pred_positions))
			value['violations'] = self.find_structural_violations(batch, atom14_pred_positions)
			violation_metrics = self.compute_violation_metrics(batch, atom14_pred_positions, value['violations'])
			ret['metrics'].update(violation_metrics)
		
		backbone_loss(ret, batch, value, self.config)

		if not('renamed_atom14_gt_positions' in value):
			value.update(self.compute_renamed_ground_truth(batch, atom14_pred_positions))
		sc_loss = sidechain_loss(batch, value, self.config)

		ret['loss'] = ((1.0 - self.config.sidechain.weight_frac)*ret['loss'] + self.config.sidechain.weight_frac*sc_loss['loss'])
		ret['sidechain_fape'] = sc_loss['loss']

		supervised_chi_loss(ret, batch, value, self.config)

		if self.config.structural_violation_loss_weight:
			if not('violations' in value):
				value['violations'] = self.find_structural_violations(batch, atom14_pred_positions)
			structural_violation_loss(ret, batch, value, self.config)

	def compute_renamed_ground_truth(self, batch:Dict[str, torch.Tensor], atom14_pred_positions:torch.Tensor)->Dict[str, torch.Tensor]:
		"""
		https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L561
		"""
		alt_naming_is_better = protein.find_optimal_renaming(
			atom14_gt_positions=batch['atom14_gt_positions'],
			atom14_alt_gt_positions=batch['atom14_alt_gt_positions'],
			atom14_atom_is_ambiguous=batch['atom14_atom_is_ambiguous'],
			atom14_gt_exists=batch['atom14_gt_exists'],
			atom14_pred_positions=batch['atom14_pred_positions'],
			atom14_pred_exists=batch['atom14_pred_exists'])
		
		renamed_atom14_gt_positions = (1.0 - alt_naming_is_better[:,None,None])*batch['atom14_gt_positions'] + \
										alt_naming_is_better[:,None,None]*batch['atom14_alt_gt_positions']
		renamed_atom14_gt_mask = (1.0 - alt_naming_is_better[:,None,None])*batch['atom14_gt_exists'] + \
										alt_naming_is_better[:,None,None]*batch['atom14_gt_exists']
		
		return {'alt_naming_is_better': alt_naming_is_better, 
				'renamed_atom14_gt_positions': renamed_atom14_gt_positions,
				'renamed_atom14_exists': renamed_atom14_gt_mask}

	def find_structural_violations(self, batch:Dict[str, torch.Tensor], atom14_pred_positions:torch.Tensor):
		"""
		https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L734
		"""
		connection_violations = protein.between_residue_bond_loss(
			pred_atom_positions=atom14_pred_positions,
			pred_atom_mask=batch['atom14_atom_exists'].to(dtype=torch.float32),
			residue_index=batch['residue_index'].to(dtype=torch.float32),
			aatype=batch['aatype'],
			tolerance_factor_soft=self.config.violation_tolerance_factor,
			tolerance_factor_hard=self.config.violation_tolerance_factor
		)
		atomtype_radius = torch.Tensor([residue_constants.van_der_waals_radius[name[0]] for name in residue_constants.atom_types])
		atom14_atom_radius = batch['atom14_atom_exists'] * torch.gather(atomtype_radius, dim=0, index=batch['residx_atom14_to_atom37'])

		between_residue_clashes = protein.between_residue_clash_loss(
			atom14_pred_positions=atom14_pred_positions,
			atom14_atom_exists=batch['atom14_atom_exists'].to(dtype=torch.float32),
			atom14_atom_radius=atom14_atom_radius,
			residue_index=batch['residue_index'].to(dtype=torch.float32),
			overlap_tolerance_soft=self.config.clash_overlap_tolerance,
			overlap_tolerance_hard=self.config.clash_overlap_tolerance
		)

		restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
			overlap_tolerance=self.conf.clash_overlap_tolerance,
			bond_length_tolerance_factor=self.conf.violation_tolerance_factor
		)
		aatypes = batch['aatype'].to(dtype=torch.long)
		atom14_dists_lower_bound = torch.gather(restype_atom14_bounds['lower_bound'], dim=0, index=aatypes)
		atom14_dists_upper_bound = torch.gather(restype_atom14_bounds['upper_bound'], dim=0, index=aatypes)
		within_residue_violations = protein.within_residue_violations(
			atom14_pred_positions=atom14_pred_positions, 
			atom14_atom_exists=batch['atom14_atom_exists'].to(dtype=torch.float32), 
			atom14_dists_lower_bound=atom14_dists_lower_bound, 
			atom14_dists_upper_bound=atom14_dists_upper_bound,
			tighten_bounds_for_loss=0.0
		)
		per_residue_violations_mask = torch.max(torch.stack([
			connection_violations['per_residue_violation_mask'],
			torch.max(between_residue_clashes['per_atom_clash_mask'], dim=-1).values,
			torch.max(within_residue_violations['per_atom_violations'], dim=-1).values]), dim=0).values

		return {'between_residues': {
					'bonds_c_n_loss_mean': connection_violations['c_n_loss_mean'],  # ()
					'angles_ca_c_n_loss_mean': connection_violations['ca_c_n_loss_mean'],  # ()
					'angles_c_n_ca_loss_mean': connection_violations['c_n_ca_loss_mean'],  # ()
					'connections_per_residue_loss_sum': connection_violations['per_residue_loss_sum'],  # (N)
					'connections_per_residue_violation_mask': connection_violations['per_residue_violation_mask'],  # (N)
					'clashes_mean_loss': between_residue_clashes['mean_loss'],  # ()
					'clashes_per_atom_loss_sum': between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
					'clashes_per_atom_clash_mask': between_residue_clashes['per_atom_clash_mask']},  # (N, 14)
				'within_residues': {
					'per_atom_loss_sum': within_residue_violations['per_atom_loss_sum'],  # (N, 14)
					'per_atom_violations': within_residue_violations['per_atom_violations']},  # (N, 14),
				'total_per_residue_violations_mask': per_residue_violations_mask}  # (N)
	
	def compute_violation_metrics(self, 
		batch:Dict[str, torch.Tensor], 
		atom14_pred_positions:torch.Tensor, 
		violations:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

		extreme_ca_ca_violations = protein.extreme_ca_ca_distance_violations(
			pred_atom_positions=atom14_pred_positions, 
			pred_atom_mask=batch['atom14_atom_exists'].to(dtype=torch.float32), 
			residue_index=batch['residue_index'].to(dtype=torch.float32)
		)
		return {
			'violations_extreme_ca_ca_distance': extreme_ca_ca_violations,
			'violations_between_residue_bond': torch.sum(
				violations['between_residues']['connections_per_residue_violation_mask'] * \
					batch['seq_mask'])/(torch.sum(batch['seq_mask'])+1e-10),
			'violations_between_residue_clash':torch.sum(
				torch.max(violations['between_residues']['clashes_per_atom_clash_mask'], dim=-1).values * \
					batch['seq_mask'])/(torch.sum(batch['seq_mask'])+1e-10),
			'violations_within_residue': torch.sum(
				torch.max(violations['within_residues']['per_atom_violations'], dim=-1).values * \
					batch['seq_mask'])/(torch.sum(batch['seq_mask'])+1e-10),
			'violations_per_residue': torch.sum(
				violations['total_per_residue_violations_mask'] * \
					batch['seq_mask'])/(torch.sum(batch['seq_mask'])+1e-10),
			}