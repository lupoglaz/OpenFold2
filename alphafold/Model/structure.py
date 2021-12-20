from numpy import broadcast
from numpy.lib.arraysetops import isin
import torch
from torch import nn
from typing import Sequence, Tuple
import numpy as np
from math import sqrt

from alphafold.Model.affine import QuatAffine 
from alphafold.Model import protein

class InvariantPointAttention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L37
	"""
	def __init__(self, config, global_config, num_res:int, num_seq:int, num_feat_2d:int, num_feat_1d:int, dist_epsilon:float=1e-8) -> None:
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
	def __init__(self, config, global_config, num_res:int, num_seq:int, num_feat_1d:int, num_feat_2d:int) -> None:
		super(FoldIteration, self).__init__()
		self.config = config
		self.global_config = global_config

		self.attention_module = InvariantPointAttention(config, global_config, 
								num_res=num_res, num_seq=num_seq, num_feat_1d=num_feat_1d, num_feat_2d=num_feat_2d)
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
