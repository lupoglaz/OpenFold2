import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.Utils.weights_loading import load_params
from alphafold.Model.linear import Linear
from alphafold.Model.Utils.tensor_utils import permute_final_dims, flatten_final_dims
import math

from alphafold.Model.Opt.msa import AttentionOpt

from .mapping import inference_subbatch

class TriangleAttentionOpt(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L900
	"""
	def __init__(self, config, global_config, pair_dim:int) -> None:
		super(TriangleAttentionOpt, self).__init__()
		self.config = config
		self.global_config = global_config

		self.query_norm = nn.LayerNorm(pair_dim)
		self.feat_2d_weights = Linear(pair_dim, config.num_head, use_bias=False, initializer='normal')
		self.attn = AttentionOpt(config, global_config, pair_dim, pair_dim, pair_dim)

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.query_norm]
		names=['query_norm']
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
		
		if ind is None:
			d = data[f'{rel_path}']['feat_2d_weights']
		else:
			d = data[f'{rel_path}']['feat_2d_weights'][ind,...]
		print(f'Loading feat_2d_weights: {d.shape} -> {self.feat_2d_weights.weight.size()}')
		self.feat_2d_weights.weight.data.copy_(torch.from_numpy(d).transpose(-1,-2))
		
		self.attn.load_weights_from_af2(data, rel_path=f'{rel_path}/attention', ind=ind)

	def forward(self, pair_act: torch.Tensor, pair_mask: torch.Tensor, is_training:bool=False) -> torch.Tensor:
		assert pair_act.ndimension() == 3
		assert pair_mask.ndimension() == 2
		if self.config.orientation == 'per_column':
			pair_act = pair_act.transpose(-2, -3)
			pair_mask = pair_mask.transpose(-1, -2)
		
		bias = (1e9*(pair_mask.to(dtype=pair_act.dtype) - 1.0))[:, None, None, :]
		assert bias.ndimension() == 4

		pair_act = self.query_norm(pair_act)
		nonbatched_bias = self.feat_2d_weights(pair_act)
		nonbatched_bias = permute_final_dims(nonbatched_bias, (2,0,1))
		# pair_act = self.attn(pair_act, pair_act, bias, nonbatched_bias)
		pair_act = inference_subbatch(	self.attn, self.global_config.subbatch_size, 
							batched_args=[pair_act, pair_act, bias],
							nonbatched_args=[nonbatched_bias],
							low_memory=(not is_training))
		
		if self.config.orientation == 'per_column':
			pair_act = pair_act.transpose(-2, -3)
		
		return pair_act


class TriangleMultiplicationOpt(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1258
	and 
	https://github.com/aqlaboratory/openfold/blob/b47138dc3052d9af633c856fb38c6934983640b2/openfold/model/triangular_multiplicative_update.py#L26
	(this implementation has a bug, which I corrected)
	"""
	def __init__(self, config, global_config, pair_dim:int) -> None:
		super(TriangleMultiplicationOpt, self).__init__()
		self.config = config
		self.global_config = global_config

		self.layer_norm_input = nn.LayerNorm(pair_dim)
		self.left_projection = Linear(pair_dim, config.num_intermediate_channel)
		self.right_projection = Linear(pair_dim, config.num_intermediate_channel)
		self.left_gate = Linear(pair_dim, config.num_intermediate_channel, initializer='gating')
		self.right_gate = Linear(pair_dim, config.num_intermediate_channel, initializer='gating')
		self.sigmoid = nn.Sigmoid()
		self.center_layer_norm = nn.LayerNorm(config.num_intermediate_channel)
		self.output_projection = Linear(config.num_intermediate_channel, pair_dim, initializer='final')
		self.gating_linear = Linear(pair_dim, pair_dim, initializer='gating')

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.layer_norm_input, self.center_layer_norm]
		names=['layer_norm_input', 'center_layer_norm']
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
		
		modules=[self.left_projection, self.right_projection, self.left_gate, self.right_gate, self.output_projection, self.gating_linear]
		names=['left_projection', 'right_projection','left_gate','right_gate','output_projection','gating_linear']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['weights']
				b = data[f'{rel_path}/{name}']['bias']
			else:
				w = data[f'{rel_path}/{name}']['weights'][ind,...]
				b = data[f'{rel_path}/{name}']['bias'][ind,...]
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w).transpose(0, 1))
			module.bias.data.copy_(torch.from_numpy(b))

	def forward(self, pair_act: torch.Tensor, pair_mask: torch.Tensor, is_training: bool=False) -> torch.Tensor:
		pair_mask = pair_mask[..., None]
		input_act = self.layer_norm_input(pair_act)

		left_proj_act = self.left_projection(input_act) * pair_mask
		right_proj_act = self.right_projection(input_act) * pair_mask
		
		left_gate_values = self.sigmoid(self.left_gate(input_act))
		right_gate_values = self.sigmoid(self.right_gate(input_act))
		
		left_proj_act *= left_gate_values
		right_proj_act *= right_gate_values

		if self.config.equation == 'ikc,jkc->ijc': #triangle_multiplication_outgoing
			left_proj_act = permute_final_dims(left_proj_act, (2,0,1))
			right_proj_act = permute_final_dims(right_proj_act, (2,1,0))
			act = torch.matmul(left_proj_act, right_proj_act)
			act = permute_final_dims(act, (1,2,0))
        
		elif self.config.equation == 'kjc,kic->ijc': #triangle_multiplication_incoming
			left_proj_act = permute_final_dims(left_proj_act, (2,1,0))
			right_proj_act = permute_final_dims(right_proj_act, (2,0,1))
			act = torch.matmul(left_proj_act, right_proj_act)
			act = permute_final_dims(act, (2,1,0))
		else:
			raise ValueError(f"Unknown equation: {self.config.equation}")

		act = self.center_layer_norm(act)
		act = self.output_projection(act)

		gate_values = self.sigmoid(self.gating_linear(input_act))
		act *= gate_values
		
		return act

class OuterProductMeanOpt(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1422
	"""
	def __init__(self, config, global_config, num_output_channel:int, msa_dim:int) -> None:
		super(OuterProductMeanOpt, self).__init__()
		self.config = config
		self.global_config = global_config

		self.layer_norm_input = nn.LayerNorm(msa_dim)
		self.left_projection = Linear(msa_dim, config.num_outer_channel)
		self.right_projection = Linear(msa_dim, config.num_outer_channel)

		self.output_w = Linear(config.num_outer_channel * config.num_outer_channel, num_output_channel, initializer='final')
	
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.layer_norm_input]
		names=['layer_norm_input']
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
		
		modules=[self.left_projection, self.right_projection]
		names=['left_projection', 'right_projection']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['weights']
				b = data[f'{rel_path}/{name}']['bias']
			else:
				w = data[f'{rel_path}/{name}']['weights'][ind,...]
				b = data[f'{rel_path}/{name}']['bias'][ind,...]
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w).transpose(0, 1))
			module.bias.data.copy_(torch.from_numpy(b))
		
		if ind is None:
			d = data[f'{rel_path}']['output_w']
		else:
			d = data[f'{rel_path}']['output_w'][ind,...]
		print(f'Loading output_w: {d.shape} -> {self.output_w.weight.size()}')
		self.output_w.weight.data.copy_(torch.from_numpy(d).view(self.output_w.weight.size(1), self.output_w.weight.size(0)).transpose(0,1))

		if ind is None:
			d = data[f'{rel_path}']['output_b']
		else:
			d = data[f'{rel_path}']['output_b'][ind,...]
		print(f'Loading output_b: {d.shape} -> {self.output_w.bias.size()}')
		self.output_w.bias.data.copy_(torch.from_numpy(d))
	
	def forward(self, msa_act: torch.Tensor, msa_mask: torch.Tensor, is_training: bool=False) -> torch.Tensor:
		msa_mask = msa_mask[..., None]
		msa_act = self.layer_norm_input(msa_act)
		left_act = msa_mask * self.left_projection(msa_act)
		right_act = msa_mask * self.right_projection(msa_act)
		
		left_act = left_act.transpose(-2, -3)
		right_act = right_act.transpose(-2, -3)
		
		# https://github.com/lupoglaz/alphafold/blob/d7a4bd4c3ab1c403d5e4d849d408782b546402dc/alphafold/model/modules.py#L1497
		def compute_chunk(left_act, right_act):
			act = torch.einsum('...bac,...dae->...bdce', left_act, right_act)
			act = act.reshape(act.shape[:-2]+(-1,))
			act = self.output_w(act)
			return act
		act = inference_subbatch(compute_chunk, self.global_config.subbatch_size,
								batched_args=[left_act, right_act], nonbatched_args=[],
								low_memory=(not is_training), 
								input_subbatch_dim=0, output_subbatch_dims=[0,1])
				
		eps = 1e-3
		norm = torch.einsum('...abc,...adc->...bdc', msa_mask, msa_mask)
		act /= (norm.to(dtype=msa_mask.dtype) + eps)
		return act

class TransitionOpt(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L484
	"""
	def __init__(self, config, global_config, num_channel:int) -> None:
		super(TransitionOpt, self).__init__()
		self.config = config
		self.global_config = global_config

		num_intermediate = int(num_channel * config.num_intermediate_factor)
		self.input_layer_norm = nn.LayerNorm(num_channel)
		self.transition = nn.Sequential(nn.Linear(num_channel, num_intermediate),
										nn.ReLU(),
										nn.Linear(num_intermediate, num_channel))
		
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.input_layer_norm]
		names=['input_layer_norm']
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
		
		modules=[self.transition[0], self.transition[-1]]
		names=['transition1', 'transition2']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['weights']
				b = data[f'{rel_path}/{name}']['bias']
			else:
				w = data[f'{rel_path}/{name}']['weights'][ind,...]
				b = data[f'{rel_path}/{name}']['bias'][ind,...]
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w).transpose(0, 1))
			module.bias.data.copy_(torch.from_numpy(b))

	def forward(self, act: torch.Tensor, mask: torch.Tensor, is_training:bool=False) -> torch.Tensor:
		mask = mask.unsqueeze(dim=-1)
		act = self.input_layer_norm(act)
		# act = self.transition(act)
		act = inference_subbatch(	self.transition, self.global_config.subbatch_size,
									batched_args = [act], nonbatched_args = [],
									low_memory = not(is_training))
		return act