import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.Utils.weights_loading import load_params
from alphafold.Model.linear import Linear
from alphafold.Model.Utils.tensor_utils import permute_final_dims, flatten_final_dims
import math

from alphafold.Model.Opt.batch_msa import AttentionFFB

from .mapping import inference_subbatch
from FastFold.Kernel import bias_dropout_add, bias_ele_dropout_residual
from FastFold.Kernel import LayerNorm as LayerNormFF

class TriangleAttentionFFB(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L900
	"""
	def __init__(self, config, global_config, pair_dim:int) -> None:
		super(TriangleAttentionFFB, self).__init__()
		self.config = config
		self.global_config = global_config

		self.query_norm = LayerNormFF(pair_dim)
		self.feat_2d_weights = Linear(pair_dim, config.num_head, use_bias=False, initializer='normal')
		self.attn = AttentionFFB(config, global_config, pair_dim, pair_dim, pair_dim, last_bias_fuse=True)
		self.out_bias = nn.parameter.Parameter(torch.zeros(pair_dim))

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
		
		fused_bias = self.attn.load_weights_from_af2(data, rel_path=f'{rel_path}/attention', ind=ind)
		self.out_bias.data.copy_(fused_bias.reshape(self.out_bias.shape))

	def forward(self, pair_act_raw: torch.Tensor, pair_mask: torch.Tensor, is_training:bool=False) -> torch.Tensor:
		assert pair_act_raw.ndimension() == 4
		assert pair_mask.ndimension() == 3
		if self.config.orientation == 'per_column':
			pair_act = pair_act_raw.transpose(-2, -3)
			pair_mask = pair_mask.transpose(-1, -2)
		else:
			pair_act = pair_act_raw
		
		pair_act = self.query_norm(pair_act)
		nonbatched_bias = self.feat_2d_weights(pair_act)
		pair_act = inference_subbatch(	self.attn, self.global_config.subbatch_size, 
							batched_args=[pair_act, pair_mask],
							nonbatched_args=[nonbatched_bias],
							low_memory=(not is_training))
		
		if self.config.orientation == 'per_column':
			pair_act = pair_act.transpose(-2, -3)
			dropout_mask = torch.ones_like(pair_act[:, :, 0:1, :], device=pair_act.device, dtype=pair_act.dtype)
		else:
			dropout_mask = torch.ones_like(pair_act[:, 0:1, :, :], device=pair_act.device, dtype=pair_act.dtype)

		return bias_dropout_add(pair_act, self.out_bias, dropout_mask, pair_act_raw, prob=self.config.dropout_rate, training=is_training)


class TriangleMultiplicationFFB(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1258
	and 
	https://github.com/aqlaboratory/openfold/blob/b47138dc3052d9af633c856fb38c6934983640b2/openfold/model/triangular_multiplicative_update.py#L26
	(this implementation has a bug, which I corrected)
	"""
	def __init__(self, config, global_config, pair_dim:int) -> None:
		super(TriangleMultiplicationFFB, self).__init__()
		self.config = config
		self.global_config = global_config

		self.layer_norm_input = LayerNormFF(pair_dim)
		self.left_right_projection = Linear(pair_dim, 2*config.num_intermediate_channel)
		self.left_right_gate = Linear(pair_dim, 2*config.num_intermediate_channel, initializer='gating')
		self.sigmoid = nn.Sigmoid()
		self.center_layer_norm = LayerNormFF(config.num_intermediate_channel)
		self.gating_linear = Linear(pair_dim, pair_dim, initializer='gating')
		self.output_projection = Linear(config.num_intermediate_channel, pair_dim, initializer='final', use_bias=False)
		self.out_bias = nn.parameter.Parameter(torch.zeros(pair_dim))
		

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
		
		lr_projection_weight = []
		lr_projection_bias = []
		lr_gate_weight = []
		lr_gate_bias = []
		modules=[None, None, None, None, self.output_projection, self.gating_linear]
		names=['left_projection', 'right_projection','left_gate','right_gate','output_projection','gating_linear']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['weights']
				b = data[f'{rel_path}/{name}']['bias']
			else:
				w = data[f'{rel_path}/{name}']['weights'][ind,...]
				b = data[f'{rel_path}/{name}']['bias'][ind,...]
			
			if name in ('left_projection', 'right_projection'):
				print(f'Loading {name}: {w.shape} -> special')
				print(f'Loading {name}.bias: {b.shape} -> special')
				lr_projection_weight.append(torch.from_numpy(w).transpose(0, 1))
				lr_projection_bias.append(torch.from_numpy(b))

			elif name in ('left_gate', 'right_gate'):
				print(f'Loading {name}.weight: {w.shape} -> special')
				print(f'Loading {name}.bias: {b.shape} -> special')
				lr_gate_weight.append(torch.from_numpy(w).transpose(0, 1))
				lr_gate_bias.append(torch.from_numpy(b))
			
			elif name in ('output_projection'):
				print(f'Loading {name}.weight: {w.shape} -> special')
				print(f'Loading {name}.bias: {b.shape} -> special')
				module.weight.data.copy_(torch.from_numpy(w).transpose(0, 1))
				self.out_bias.data.copy_(torch.from_numpy(b))
				
			else:
				print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
				print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
				module.weight.data.copy_(torch.from_numpy(w).transpose(0, 1))
				module.bias.data.copy_(torch.from_numpy(b))

		lr_projection_weight = torch.cat(lr_projection_weight, dim=0)
		lr_projection_bias = torch.cat(lr_projection_bias, dim=0)
		print(f'Loading left_right_projection.weight: {lr_projection_weight.shape} -> {self.left_right_projection.weight.size()}')
		print(f'Loading left_right_projection.weight: {lr_projection_bias.shape} -> {self.left_right_projection.bias.size()}')
		self.left_right_projection.weight.data.copy_(lr_projection_weight)
		self.left_right_projection.bias.data.copy_(lr_projection_bias)
		
		lr_gate_weight = torch.cat(lr_gate_weight, dim=0)
		lr_gate_bias = torch.cat(lr_gate_bias, dim=0)
		print(f'Loading left_right_gate.weight: {lr_gate_weight.shape} -> {self.left_right_gate.weight.size()}')
		print(f'Loading left_right_gate.weight: {lr_gate_bias.shape} -> {self.left_right_gate.bias.size()}')
		self.left_right_gate.weight.data.copy_(lr_gate_weight)
		self.left_right_gate.bias.data.copy_(lr_gate_bias)
		

	def forward(self, pair_act_raw: torch.Tensor, pair_mask: torch.Tensor, is_training: bool=False) -> torch.Tensor:
		pair_mask = pair_mask[..., None]
		input_act = self.layer_norm_input(pair_act_raw)

		left_right_proj_act = self.left_right_projection(input_act)
		left_right_proj_act = left_right_proj_act * pair_mask
		
		left_right_proj_act *= self.sigmoid(self.left_right_gate(input_act))
		left_proj_act, right_proj_act = left_right_proj_act.chunk(2, dim=-1)

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
				
		dropout_mask = torch.ones_like(act, device=act.device, dtype=act.dtype)
		return bias_ele_dropout_residual(act, self.out_bias, gate_values, dropout_mask, pair_act_raw, prob=self.config.dropout_rate, training=is_training)

class OuterProductMeanFFB(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1422
	"""
	def __init__(self, config, global_config, num_output_channel:int, msa_dim:int) -> None:
		super(OuterProductMeanFFB, self).__init__()
		self.config = config
		self.global_config = global_config

		self.layer_norm_input = LayerNormFF(msa_dim)
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
		norm = torch.einsum('...abc,...adc->...bdc', msa_mask, msa_mask)#>=1
		act /= (norm.to(dtype=msa_mask.dtype) + eps)
		return act

class TransitionFFB(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L484
	"""
	def __init__(self, config, global_config, num_channel:int) -> None:
		super(TransitionFFB, self).__init__()
		self.config = config
		self.global_config = global_config

		num_intermediate = int(num_channel * config.num_intermediate_factor)
		self.input_layer_norm = LayerNormFF(num_channel)
		self.transition = nn.Sequential(Linear(num_channel, num_intermediate, initializer='relu'),
										nn.ReLU(),
										Linear(num_intermediate, num_channel, initializer='final'))
		
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

	def forward(self, act_raw: torch.Tensor, mask: torch.Tensor, is_training:bool=False) -> torch.Tensor:
		mask = mask.unsqueeze(dim=-1)
		act = self.input_layer_norm(act_raw)
		# act = self.transition(act)
		act = inference_subbatch(	self.transition, self.global_config.subbatch_size,
									batched_args = [act], nonbatched_args = [],
									low_memory = not(is_training))
		return act + act_raw