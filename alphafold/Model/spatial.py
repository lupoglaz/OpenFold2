from numpy import broadcast
import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.msa import Attention

class TriangleAttention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L900
	"""
	def __init__(self, config, global_config, pair_dim:int) -> None:
		super(TriangleAttention, self).__init__()
		self.config = config
		self.global_config = global_config

		self.query_norm = nn.LayerNorm(pair_dim)
		self.feat_2d_weights = nn.Parameter(torch.zeros(pair_dim, config.num_head))
		self.attn = Attention(config, global_config, pair_dim, pair_dim, pair_dim)

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
		print(f'Loading feat_2d_weights: {d.shape} -> {self.feat_2d_weights.size()}')
		self.feat_2d_weights.data.copy_(torch.from_numpy(d))
		
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
		nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
		pair_act = self.attn(pair_act, pair_act, bias, nonbatched_bias)

		if self.config.orientation == 'per_column':
			pair_act = pair_act.transpose(-2, -3)
		
		return pair_act

class TriangleMultiplication(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1258
	"""
	def __init__(self, config, global_config, pair_dim:int) -> None:
		super(TriangleMultiplication, self).__init__()
		self.config = config
		self.global_config = global_config

		self.layer_norm_input = nn.LayerNorm(pair_dim)
		self.left_projection = nn.Linear(pair_dim, config.num_intermediate_channel)
		self.right_projection = nn.Linear(pair_dim, config.num_intermediate_channel)
		self.left_gate = nn.Linear(pair_dim, config.num_intermediate_channel)
		self.right_gate = nn.Linear(pair_dim, config.num_intermediate_channel)
		self.sigmoid = nn.Sigmoid()
		self.center_layer_norm = nn.LayerNorm(config.num_intermediate_channel)
		self.output_projection = nn.Linear(config.num_intermediate_channel, pair_dim)
		self.gating_linear = nn.Linear(pair_dim, pair_dim)

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

		act = torch.einsum(self.config.equation, left_proj_act, right_proj_act)
		act = self.center_layer_norm(act)
		act = self.output_projection(act)

		gate_values = self.sigmoid(self.gating_linear(input_act))
		act *= gate_values
		
		return act

class OuterProductMean(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1422
	"""
	def __init__(self, config, global_config, num_output_channel:int, msa_dim:int) -> None:
		super(OuterProductMean, self).__init__()
		self.config = config
		self.global_config = global_config

		self.layer_norm_input = nn.LayerNorm(msa_dim)
		self.left_projection = nn.Linear(msa_dim, config.num_outer_channel)
		self.right_projection = nn.Linear(msa_dim, config.num_outer_channel)

		self.output_w = nn.Parameter(torch.zeros(config.num_outer_channel, config.num_outer_channel, num_output_channel))
		self.output_b = nn.Parameter(torch.zeros(num_output_channel))
	
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
		print(f'Loading output_w: {d.shape} -> {self.output_w.size()}')
		self.output_w.data.copy_(torch.from_numpy(d))

		if ind is None:
			d = data[f'{rel_path}']['output_b']
		else:
			d = data[f'{rel_path}']['output_b'][ind,...]
		print(f'Loading output_b: {d.shape} -> {self.output_b.size()}')
		self.output_b.data.copy_(torch.from_numpy(d))
	
	def forward(self, msa_act: torch.Tensor, msa_mask: torch.Tensor, is_training: bool=False) -> torch.Tensor:
		msa_mask = msa_mask[..., None].to(dtype=torch.long)
		msa_act = self.layer_norm_input(msa_act)
		left_act = msa_mask * self.left_projection(msa_act)
		right_act = msa_mask * self.right_projection(msa_act)

		act = torch.einsum('abc,ade->dceb', left_act, right_act)
		act = torch.einsum('dceb,cef->bdf', act, self.output_w) + self.output_b
		
		eps = 1e-3
		norm = torch.einsum('abc,adc->bdc', msa_mask, msa_mask)#>=1
		act /= (norm.to(dtype=msa_mask.dtype) + eps)
		return act

class Transition(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L484
	"""
	def __init__(self, config, global_config, num_channel:int) -> None:
		super(Transition, self).__init__()
		self.config = config
		self.global_config = global_config

		num_intermediate = int(num_channel * config.num_intermediate_factor)
		self.input_layer_norm = nn.LayerNorm(num_channel)
		self.transition1 = nn.Linear(num_channel, num_intermediate)
		self.relu = nn.ReLU()
		self.transition2 = nn.Linear(num_intermediate, num_channel)
	
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
		
		modules=[self.transition1, self.transition2]
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
		act = self.transition1(act)
		act = self.relu(act)
		act = self.transition2(act)
		return act