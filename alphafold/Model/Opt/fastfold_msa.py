import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.Utils.weights_loading import load_params
from alphafold.Model.linear import Linear
from alphafold.Model.Utils.tensor_utils import permute_final_dims, flatten_final_dims
import math

from alphafold.Model.msa import MSAColumnAttention, MSAColumnGlobalAttention
from .mapping import inference_subbatch
from einops import rearrange

class AttentionFF(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L546
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L167
	and 
	https://github.com/hpcaitech/FastFold/blob/16d10d6a7852520601fa35cf1b15a8d89668c59b/fastfold/model/ops.py#L123
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int,
				q_chunk_size:int=None, kv_chunk_size:int=None) -> None:
		super(AttentionFF, self).__init__()
		self.config = config
		self.global_config = global_config
		self.output_dim = output_dim

		all_key_dim = key_dim
		all_value_dim = value_dim
		self.num_head = self.config.num_head
		assert all_key_dim % self.num_head == 0
		assert all_value_dim % self.num_head == 0
		self.key_dim = all_key_dim // self.num_head
		self.value_dim = all_value_dim // self.num_head

		self.q_weights = Linear(all_key_dim, all_key_dim, use_bias=False, initializer='glorot')
		self.k_weights = Linear(all_value_dim, all_value_dim, use_bias=False, initializer='glorot')
		self.v_weights = Linear(all_value_dim, all_value_dim, use_bias=False, initializer='glorot')
		# self.qkv_weights = Linear(all_key_dim, all_key_dim, use_bias=False, initializer='glorot')
		self.o_linear = Linear(all_value_dim, self.output_dim, initializer='final')

		self.softmax = nn.Softmax(dim=-1)
		self.sigmoid = nn.Sigmoid()

		if self.config.gating:
			self.gating_linear = Linear(all_key_dim, all_value_dim, initializer='gating')

		
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.q_weights, self.k_weights, self.v_weights, self.o_linear]
		param_names=[('query_w',None), ('key_w',None), ('value_w',None), ('output_w', 'output_b')]
		if self.config.gating:
			modules.extend([self.gating_linear])
			param_names.extend([('gating_w', 'gating_b')])
		for module, (name_w, name_b) in zip(modules, param_names):
			if rel_path is None:
				data_weight = data[f'{name_w}']
				if not(name_b is None):
					data_bias = data[f'{name_b}']
			else:
				if ind is None:
					data_weight = data[f'{rel_path}'][f'{name_w}']
					if not(name_b is None):
						data_bias = data[f'{rel_path}'][f'{name_b}']
				else:
					data_weight = data[f'{rel_path}'][f'{name_w}'][ind,...]
					if not(name_b is None):
						data_bias = data[f'{rel_path}'][f'{name_b}'][ind,...]
			print(f'Loading {name_w}: {data_weight.shape} -> {module.weight.size()}')
			if name_w in ('query_w', 'key_w', 'value_w', 'gating_w'):
				data_weight = torch.from_numpy(data_weight).reshape(module.weight.shape).transpose(-1,-2)
			else:
				data_weight = torch.from_numpy(data_weight).permute(2,0,1).reshape(module.weight.shape)
			module.weight.data.copy_(data_weight)
			if not(name_b is None):
				print(f'Loading {name_b}: {data_bias.shape} -> {module.bias.size()}')
				module.bias.data.copy_(torch.from_numpy(data_bias).reshape(module.bias.shape))

	def forward(self, q_data: torch.Tensor, m_data: torch.Tensor, bias: torch.Tensor, nonbatched_bias: torch.Tensor=None) -> torch.Tensor:
		"""
		Arguments: 
			q_data: [batch_size, num_queries, querry_dim]
			m_data: [batch_size, num_keys, value_dim]
			bias: [batch_size, num_queries, num_keys]
			nonbatched_bias: [num_queries, num_keys]
		Returns:
			[batch_size, num_queries, output_dim]
		"""
		flat_head = lambda t: t.view(t.shape[:-1] + (self.num_head, -1))
		assert self.key_dim * self.num_head == q_data.size(-1)
		assert self.value_dim * self.num_head == m_data.size(-1)
		
		q = self.q_weights(q_data) * (1./math.sqrt(self.key_dim))
		q = flat_head(q)

		k = self.k_weights(m_data)
		k = flat_head(k)

		v = self.v_weights(m_data)
		v = flat_head(v)
				
		q = permute_final_dims(q, (1, 0, 2))
		k = permute_final_dims(k, (1, 2, 0))
		logits = torch.matmul(q, k) + bias
		del q, k

		if not(nonbatched_bias is None):
			logits += nonbatched_bias.unsqueeze(dim=0)
		

		weights = self.softmax(logits)
		v = permute_final_dims(v, (1, 0, 2))
		weighted_avg = torch.matmul(weights, v).transpose(-2, -3)

		if self.config.gating:
			gate_values = self.gating_linear(q_data)
			gate_values = self.sigmoid(gate_values)
			gate_values = flat_head(gate_values)
			weighted_avg *= gate_values

		weighted_avg = flatten_final_dims(weighted_avg, 2)
		output = self.o_linear(weighted_avg)
		
		return output

class GlobalAttentionOpt(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L630
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L300
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int) -> None:
		super(GlobalAttentionOpt, self).__init__()
		self.config = config
		self.global_config = global_config
		self.output_dim = output_dim

		all_key_dim = key_dim
		all_value_dim = value_dim
		self.num_head = self.config.num_head
		assert all_key_dim % self.num_head == 0
		assert all_value_dim % self.num_head == 0
		self.key_dim = all_key_dim // self.num_head
		self.value_dim = all_value_dim // self.num_head

		self.q_weights = Linear(all_key_dim, all_key_dim, use_bias=False, initializer='glorot')
		self.k_weights = Linear(all_value_dim, self.key_dim, use_bias=False, initializer='glorot')
		self.v_weights = Linear(all_value_dim, self.value_dim, use_bias=False, initializer='glorot')
		self.o_linear = Linear(all_value_dim, self.output_dim, initializer='final')

		self.softmax = nn.Softmax(dim=-1)
		self.sigmoid = nn.Sigmoid()

		if self.config.gating:
			self.gating_linear = Linear(all_key_dim, all_value_dim, initializer='gating')
	
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.q_weights, self.k_weights, self.v_weights, self.o_linear]
		param_names=[('query_w',None), ('key_w',None), ('value_w',None), ('output_w', 'output_b')]
		if self.config.gating:
			modules.extend([self.gating_linear])
			param_names.extend([('gating_w', 'gating_b')])
		for module, (name_w, name_b) in zip(modules, param_names):
			if rel_path is None:
				data_weight = data[f'{name_w}']
				if not(name_b is None):
					data_bias = data[f'{name_b}']
			else:
				if ind is None:
					data_weight = data[f'{rel_path}'][f'{name_w}']
					if not(name_b is None):
						data_bias = data[f'{rel_path}'][f'{name_b}']
				else:
					data_weight = data[f'{rel_path}'][f'{name_w}'][ind,...]
					if not(name_b is None):
						data_bias = data[f'{rel_path}'][f'{name_b}'][ind,...]
			print(f'Loading {name_w}: {data_weight.shape} -> {module.weight.size()}')
			if name_w in ('query_w', 'gating_w'):
				data_weight = torch.from_numpy(data_weight).reshape(module.weight.shape).transpose(-1,-2)
			elif name_w in ('output_w'):
				data_weight = torch.from_numpy(data_weight).permute(2,0,1).reshape(module.weight.shape)
			else:
				data_weight = torch.from_numpy(data_weight).transpose(-1,-2)

			module.weight.data.copy_(data_weight)
			if not(name_b is None):
				print(f'Loading {name_b}: {data_bias.shape} -> {module.bias.size()}')
				module.bias.data.copy_(torch.from_numpy(data_bias).reshape(module.bias.shape))

	def mask_mean(self, mask: torch.Tensor, value: torch.Tensor, dims: Sequence[int]=None) -> torch.Tensor:
		"""
		https://github.com/lupoglaz/alphafold/blob/f485c308855ca8b67cf7f575f0dc4af4c432d4d5/alphafold/model/utils.py#L42
		"""
		assert mask.ndimension() == value.ndimension()
		if dims is None:
			dims = list(range(mask.ndimension()))
		
		broadcast_factor = 1.0
		for dim_ in dims:
			value_size = value.size(dim_)
			mask_size = mask.size(dim_)
			if mask_size == 1:
				broadcast_factor *= value_size
			else:
				assert value_size == mask_size
		return torch.sum(mask*value, dim=dims) / (torch.sum(mask, dim=dims) * broadcast_factor + 1e-10)

	def forward(self, q_data: torch.Tensor, m_data: torch.Tensor, q_mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
		"""
		Arguments: 
			q_data: [batch_size, num_queries, querry_dim]
			m_data: [batch_size, num_keys, value_dim]
			q_mask: [batch_size, num_queries, querry_dim]
			bias: [batch_size, num_queries, num_keys]
		Returns:
			[batch_size, num_queries, output_dim]
		"""
		flat_head = lambda t: t.view(t.shape[:-1] + (self.num_head, -1))
		assert self.key_dim * self.num_head == q_data.size(-1)
		assert self.value_dim * self.num_head == m_data.size(-1)

		v = self.v_weights(m_data)
		k = self.k_weights(m_data)

		q_avg = self.mask_mean(q_mask, q_data, dims=[1])
		q = self.q_weights(q_avg) * (1.0/math.sqrt(self.key_dim))
		q = flat_head(q)

		bias = (1e9*(q_mask[:,None,:,0] - 1.0))
		logits = torch.matmul(q, k.transpose(-1,-2)) + bias
		weights = self.softmax(logits)
		weighted_avg = torch.matmul(weights, v)
		
		if self.config.gating:
			gate_values = self.gating_linear(q_data)
			gate_values = self.sigmoid(gate_values)
			gate_values = flat_head(gate_values)
			weighted_avg = weighted_avg.unsqueeze(dim=-3) * gate_values
			weighted_avg = flatten_final_dims(weighted_avg, num_dims=2)
			output = self.o_linear(weighted_avg)
		else:
			raise NotImplemented()
		return output

class MSARowAttentionWithPairBiasOpt(nn.Module):
	"""
	Optimized MSARowAttentionWithPairBias
	"""
	def __init__(self, config, global_config, pair_dim:int, msa_dim:int) -> None:
		super(MSARowAttentionWithPairBiasOpt, self).__init__()
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.feat_2d_norm = nn.LayerNorm(pair_dim)
		self.feat_2d_weights = Linear(pair_dim, config.num_head, use_bias=False, initializer='normal')
		self.attn = AttentionOpt(config, global_config, msa_dim, msa_dim, msa_dim)

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.query_norm, self.feat_2d_norm]
		names=['query_norm', 'feat_2d_norm']
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
		

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, pair_act:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_row'

		bias = (1e9 * (msa_mask.to(dtype=msa_act.dtype)-1.0))[:,None,None,:]
		msa_act = self.query_norm(msa_act)
		pair_act = self.feat_2d_norm(pair_act)
		nonbatched_bias = self.feat_2d_weights(pair_act)
		nonbatched_bias = permute_final_dims(nonbatched_bias, (2,0,1))
		# msa_act = self.attn(msa_act, msa_act, bias, nonbatched_bias)
		msa_act = inference_subbatch(self.attn, self.global_config.subbatch_size, 
									batched_args=[msa_act, msa_act, bias],
									nonbatched_args=[nonbatched_bias],
									low_memory=(not is_training))
		return msa_act

class MSAColumnAttentionOpt(MSAColumnAttention):
	"""
	Optimized MSAColumnAttention
	"""
	def __init__(self, config, global_config, msa_dim:int) -> None:
		super(MSAColumnAttentionOpt, self).__init__(config, global_config, msa_dim)
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.attn = AttentionOpt(config, global_config, msa_dim, msa_dim, msa_dim)

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_column'

		msa_act = msa_act.transpose(-2, -3)
		msa_mask = msa_mask.transpose(-1, -2)
		bias = (1e9 * (msa_mask.to(dtype=msa_act.dtype)-1.0))[:,None,None,:]
		assert bias.ndimension() == 4

		msa_act = self.query_norm(msa_act)
		# msa_act = self.attn(msa_act, msa_act, bias)
		msa_act = inference_subbatch(	self.attn, self.global_config.subbatch_size, 
							batched_args=[msa_act, msa_act, bias],
							nonbatched_args=[None],
							low_memory=(not is_training))

		msa_act = msa_act.transpose(-2, -3)
		return msa_act

class MSAColumnGlobalAttentionOpt(MSAColumnGlobalAttention):
	"""
	Optimized MSAColumnGlobalAttention
	"""
	def __init__(self, config, global_config, msa_dim:int) -> None:
		super(MSAColumnGlobalAttentionOpt, self).__init__(config, global_config, msa_dim)
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.attn = GlobalAttentionOpt(config, global_config, msa_dim, msa_dim, msa_dim)

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_column'

		msa_act = msa_act.transpose(-2, -3)
		msa_mask = msa_mask.transpose(-1, -2)
		bias = (1e9 * (msa_mask.to(dtype=msa_act.dtype)-1.0))[:,None,None,:]
		msa_mask = msa_mask.unsqueeze(dim=-1)
		assert bias.ndimension() == 4

		msa_act = self.query_norm(msa_act)
		# msa_act = self.attn(msa_act, msa_act, msa_mask, bias)
		msa_act = inference_subbatch(	self.attn, self.global_config.subbatch_size, 
							batched_args=[msa_act, msa_act, msa_mask, bias],
							nonbatched_args=[],
							low_memory=(not is_training))
		msa_act = msa_act.transpose(-2, -3)
		return msa_act