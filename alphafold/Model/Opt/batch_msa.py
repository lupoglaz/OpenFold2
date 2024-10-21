import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.Utils.weights_loading import load_params
from alphafold.Model.linear import Linear
from alphafold.Model.Utils.tensor_utils import permute_final_dims, flatten_final_dims
import math

from alphafold.Model.msa import MSAColumnAttention, MSAColumnGlobalAttention
from alphafold.Model.Opt.msa import AttentionOpt
from .mapping import inference_subbatch
from einops import rearrange
from FastFold.Kernel import scale_mask_softmax, scale_mask_bias_softmax, bias_sigmod_ele, bias_dropout_add
from FastFold.Kernel import LayerNorm as LayerNormFF


class AttentionFFB(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L546
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L167
	and 
	https://github.com/hpcaitech/FastFold/blob/16d10d6a7852520601fa35cf1b15a8d89668c59b/fastfold/model/ops.py#L123
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int,
				last_bias_fuse:bool=False) -> None:
		super(AttentionFFB, self).__init__()
		self.config = config
		self.global_config = global_config
		self.output_dim = output_dim
		self.last_bias_fuse = last_bias_fuse

		all_key_dim = key_dim
		all_value_dim = value_dim
		self.num_head = self.config.num_head
		assert all_key_dim == all_value_dim
		assert all_key_dim % self.num_head == 0
		assert all_value_dim % self.num_head == 0
		self.key_dim = all_key_dim // self.num_head
		self.value_dim = all_value_dim // self.num_head

		self.scaling = (1./math.sqrt(self.key_dim))
		# self.scaling = 0.0

		self.qkv_weights = Linear(all_key_dim, 3*all_key_dim, use_bias=False, initializer='glorot')
		self.o_linear = Linear(all_value_dim, self.output_dim, initializer='final', use_bias=(not last_bias_fuse))

		self.softmax = nn.Softmax(dim=-1)
		self.sigmoid = nn.Sigmoid()

		if self.config.gating:
			self.gating_linear = Linear(all_key_dim, all_value_dim, initializer='gating', use_bias=False)
			self.gating_bias = nn.Parameter(torch.ones(self.num_head * self.key_dim))

		
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[None, None, None, self.o_linear]
		param_names=[('query_w',None), ('key_w',None), ('value_w',None), ('output_w', 'output_b')]
		if self.config.gating:
			modules.extend([self.gating_linear])
			param_names.extend([('gating_w', 'gating_b')])
		qkv_weights = []
		fused_bias = None
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
			if name_w in ('query_w', 'key_w', 'value_w', 'gating_w'):
				print(f'Loading {name_w}: {data_weight.shape} -> special')
				data_weight = torch.from_numpy(data_weight).reshape(self.value_dim*self.num_head, self.key_dim*self.num_head).transpose(-1,-2)
			else:
				print(f'Loading {name_w}: {data_weight.shape} -> {module.weight.size()}')
				data_weight = torch.from_numpy(data_weight).permute(2,0,1).reshape(module.weight.shape)
			if name_w in ('query_w', 'key_w', 'value_w'):
				qkv_weights.append(data_weight)
			else:
				module.weight.data.copy_(data_weight)
			if not(name_b is None):
				if name_b in ('gating_b'):
					print(f'Loading {name_b}: {data_bias.shape} -> {self.gating_bias.data.size()}')
					self.gating_bias.data.copy_(torch.from_numpy(data_bias).reshape(self.gating_bias.data.shape))
				elif self.last_bias_fuse and name_b == 'output_b':
					print(f'Sending {name_b}: {data_bias.shape} -> previous module')
					fused_bias = torch.from_numpy(data_bias)
				else:
					print(f'Loading {name_b}: {data_bias.shape} -> {module.bias.size()}')
					module.bias.data.copy_(torch.from_numpy(data_bias).reshape(module.bias.shape))
		
		qkv_weights = torch.cat(qkv_weights, dim=0)
		print(f'Loading qkv_weights: {qkv_weights.shape} -> {self.qkv_weights.weight.size()}')
		self.qkv_weights.weight.data.copy_(qkv_weights)
		return fused_bias

	def forward(self, in_data: torch.Tensor, mask: torch.Tensor, nonbatched_bias: torch.Tensor=None) -> torch.Tensor:
		"""
		Arguments: 
			q_data: [batch_size, num_queries, querry_dim]
			m_data: [batch_size, num_keys, value_dim]
			bias: [batch_size, num_queries*num_keys]
			nonbatched_bias: [batch_size, num_querries, num_keys, hum_head]
		Returns:
			[batch_size, num_queries, output_dim]
		"""
		
		qkv = self.qkv_weights(in_data).chunk(3, dim=-1)
		q, k, v = map(lambda t: rearrange(t, 'b0 b1 n (h d) -> b0 b1 h n d', h=self.num_head), qkv)
		logits = torch.matmul(q, k.transpose(-1,-2))
				
		if not(nonbatched_bias is None):
			nonbatched_bias = rearrange(nonbatched_bias, 'b q k h -> b h q k')
			weights = scale_mask_bias_softmax(logits, mask, nonbatched_bias, self.scaling)
			print(logits.size(), mask.size(), nonbatched_bias.size(), weights.size())
			# print(weights)
		else:
			#head should be 3rd dimension
			weights = scale_mask_softmax(logits, mask, self.scaling)
			print(logits.size(), mask.size(), weights.size())

		weighted_avg = torch.matmul(weights, v)
		weighted_avg = rearrange(weighted_avg, 'b0 b1 h n d -> b0 b1 n (h d)')

		if self.config.gating:
			gate_values = self.gating_linear(in_data)
			weighted_avg = bias_sigmod_ele(gate_values, self.gating_bias, weighted_avg)
		
		output = self.o_linear(weighted_avg)
		
		return output


class MSARowAttentionWithPairBiasFFB(nn.Module):
	"""
	Optimized MSARowAttentionWithPairBias
	Combines dropout, residual connection and MSARowAttentionWithPairBias
	"""
	def __init__(self, config, global_config, pair_dim:int, msa_dim:int) -> None:
		super(MSARowAttentionWithPairBiasFFB, self).__init__()
		self.config = config
		self.global_config = global_config
		self.query_norm = LayerNormFF(msa_dim)
		self.feat_2d_norm = LayerNormFF(pair_dim)
		self.feat_2d_weights = Linear(pair_dim, config.num_head, use_bias=False, initializer='normal')
		self.attn = AttentionFFB(config, global_config, msa_dim, msa_dim, msa_dim, last_bias_fuse=True)
		self.out_bias = nn.parameter.Parameter(torch.zeros(msa_dim))

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
		
		fused_bias = self.attn.load_weights_from_af2(data, rel_path=f'{rel_path}/attention', ind=ind)
		self.out_bias.data.copy_(fused_bias.reshape(self.out_bias.shape))
		

	def forward(self, msa_act_raw:torch.Tensor, msa_mask:torch.Tensor, pair_act:torch.Tensor, is_training:bool=False):
		assert msa_act_raw.ndimension() == 4
		assert msa_mask.ndimension() == 3
		assert self.config.orientation == 'per_row'
		
		msa_act = self.query_norm(msa_act_raw)
		pair_act = self.feat_2d_norm(pair_act)
		nonbatched_bias = self.feat_2d_weights(pair_act)
		
		msa_act = inference_subbatch(self.attn, self.global_config.subbatch_size, 
									batched_args=[msa_act, msa_mask],
									nonbatched_args=[nonbatched_bias],
									low_memory=(not is_training))

		dropout_mask = torch.ones_like(msa_act, device=msa_act.device, dtype=msa_act.dtype)
		return bias_dropout_add(msa_act, self.out_bias, dropout_mask, msa_act_raw, prob=self.config.dropout_rate, training=is_training)
		

class MSAColumnAttentionFFB(MSAColumnAttention):
	"""
	Optimized MSAColumnAttention
	"""
	def __init__(self, config, global_config, msa_dim:int) -> None:
		super(MSAColumnAttentionFFB, self).__init__(config, global_config, msa_dim)
		self.config = config
		self.global_config = global_config
		self.query_norm = LayerNormFF(msa_dim)
		self.attn = AttentionFFB(config, global_config, msa_dim, msa_dim, msa_dim)

	def forward(self, msa_act_raw:torch.Tensor, msa_mask:torch.Tensor, is_training:bool=False):
		assert msa_act_raw.ndimension() == 4
		assert msa_mask.ndimension() == 3
		assert self.config.orientation == 'per_column'
		
		msa_act = msa_act_raw.transpose(-2, -3)
		msa_mask = msa_mask.transpose(-1, -2)
		
		msa_act = self.query_norm(msa_act)
		
		msa_act = inference_subbatch(self.attn, self.global_config.subbatch_size, 
							batched_args=[msa_act, msa_mask],
							nonbatched_args=[None],
							low_memory=(not is_training))

		msa_act = msa_act.transpose(-2, -3)
		return msa_act + msa_act_raw


class GlobalAttentionOptB(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L630
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L300
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int) -> None:
		super(GlobalAttentionOptB, self).__init__()
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

		q_avg = self.mask_mean(q_mask, q_data, dims=[2])
		q = self.q_weights(q_avg) * (1.0/math.sqrt(self.key_dim))
		q = flat_head(q)

		bias = (1e9*(q_mask[:,:,None,:,0] - 1.0))
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

class MSAColumnGlobalAttentionOptB(MSAColumnGlobalAttention):
	"""
	Optimized MSAColumnGlobalAttention
	"""
	def __init__(self, config, global_config, msa_dim:int) -> None:
		super(MSAColumnGlobalAttentionOptB, self).__init__(config, global_config, msa_dim)
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.attn = GlobalAttentionOptB(config, global_config, msa_dim, msa_dim, msa_dim)

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 4
		assert msa_mask.ndimension() == 3
		assert self.config.orientation == 'per_column'

		msa_act = msa_act.transpose(-2, -3)
		msa_mask = msa_mask.transpose(-1, -2)
		bias = (1e9 * (msa_mask.to(dtype=msa_act.dtype)-1.0))[:,:,None,None,:]
		msa_mask = msa_mask.unsqueeze(dim=-1)
		assert bias.ndimension() == 5

		msa_act = self.query_norm(msa_act)
		# msa_act = self.attn(msa_act, msa_act, msa_mask, bias)
		msa_act = inference_subbatch(	self.attn, self.global_config.subbatch_size, 
										batched_args=[msa_act, msa_act, msa_mask, bias],
										nonbatched_args=[],
										low_memory=(not is_training))
		msa_act = msa_act.transpose(-2, -3)
		return msa_act