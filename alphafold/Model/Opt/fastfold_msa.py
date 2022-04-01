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


class AttentionFF(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L546
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L167
	and 
	https://github.com/hpcaitech/FastFold/blob/16d10d6a7852520601fa35cf1b15a8d89668c59b/fastfold/model/ops.py#L123
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int,
				last_bias_fuse:bool=False) -> None:
		super(AttentionFF, self).__init__()
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

		# self.scaling = (1./math.sqrt(self.key_dim))
		self.scaling = 0.0

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
			bias: [batch_size, num_queries, num_keys]
			nonbatched_bias: [num_queries, num_keys]
		Returns:
			[batch_size, num_queries, output_dim]
		"""
		qkv = self.qkv_weights(in_data).chunk(3, dim=-1)
		q, k, v = map(lambda t: rearrange(t, 'b1 n (h d) -> b1 h n d', h=self.num_head), qkv)

		logits = torch.matmul(q, k.transpose(-1,-2))
		
		if not(nonbatched_bias is None):
			nonbatched_bias = rearrange(nonbatched_bias, 'b q k h -> b h q k')
			# print('FF nonbbias:', nonbatched_bias.size())
			# print('FF logits:', logits.size())
			weights = scale_mask_bias_softmax(logits.unsqueeze(1), mask, nonbatched_bias, self.scaling).squeeze(1)
			# print('FF weights:', weights.size())
			# return weights
		else:
			#head should be 3rd dimension
			weights = scale_mask_softmax(logits.unsqueeze(1), mask, self.scaling).squeeze(1)
		
		weighted_avg = torch.matmul(weights, v)
		# print('FF weighted_avg:', weighted_avg.size())
		weighted_avg = rearrange(weighted_avg, 'b1 h n d -> b1 n (h d)')

		if self.config.gating:
			gate_values = self.gating_linear(in_data)
			weighted_avg = bias_sigmod_ele(gate_values, self.gating_bias, weighted_avg)
		
		output = self.o_linear(weighted_avg)
		
		return output

class MSARowAttentionWithPairBiasFF(nn.Module):
	"""
	Optimized MSARowAttentionWithPairBias
	"""
	def __init__(self, config, global_config, pair_dim:int, msa_dim:int) -> None:
		super(MSARowAttentionWithPairBiasFF, self).__init__()
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.feat_2d_norm = nn.LayerNorm(pair_dim)
		self.feat_2d_weights = Linear(pair_dim, config.num_head, use_bias=False, initializer='normal')
		self.attn = AttentionFF(config, global_config, msa_dim, msa_dim, msa_dim, last_bias_fuse=True)
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
		assert msa_act_raw.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_row'
		
		msa_act = self.query_norm(msa_act_raw)
		pair_act = self.feat_2d_norm(pair_act)
		nonbatched_bias = self.feat_2d_weights(pair_act).unsqueeze(0)
		msa_act = inference_subbatch(self.attn, self.global_config.subbatch_size, 
									batched_args=[msa_act, msa_mask],
									nonbatched_args=[nonbatched_bias],
									low_memory=(not is_training))
		#!!! Bias dropout add
		# msa_act = msa_act.unsqueeze(dim=1)
		# msa_act_raw = msa_act_raw.unsqueeze(dim=1)
		# dropout_mask = torch.ones_like(msa_act, device=msa_act.device, dtype=msa_act.dtype)
		# return bias_dropout_add(msa_act, self.out_bias, dropout_mask, msa_act_raw, prob=0.0, training=self.training).squeeze(dim=1)
		
		return msa_act

class MSAColumnAttentionFF(MSAColumnAttention):
	"""
	Optimized MSAColumnAttention
	"""
	def __init__(self, config, global_config, msa_dim:int) -> None:
		super(MSAColumnAttentionFF, self).__init__(config, global_config, msa_dim)
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.attn = AttentionFF(config, global_config, msa_dim, msa_dim, msa_dim)

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_column'
		
		msa_act = msa_act.transpose(-2, -3)
		msa_mask = msa_mask.transpose(-1, -2)
		
		msa_act = self.query_norm(msa_act)
		
		msa_act = inference_subbatch(	self.attn, self.global_config.subbatch_size, 
							batched_args=[msa_act, msa_mask],
							nonbatched_args=[None],
							low_memory=(not is_training))

		msa_act = msa_act.transpose(-2, -3)
		return msa_act

