import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.Utils.weights_loading import load_params
from alphafold.Model.linear import Linear
from alphafold.Model.Utils.tensor_utils import permute_final_dims, flatten_final_dims
import math

from alphafold.Model.msa import MSAColumnAttention, MSAColumnGlobalAttention
from alphafold.Model.Opt.msa import AttentionOpt
from alphafold.Model.Opt.mapping import inference_subbatch
from einops import rearrange
from FastFold.Kernel import scale_mask_softmax, scale_mask_bias_softmax, bias_sigmod_ele, bias_dropout_add
from FastFold.Kernel import LayerNorm as LayerNormFF


class AttentionFlash(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L546
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L167
	and 
	https://github.com/hpcaitech/FastFold/blob/16d10d6a7852520601fa35cf1b15a8d89668c59b/fastfold/model/ops.py#L123
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int,
				last_bias_fuse:bool=False) -> None:
		super(AttentionFlash, self).__init__()
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