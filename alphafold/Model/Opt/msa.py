import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.Utils.weights_loading import load_params
from alphafold.Model.linear import Linear
from alphafold.Model.Utils.tensor_utils import permute_final_dims, flatten_final_dims
import math

from alphafold.Model.msa import MSAColumnAttention, MSAColumnGlobalAttention
from .mapping import inference_subbatch

class AttentionOpt(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L546
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L167
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int,
				q_chunk_size:int=None, kv_chunk_size:int=None) -> None:
		super(AttentionOpt, self).__init__()
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
		self.o_linear = Linear(all_value_dim, self.output_dim, initializer='final')

		self.softmax = nn.Softmax(dim=-1)
		self.sigmoid = nn.Sigmoid()

		if self.config.gating:
			self.gating_linear = Linear(all_key_dim, all_value_dim, initializer='gating')

		#Memory optimization
		assert not((q_chunk_size is None) ^ (kv_chunk_size is None))
		self.q_chunk_size = q_chunk_size
		self.kv_chunk_size = kv_chunk_size
		
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

	def iterative_qkv(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, bias:torch.Tensor, nonbatched_bias:torch.Tensor=None):
		"""
		https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L377

		incorrect, does nothing for the gradient propagation!!!
		"""
		num_q, num_kv = q.shape[-3], k.shape[-3]
		biases = [b.expand(b.shape[:-2] + (q.shape[-2],) + k.shape[-2],) for b in [bias, nonbatched_bias]]
		output = q.new_zeros(q.shape)
		for q_idx in range(0, num_q, self.q_chunk_size):
			q_chunk = q[..., q_idx:q_idx+self.q_chunk_size, :, :]
			big_bias_chunks = [b[...,q_idx:q_idx+self.q_chunk_size, :] for b in biases]

			maxs, weights, values = [], [], []
			for kv_idx in range(0, num_kv, self.kv_chunk_size):
				k_chunk = k[..., kv_idx:kv_idx+self.kv_chunk_size, :, :]
				v_chunk = v[..., kv_idx:kv_idx+self.kv_chunk_size, :, :]
				small_bias_chunks = [b[..., kv_idx:kv_idx+self.kv_chunk_size] for b in biases]
				a = torch.einsum('...qhd,...khd->hqk', q_chunk, k_chunk)
				for b in small_bias_chunks:
					a += b
				a = a.transpose(-2, -3)
				max_a = torch.max(a.detach(), dim=-1, keepdim=True).values
				exp_a = torch.exp(a - max_a)
				exp_v = torch.einsum('...vhf,...qhv->qhf', v_chunk, exp_a)

				maxs.append(max_a.squeeze(dim=-1))
				weights.append(torch.sum(exp_a, dim=-1))
				values.append(exp_v)
			
			chunk_max = torch.stack(maxs, dim=-3)
			chunk_weights = torch.stack(weights, dim=-3)
			chunk_values = torch.stack(values, dim=-4)

			global_max = torch.max(chunk_max, dim=-3, keepdim=True).values
			max_diffs = torch.exp(chunk_max - global_max)
			chunk_values *= max_diffs.unsqueeze(dim=-1)
			chunk_weights *= max_diffs

			all_values = torch.sum(chunk_values, dim=-4)
			all_weights = torch.sum(chunk_weights.unsqueeze(dim=-1), dim=-4)
			output[..., q_idx:q_idx+self.q_chunk_size, :, :] = all_values/all_weights
		return output
		

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
		
		q = self.q_weights(q_data) * 0.0#(1./math.sqrt(self.key_dim))
		q = flat_head(q)

		k = self.k_weights(m_data)
		k = flat_head(k)

		v = self.v_weights(m_data)
		v = flat_head(v)
		
		#Low memory:
		if not(self.q_chunk_size is None):
			weighted_avg = self.iterative_qkv(q, k, v, bias, nonbatched_bias)
		
		#High memory:
		else:
			q = permute_final_dims(q, (1, 0, 2))
			k = permute_final_dims(k, (1, 2, 0))
			# print('Opt q:',q.size())
			# print(q[48,6,0,:10])
			# return q
			
			# print('Opt k:',k.size())
			# print(k[48,6,0,:10])
			# print('Opt bias:',bias.size())
			# print(bias[48,0,0,:10])
			# return torch.matmul(q, k) * math.sqrt(self.key_dim)
			logits = torch.matmul(q, k) + bias
			del q, k
			
			# print('Opt logits:',logits.size())
			# print(logits[48,6,0,:10])

			if not(nonbatched_bias is None):
				# print('Opt nonb bias:', nonbatched_bias.unsqueeze(dim=0).size())
				# print('Opt logits:', logits.size())
				logits += nonbatched_bias.unsqueeze(dim=0)

			weights = self.softmax(logits)
			v = permute_final_dims(v, (1, 0, 2))
			print('Opt weights:',weights.size())
			print(weights[0,0,:4,:4])
			return weights
			# print('Opt bias:', bias.size())
			
			weighted_avg = torch.matmul(weights, v).transpose(-2, -3)

		if self.config.gating:
			gate_values = self.gating_linear(q_data)
			gate_values = self.sigmoid(gate_values)
			gate_values = flat_head(gate_values)
			weighted_avg *= gate_values

		weighted_avg = flatten_final_dims(weighted_avg, 2)
		# print('Opt weighted_avg:', weighted_avg.size())
		# print(weighted_avg[1,1,:10])
		output = self.o_linear(weighted_avg)
		# print('Opt output:', output.size())
		# print(output[1,1,:10])
		
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