import torch
from torch import nn
from typing import Sequence, Tuple
from alphafold.Model.Utils.weights_loading import load_params

class Attention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L546
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int) -> None:
		super(Attention, self).__init__()
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

		self.q_weights = nn.Parameter(torch.zeros(all_key_dim, self.num_head, self.key_dim))
		self.k_weights = nn.Parameter(torch.zeros(all_value_dim, self.num_head, self.key_dim))
		self.v_weights = nn.Parameter(torch.zeros(all_value_dim, self.num_head, self.value_dim))
		self.o_weights = nn.Parameter(torch.zeros(self.num_head, self.value_dim, self.output_dim))
		self.o_bias = nn.Parameter(torch.zeros(self.output_dim))

		self.softmax = nn.Softmax(dim=-1)
		self.sigmoid = nn.Sigmoid()

		if self.global_config.zero_init:
			pass
		else:
			NotImplementedError()

		if self.config.gating:
			self.gating_w = nn.Parameter(torch.zeros(all_key_dim, self.num_head, self.value_dim))
			self.gating_b = nn.Parameter(torch.ones(self.num_head, self.value_dim))

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.q_weights, self.k_weights, self.v_weights, self.o_weights, self.o_bias]
		names=['query_w', 'key_w', 'value_w', 'output_w', 'output_b']
		if self.config.gating:
			modules.extend([self.gating_w, self.gating_b])
			names.extend(['gating_w', 'gating_b'])
		load_params(data, modules, names, rel_path=rel_path, ind=ind)

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
		assert self.key_dim * self.num_head == q_data.size(-1)
		assert self.value_dim * self.num_head == m_data.size(-1)

		q = torch.einsum('bqa,ahc->bqhc', q_data, self.q_weights) * self.key_dim **(-0.5)
		k = torch.einsum('bka,ahc->bkhc', m_data, self.k_weights)
		v = torch.einsum('bka,ahc->bkhc', m_data, self.v_weights)
		logits = torch.einsum('bqhc,bkhc->bhqk', q, k) + bias
		if not(nonbatched_bias is None):
			logits += nonbatched_bias.unsqueeze(dim=0)
		weights = self.softmax(logits)
		weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v)
		
		if self.config.gating:
			gate_values = torch.einsum('bqc,chv->bqhv', q_data, self.gating_w) + self.gating_b
			gate_values = self.sigmoid(gate_values)
			weighted_avg *= gate_values
		
		output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.o_weights) + self.o_bias
		return output

class MSARowAttentionWithPairBias(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L720
	"""
	def __init__(self, config, global_config, pair_dim:int, msa_dim:int) -> None:
		super(MSARowAttentionWithPairBias, self).__init__()
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.feat_2d_norm = nn.LayerNorm(pair_dim)
		self.feat_2d_weights = nn.Parameter(torch.randn(pair_dim, config.num_head))
		self.attn = Attention(config, global_config, msa_dim, msa_dim, msa_dim)

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
		print(f'Loading feat_2d_weights: {d.shape} -> {self.feat_2d_weights.size()}')
		self.feat_2d_weights.data.copy_(torch.from_numpy(d))
		
		self.attn.load_weights_from_af2(data, rel_path=f'{rel_path}/attention', ind=ind)
		

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, pair_act:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_row'

		bias = (1e9 * (msa_mask.to(dtype=torch.float32)-1.0))[:,None,None,:]
		msa_act = self.query_norm(msa_act)
		pair_act = self.feat_2d_norm(pair_act)
		nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
		msa_act = self.attn(msa_act, msa_act, bias, nonbatched_bias)

		return msa_act

class MSAColumnAttention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L787
	"""
	def __init__(self, config, global_config, msa_dim:int) -> None:
		super(MSAColumnAttention, self).__init__()
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.attn = Attention(config, global_config, msa_dim, msa_dim, msa_dim)
	
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
		
		self.attn.load_weights_from_af2(data, rel_path=f'{rel_path}/attention', ind=ind)

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_column'

		msa_act = msa_act.transpose(-2, -3)
		msa_mask = msa_mask.transpose(-1, -2)
		bias = (1e9 * (msa_mask.to(dtype=torch.float32)-1.0))[:,None,None,:]
		assert bias.ndimension() == 4

		msa_act = self.query_norm(msa_act)
		msa_act = self.attn(msa_act, msa_act, bias)
		msa_act = msa_act.transpose(-2, -3)
		return msa_act
		

class GlobalAttention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L630
	"""
	def __init__(self, config, global_config, output_dim:int, key_dim:int, value_dim:int) -> None:
		super(GlobalAttention, self).__init__()
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

		self.q_weights = nn.Parameter(torch.zeros(all_key_dim, self.num_head, self.key_dim))
		self.k_weights = nn.Parameter(torch.zeros(all_value_dim, self.key_dim))
		self.v_weights = nn.Parameter(torch.zeros(all_value_dim, self.value_dim))
		self.o_weights = nn.Parameter(torch.zeros(self.num_head, self.value_dim, self.output_dim))
		self.o_bias = nn.Parameter(torch.zeros(self.output_dim))

		self.softmax = nn.Softmax(dim=-1)
		self.sigmoid = nn.Sigmoid()

		if self.global_config.zero_init:
			pass
		else:
			NotImplementedError()

		if self.config.gating:
			self.gating_w = nn.Parameter(torch.zeros(all_key_dim, self.num_head, self.value_dim))
			self.gating_b = nn.Parameter(torch.ones(self.num_head, self.value_dim))
	
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer', ind:int=None):
		modules=[self.q_weights, self.k_weights, self.v_weights, self.o_weights, self.o_bias]
		names=['query_w', 'key_w', 'value_w', 'output_w', 'output_b']
		if self.config.gating:
			modules.extend([self.gating_w, self.gating_b])
			names.extend(['gating_w', 'gating_b'])
		for module, name in zip(modules, names):
			if rel_path is None:
				d = data[f'{name}']
			else:
				if ind is None:
					d = data[f'{rel_path}'][f'{name}']
				else:
					d = data[f'{rel_path}'][f'{name}'][ind,...]
			print(f'Loading {name}: {d.shape} -> {module.size()}')
			module.data.copy_(torch.from_numpy(d))

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
		assert self.key_dim * self.num_head == q_data.size(-1)
		assert self.value_dim * self.num_head == m_data.size(-1)

		v = torch.einsum('bka,ac->bkc', m_data, self.v_weights)

		q_avg = self.mask_mean(q_mask, q_data, dims=[1])

		q = torch.einsum('ba,ahc->bhc', q_avg, self.q_weights) * self.key_dim **(-0.5)
		k = torch.einsum('bka,ac->bkc', m_data, self.k_weights)
		bias = (1e9*(q_mask[:,None,:,0].to(dtype=torch.float32) - 1.0))

		logits = torch.einsum('bhc,bkc->bhk', q, k) + bias
		weights = self.softmax(logits)
		weighted_avg = torch.einsum('bhk,bkc->bhc', weights, v)
		
		if self.config.gating:
			gate_values = torch.einsum('bqc,chv->bqhv', q_data, self.gating_w) + self.gating_b
			gate_values = self.sigmoid(gate_values)
			weighted_avg = weighted_avg[:, None] * gate_values
			output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.o_weights) + self.o_bias
		else:
			output = torch.einsum('bhc,hco->bo', weighted_avg, self.o_weights) + self.o_bias
			output = output[:, None]
		return output

class MSAColumnGlobalAttention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L842
	"""
	def __init__(self, config, global_config, msa_dim:int) -> None:
		super(MSAColumnGlobalAttention, self).__init__()
		self.config = config
		self.global_config = global_config
		self.query_norm = nn.LayerNorm(msa_dim)
		self.attn = GlobalAttention(config, global_config, msa_dim, msa_dim, msa_dim)
	
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
		
		self.attn.load_weights_from_af2(data, rel_path=f'{rel_path}/attention', ind=ind)

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_column'

		msa_act = msa_act.transpose(-2, -3)
		msa_mask = msa_mask.transpose(-1, -2)
		bias = (1e9 * (msa_mask.to(dtype=torch.float32)-1.0))[:,None,None,:]
		msa_mask = msa_mask.unsqueeze(dim=-1)
		assert bias.ndimension() == 4

		msa_act = self.query_norm(msa_act)
		msa_act = self.attn(msa_act, msa_act, msa_mask, bias)
		msa_act = msa_act.transpose(-2, -3)
		return msa_act