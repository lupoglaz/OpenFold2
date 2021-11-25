import torch
from torch import nn
from typing import Tuple

class Attention:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L546
	"""
	def __init__(self, config, global_config, output_dim) -> None:
		super(Attention, self).__init__()
		self.config = config
		self.global_config = global_config
		self.output_dim = output_dim

		all_key_dim = self.config.key_dim
		all_value_dim = self.config.value_dim
		self.num_head = self.config.num_head
		assert all_key_dim % self.num_head == 0
		assert all_value_dim % self.num_head == 0
		self.key_dim = all_key_dim // num_head
		self.value_dim = all_value_dim // num_head

		self.q_weights = nn.Parameter(torch.zeros(all_key_dim, self.num_head, self.key_dim))
		self.k_weights = nn.Parameter(torch.zeros(all_value_dim, self.num_head, self.key_dim))
		self.v_weights = nn.Parameter(torch.zeros(all_value_dim, self.num_head, self.value_dim))
		self.o_weights = nn.Parameter(torch.zeros(self.num_head, self.value_dim, self.output_dim))
		self.o_bias = nn.Parameter(torch.zeros(self.output_dim))

		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()

		if self.global_config.zero_init:
			pass
		else:
			NotImplementedError()

		if self.config.gating:
			self.gating_w = nn.Parameter(torch.zeros(all_key_dim, self.num_head, self.value_dim))
			self.gating_b = nn.Parameter(torch.ones(self.num_head, self.value_dim))

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

class MSARowAttentionWithPairBias:
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
		self.attn = Attention(config, global_config, msa_dim)

	def forward(self, msa_act:torch.Tensor, msa_mask:torch.Tensor, pair_act:torch.Tensor, is_training:bool=False):
		assert msa_act.ndimension() == 3
		assert msa_mask.ndimension() == 2
		assert self.config.orientation == 'per_row'

		bias = (1e9 * (msa_mask-1.0))[:,None,None,:]
		msa_act = self.query_norm(msa_act)
		pair_act = self.feat_2d_norm(pair_act)
		nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
		msa_act = self.attn(msa_act, msa_act, bias, nonbatched_bias)

		return msa_act
