import torch
from torch import nn

from alphafold.Model.Opt.batch_msa import *
from alphafold.Model.Opt.batch_spatial import *
from alphafold.Model.linear import Linear

from alphafold.Model.embedders import *


def dropout_wrapper(module:nn.Module, input_act:torch.Tensor, mask:torch.Tensor,  
					global_config, 
					safe_key:int=None,
					output_act:torch.Tensor=None, 
					is_training:bool=False, **kwargs):
	""""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L63
	"""
	def apply_dropout(*, tensor:torch.Tensor, rate:float, is_training:bool, safe_key:int=None, broadcast_dim:int=None):
		"""
		https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L50
		"""
		if is_training and rate>0.0:
			shape = list(tensor.shape)
			if not(broadcast_dim is None):
				shape[broadcast_dim] = 1
			keep_rate = 1.0 - rate
			p = torch.zeros_like(tensor).fill_(keep_rate)
			keep = torch.bernoulli(p)
			return keep * tensor / keep_rate
		else:
			return tensor
	if output_act is None:
		output_act = input_act
	residual = module(input_act, mask, is_training=is_training, **kwargs)
	dropout_rate = 0.0 if global_config.deterministic else module.config.dropout_rate

	if module.config.shared_dropout:
		if module.config.orientation == 'per_row':
			broadcast_dim = 0
		else:
			broadcast_dim = 1
	else:
		broadcast_dim = None
	
	residual = apply_dropout(tensor=residual, safe_key=safe_key, rate=dropout_rate, is_training=is_training, broadcast_dim=broadcast_dim)
	new_act = output_act + residual
	return new_act

class EvoformerIterationFFB(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1561
	"""
	def __init__(self, config, global_config, msa_dim: int, pair_dim: int, is_extra_msa: bool) -> None:
		super(EvoformerIterationFFB, self).__init__()
		self.config = config
		self.global_config = global_config
		self.is_extra_msa = is_extra_msa
		
		if is_extra_msa:
			self.msa_column_attention = MSAColumnGlobalAttentionOptB(config.msa_column_attention, global_config, msa_dim)
		else:
			self.msa_column_attention = MSAColumnAttentionFFB(config.msa_column_attention, global_config, msa_dim)

		self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBiasFFB(config.msa_row_attention_with_pair_bias, global_config, pair_dim, msa_dim)		
		self.msa_transition = TransitionFFB(config.msa_transition, global_config, msa_dim)
		self.outer_product_mean = OuterProductMeanFFB(config.outer_product_mean, global_config, pair_dim, msa_dim)
		self.triangle_multiplication_outgoing = TriangleMultiplicationFFB(config.triangle_multiplication_outgoing, global_config, pair_dim)
		self.triangle_multiplication_incoming = TriangleMultiplicationFFB(config.triangle_multiplication_incoming, global_config, pair_dim)
		self.triangle_attention_starting_node = TriangleAttentionFFB(config.triangle_attention_starting_node, global_config, pair_dim)
		self.triangle_attention_ending_node = TriangleAttentionFFB(config.triangle_attention_ending_node, global_config, pair_dim)
		self.pair_transition = TransitionFFB(config.pair_transition, global_config, pair_dim)

	def load_weights_from_af2(self, data, rel_path: str='evoformer_iteration', ind:int=None):
		self.msa_row_attention_with_pair_bias.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_row_attention_with_pair_bias', ind=ind)
		if not self.is_extra_msa:
			self.msa_column_attention.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_column_attention', ind=ind)
		else:
			self.msa_column_attention.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_column_global_attention', ind=ind)
		self.msa_transition.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_transition', ind=ind)
		self.outer_product_mean.load_weights_from_af2(data, rel_path=f'{rel_path}/outer_product_mean', ind=ind)
		self.triangle_multiplication_outgoing.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_multiplication_outgoing', ind=ind)
		self.triangle_multiplication_incoming.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_multiplication_incoming', ind=ind)
		self.triangle_attention_starting_node.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_attention_starting_node', ind=ind)
		self.triangle_attention_ending_node.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_attention_ending_node', ind=ind)
		self.pair_transition.load_weights_from_af2(data, rel_path=f'{rel_path}/pair_transition', ind=ind)

	def forward(self, msa_act:torch.Tensor, pair_act:torch.Tensor, msa_mask:torch.Tensor, pair_mask:torch.Tensor, 
					is_training: bool=False, low_memory:bool=False) -> Mapping[str, torch.Tensor]:
		
		#MSA stack
		msa_act = self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act=pair_act, is_training=is_training, low_memory=low_memory)
		if self.is_extra_msa:
			#Using Opt module + original dropout and skip connection
			msa_act = dropout_wrapper(self.msa_column_attention, msa_act, msa_mask, is_training=is_training, low_memory=low_memory, 
										global_config=self.global_config)
		else:
			#Using FastFold module with dropout and skip
			msa_act = self.msa_column_attention(msa_act, msa_mask, is_training=is_training, low_memory=low_memory)
		#MSA - > MSA
		msa_act = self.msa_transition(msa_act, msa_mask, is_training=is_training, low_memory=low_memory)
		#MSA - > pair
		pair_act = pair_act + self.outer_product_mean(msa_act, msa_mask, is_training=is_training, low_memory=low_memory)
		#Pair stack
		pair_act = self.triangle_multiplication_outgoing(pair_act, pair_mask, is_training=is_training)
		pair_act = self.triangle_multiplication_incoming(pair_act, pair_mask, is_training=is_training)
		pair_act = self.triangle_attention_starting_node(pair_act, pair_mask, is_training=is_training, low_memory=low_memory)
		pair_act = self.triangle_attention_ending_node(pair_act, pair_mask, is_training=is_training, low_memory=low_memory)
		
		#Pair - > Pair
		pair_act = self.pair_transition(pair_act, pair_mask, is_training=is_training, low_memory=low_memory)
		return msa_act, pair_act