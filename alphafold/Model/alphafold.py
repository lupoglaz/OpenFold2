import torch
from torch import nn
from alphafold.Model.msa import *
from alphafold.Model.spatial import *
from alphafold.Model.embedders import *
from typing import Mapping
import functools

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
			p = torch.zeros(*tuple(shape), dtype=tensor.dtype, device=tensor.device).fill_(keep_rate)
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

class EvoformerIteration(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1561
	"""
	def __init__(self, config, global_config, msa_dim: int, pair_dim: int, is_extra_msa: bool) -> None:
		super(EvoformerIteration, self).__init__()
		self.config = config
		self.global_config = global_config
		self.is_extra_msa = is_extra_msa

		self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(config.msa_row_attention_with_pair_bias, global_config, pair_dim, msa_dim)
		
		if not is_extra_msa:
			self.msa_column_attention = MSAColumnAttention(config.msa_column_attention, global_config, msa_dim)
		else:
			self.msa_column_attention = MSAColumnGlobalAttention(config.msa_column_attention, global_config, msa_dim)

		self.msa_transition = Transition(config.msa_transition, global_config, msa_dim)
		self.outer_product_mean = OuterProductMean(config.outer_product_mean, global_config, pair_dim, msa_dim)
		self.triangle_multiplication_outgoing = TriangleMultiplication(config.triangle_multiplication_outgoing, global_config, pair_dim)
		self.triangle_multiplication_incoming = TriangleMultiplication(config.triangle_multiplication_incoming, global_config, pair_dim)
		self.triangle_attention_starting_node = TriangleAttention(config.triangle_attention_starting_node, global_config, pair_dim)
		self.triangle_attention_ending_node = TriangleAttention(config.triangle_attention_ending_node, global_config, pair_dim)
		self.pair_transition = Transition(config.pair_transition, global_config, pair_dim)

	def load_weights_from_af2(self, data, rel_path: str='evoformer_iteration'):
		self.msa_row_attention_with_pair_bias.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_row_attention_with_pair_bias')
		if not self.is_extra_msa:
			self.msa_column_attention.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_column_attention')
		else:
			self.msa_column_attention.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_column_global_attention')
		self.msa_transition.load_weights_from_af2(data, rel_path=f'{rel_path}/msa_transition')
		self.outer_product_mean.load_weights_from_af2(data, rel_path=f'{rel_path}/outer_product_mean')
		self.triangle_multiplication_outgoing.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_multiplication_outgoing')
		self.triangle_multiplication_incoming.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_multiplication_incoming')
		self.triangle_attention_starting_node.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_attention_starting_node')
		self.triangle_attention_ending_node.load_weights_from_af2(data, rel_path=f'{rel_path}/triangle_attention_ending_node')
		self.pair_transition.load_weights_from_af2(data, rel_path=f'{rel_path}/pair_transition')

	def forward(self, 	activations:Mapping[str, torch.Tensor], 
						masks:Mapping[str, torch.Tensor], 
						is_training: bool=False) -> Mapping[str, torch.Tensor]:
		msa_act, pair_act = activations['msa'], activations['pair']
		msa_mask, pair_mask = masks['msa'], masks['pair']
		DO = functools.partial(dropout_wrapper, is_training=is_training, global_config=self.global_config)

		msa_act = DO(self.msa_row_attention_with_pair_bias, msa_act, msa_mask, pair_act=pair_act)
		msa_act = DO(self.msa_column_attention, msa_act, msa_mask)
		msa_act = DO(self.msa_transition, msa_act, msa_mask)
		pair_act = DO(self.outer_product_mean, msa_act, msa_mask, output_act=pair_act)
		pair_act = DO(self.triangle_multiplication_outgoing, pair_act, pair_mask)
		pair_act = DO(self.triangle_multiplication_incoming, pair_act, pair_mask)
		pair_act = DO(self.triangle_attention_starting_node, pair_act, pair_mask)
		pair_act = DO(self.triangle_attention_ending_node, pair_act, pair_mask)
		pair_act = DO(self.pair_transition, pair_act, pair_mask)

		return {'msa': msa_act, 'pair': pair_act}
		
class EmbeddingsAndEvoformer(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1561
	"""
	def __init__(self, config, global_config, msa_dim: int, target_dim: int, is_extra_msa: bool) -> None:
		super(EmbeddingsAndEvoformer, self).__init__()
		self.config = config
		self.global_config = global_config
		self.input_emb = InputEmbeddings(config, global_config, msa_dim=msa_dim, target_dim=target_dim)
		self.recycle_emb = RecycleEmbedding(config, global_config)
		self.extra_msa_emb = ExtraMSAEmbedding(config, global_config, msa_dim=msa_dim)
		EvoformerIteration()

	def forward(self, batch: Mapping[str, torch.Tensor], is_training:bool=False, safe_key=None):
		pass

class AlphaFoldIteration(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
	
	def forward(self, ensembled_batch, non_ensembled_batch, is_training, 
				compute_loss=False, ensemble_representations=False, return_representations=False):
		raise Exception(NotImplemented)

class AlphaFold(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.impl = AlphaFoldIteration(self.config)

	def forward(self, batch, is_training, 
				compute_loss=False, ensemble_representations=False, return_representations=False):
		raise Exception(NotImplemented)
		# print(batch['aatype'])
		# batch_size, num_residues = batch['aatype']