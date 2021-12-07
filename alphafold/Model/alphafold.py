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
	def __init__(self, config, global_config, target_dim:int, msa_dim:int, extra_msa_dim:int) -> None:
		super(EmbeddingsAndEvoformer, self).__init__()
		self.config = config
		self.global_config = global_config
		
		self.input_emb = InputEmbeddings(config, global_config, msa_dim=msa_dim, target_dim=target_dim)
		self.recycle_emb = RecycleEmbedding(config, global_config)
		self.extra_msa_emb = ExtraMSAEmbedding(config, global_config, msa_dim=extra_msa_dim)
		self.extra_msa_stack = nn.ModuleList()
		for i in range(self.config.extra_msa_stack_num_block):
			self.extra_msa_stack.append(EvoformerIteration(	config.evoformer, global_config, 
															msa_dim=config.extra_msa_channel, 
															pair_dim=config.pair_channel, 
															is_extra_msa=True))
		self.evoformer_stack = []
		for i in range(self.config.evoformer_num_block):
			self.evoformer_stack.append(EvoformerIteration(	config.evoformer, global_config, 
															msa_dim=config.msa_channel, 
															pair_dim=config.pair_channel, 
															is_extra_msa=False))
		self.single_activations = nn.Linear(config.msa_channel, config.seq_channel)
	
	def load_weights_from_af2(self, data, rel_path: str='evoformer_iteration'):
		self.input_emb.load_weights_from_af2(data, rel_path=f'{rel_path}')
		try:
			self.recycle_emb.load_weights_from_af2(data, rel_path=f'{rel_path}')
		except:
			pass
		self.extra_msa_emb.load_weights_from_af2(data, rel_path=f'{rel_path}')
		for ind, extra_msa_iter in enumerate(self.extra_msa_stack):
			extra_msa_iter.load_weights_from_af2(data, rel_path=f'{rel_path}/extra_msa_stack', ind=ind)
		for ind, evoformer_iter in enumerate(self.evoformer_stack):
			evoformer_iter.load_weights_from_af2(data, rel_path=f'{rel_path}/evoformer_iteration', ind=ind)

		modules=[self.single_activations]
		names=['single_activations']
		for module, name in zip(modules, names):
			w = data[f'{rel_path}/{name}']['weights']
			b = data[f'{rel_path}/{name}']['bias']
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w).transpose(0, 1))
			module.bias.data.copy_(torch.from_numpy(b))
		
	def forward(self, batch: Mapping[str, torch.Tensor], is_training:bool=False, safe_key=None):
		inp_msa_act, inp_pair_act = self.input_emb(batch)

		rec_msa_act, rec_pair_act = self.recycle_emb(batch)
		if not(rec_msa_act is None):
			inp_msa_act = inp_msa_act.index_add(0, 0, rec_msa_act)
		if not(rec_pair_act is None):
			inp_pair_act += rec_pair_act

		if self.config.template.enabled:
			raise Exception(NotImplemented)
		
		mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

		extra_msa_act = self.extra_msa_emb(batch)
		extra_act = {'msa': extra_msa_act, 'pair': inp_pair_act}
		extra_masks = {'msa':batch['extra_msa_mask'], 'pair': mask_2d}
		for extra_msa_iteration in self.extra_msa_stack:
			extra_act = extra_msa_iteration(activations=extra_act, masks=extra_masks, is_training=is_training)

		evoformer_act = {'msa': inp_msa_act, 'pair': extra_act['pair']}
		evoformer_masks = {'msa': batch['msa_mask'], 'pair': mask_2d}
		for evoformer_iteration in self.evoformer_stack:
			evoformer_act = evoformer_iteration(activations=evoformer_act, masks=evoformer_masks, is_training=is_training)

		msa_act = evoformer_act['msa']
		pair_act = evoformer_act['pair']
		single_act = self.single_activations(msa_act[0])
		output = {
			'single': single_act,
			'pair': pair_act,
			'msa': msa_act[:batch['msa_feat'].size(0), :, :],
			'msa_first_row': msa_act[0],
			#extra debug:
			# 'inp_msa_act': inp_msa_act,
			# 'inp_pair_act': inp_pair_act,
			# 'mask_2d': mask_2d,
			# 'extra_msa_act': extra_msa_act,
			# 'single_act': single_act,
			# 'extra_msa_output_msa': extra_act['msa'],
			# 'extra_msa_output_pair': extra_act['pair']
		}
		return output

class AlphaFoldIteration(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L123
	"""
	def __init__(self, config, global_config):
		super().__init__()
		self.config = config
		self.global_config = global_config
	
	def forward(self, ensembled_batch, non_ensembled_batch, 
				is_training:bool=False, compute_loss:bool=False, 
				ensemble_representations:bool=False, return_representations:bool=False):
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