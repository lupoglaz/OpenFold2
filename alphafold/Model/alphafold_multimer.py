
import torch
from torch import nn
import torch.nn.functional as F
from alphafold.Model.alphafold import EvoformerIterationFF
from alphafold.Model.data_transforms import nearest_neighbor_clusters
from embedders import InputEmbeddings, RecycleEmbedding, ExtraMSAEmbedding
from deepspeed import checkpointing as ds_chk
import alphafold.Model.data_transforms_multimer as transf

import functools
from typing import Mapping

class EmbeddingsAndEvoformer(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1681
	"""
	def __init__(self, config, global_config, target_dim:int, msa_dim:int, extra_msa_dim:int,
				clear_cache:bool=False) -> None:
		super(EmbeddingsAndEvoformer, self).__init__()
		self.config = config
		self.global_config = global_config
		self.clear_cache = clear_cache
		
		self.input_emb = InputEmbeddings(config, global_config, msa_dim=msa_dim, target_dim=target_dim)
		self.recycle_emb = RecycleEmbedding(config, global_config)
		self.extra_msa_emb = ExtraMSAEmbedding(config, global_config, msa_dim=extra_msa_dim)
		self.extra_msa_stack = nn.ModuleList()
		for i in range(self.config.extra_msa_stack_num_block):
			self.extra_msa_stack.append(EvoformerIterationFF(	config.evoformer, global_config, 
															msa_dim=config.extra_msa_channel, 
															pair_dim=config.pair_channel, 
															is_extra_msa=True))
		self.evoformer_stack = nn.ModuleList()
		for i in range(self.config.evoformer_num_block):
			self.evoformer_stack.append(EvoformerIterationFF(	config.evoformer, global_config, 
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
		

	def _relative_encoding(self, batch:Mapping[str, torch.Tensor]):
		pass
	
	def forward(self, batch: Mapping[str, torch.Tensor], is_training:bool=False, safe_key=None):
		batch['msa_profile'] = transf.make_msa_profile(batch)
		batch['target_feat'] = F.one_hot(batch['aatype'], 21)
		batch = transf.sample_msa(batch, self.config.num_msa)
		batch = transf.make_masked_msa(batch, self.config.masked_msa)
		batch = transf.nearest_neighbor_clusters(batch)
		batch = transf.create_msa_feat(batch)

		inp_msa_act, inp_pair_act = self.input_emb(batch)
		rec_msa_act, rec_pair_act = self.recycle_emb(batch)
		if not(rec_msa_act is None):
			inp_msa_act[0] += rec_msa_act
		if not(rec_pair_act is None):
			inp_pair_act += rec_pair_act
		if self.config.max_relative_idx:
			inp_pair_act += self._relative_encoding(batch)
		
		if self.config.template.enabled:
			raise Exception(NotImplemented)

		mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

		#EXTRA MSA STACK
		batch = transf.create_extra_msa_feature(batch, self.config.num_extra_msa)
		extra_msa_act = self.extra_msa_emb(batch)
		extra_pair_act = inp_pair_act

		def call_iteration_extra(msa_act, pair_act, msa_mask, pair_mask, index=None):
			return self.extra_msa_stack[index](msa_act, pair_act, msa_mask, pair_mask, is_training=True)

		for i, extra_msa_iteration in enumerate(self.extra_msa_stack):
			if is_training:
				extra_msa_act, extra_pair_act = ds_chk.checkpoint(	functools.partial(call_iteration_extra, index=i), 
																	extra_msa_act, extra_pair_act, batch['extra_msa_mask'], mask_2d)
			else:
				extra_msa_act, extra_pair_act = extra_msa_iteration(extra_msa_act, extra_pair_act, 
																	batch['extra_msa_mask'], mask_2d, is_training=is_training)
			
		#EVOFORMER STACK
		msa_act, pair_act = inp_msa_act, extra_pair_act
		msa_mask, pair_mask = batch['msa_mask'], mask_2d
		
		def call_iteration(msa_act, pair_act, msa_mask, pair_mask, index=None):
			return self.evoformer_stack[index](msa_act, pair_act, msa_mask, pair_mask, is_training=True)

		for i, evoformer_iteration in enumerate(self.evoformer_stack):
			if is_training:
				msa_act, pair_act = ds_chk.checkpoint(functools.partial(call_iteration, index=i), msa_act, pair_act, msa_mask, pair_mask)
			else:
				msa_act, pair_act = evoformer_iteration(msa_act, pair_act, msa_mask, pair_mask, is_training=is_training)
			

		single_act = self.single_activations(msa_act[0])
		output = {
			'single': single_act,
			'pair': pair_act,
			'msa': msa_act[:batch['msa_feat'].size(0), :, :],
			'msa_first_row': msa_act[0]
		}
		return output