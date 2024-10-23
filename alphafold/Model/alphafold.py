import torch
from torch import nn

from OmniLayers.Attention.OrigAttention import *
from OmniLayers.Multiplication.OrigMultiplication import *
from OmniLayers.Evoformer.OrigEvoformer import EvoformerIterationOrig
from OmniLayers.Linear.FFLinear import LinearFF as Linear

from alphafold.Model.embedders import *
from alphafold.Model.Heads import *
from typing import Mapping, OrderedDict
import functools
from deepspeed import checkpointing as ds_chk

class EmbeddingsAndEvoformer(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1681
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
			self.extra_msa_stack.append(EvoformerIterationOrig(	config.evoformer, global_config, 
															msa_dim=config.extra_msa_channel, 
															pair_dim=config.pair_channel, 
															is_extra_msa=True))
		self.evoformer_stack = nn.ModuleList()
		for i in range(self.config.evoformer_num_block):
			self.evoformer_stack.append(EvoformerIterationOrig(	config.evoformer, global_config, 
															msa_dim=config.msa_channel, 
															pair_dim=config.pair_channel, 
															is_extra_msa=False))
		self.single_activations = Linear(config.msa_channel, config.seq_channel)
	
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
		# print('Embedding dtype:', self.input_emb.preprocess_1d.weight.dtype)
		# print('Target feat dtype:', batch['target_feat'].dtype)

		inp_msa_act, inp_pair_act = self.input_emb(batch)

		rec_msa_act, rec_pair_act = self.recycle_emb(batch)
		if not(rec_msa_act is None):
			inp_msa_act[0] += rec_msa_act
		if not(rec_pair_act is None):
			inp_pair_act += rec_pair_act

		if self.config.template.enabled:
			raise Exception(NotImplemented)
		
		mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

		extra_msa_act = self.extra_msa_emb(batch)
		extra_pair_act = inp_pair_act

		def call_iteration_ext(msa_act, pair_act, msa_mask, pair_mask, index=None):
			return self.extra_msa_stack[index](msa_act, pair_act, msa_mask, pair_mask, True)

		for i, extra_msa_iteration in enumerate(self.extra_msa_stack):
			if is_training:
				extra_msa_act, extra_pair_act = ds_chk.checkpoint(functools.partial(call_iteration_ext, index=i), 
																	extra_msa_act, extra_pair_act, 
																	batch['extra_msa_mask'], mask_2d)
			else:
				extra_msa_act, extra_pair_act = extra_msa_iteration(extra_msa_act, extra_pair_act, 
																	batch['extra_msa_mask'], mask_2d, 
																	is_training=is_training)
			

		msa_act, pair_act = inp_msa_act, extra_pair_act
		msa_mask, pair_mask = batch['msa_mask'], mask_2d
		
		def call_iteration_evo(msa_act, pair_act, msa_mask, pair_mask, index=None):
			return self.evoformer_stack[index](msa_act, pair_act, msa_mask, pair_mask, True)

		for i, evoformer_iteration in enumerate(self.evoformer_stack):
			if is_training:
				msa_act, pair_act = ds_chk.checkpoint(functools.partial(call_iteration_evo, index=i), msa_act, pair_act, msa_mask, pair_mask)
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

class AlphaFoldIteration(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L123
	"""
	HIGH = 1
	LOW = 2
	def __init__(self, config, global_config, target_dim:int, msa_dim:int, extra_msa_dim:int):
		super().__init__()
		self.config = config
		self.global_config = global_config

		self.evoformer_module = EmbeddingsAndEvoformer(	config.embeddings_and_evoformer, global_config, 
														target_dim=target_dim,
														msa_dim=msa_dim,
														extra_msa_dim=extra_msa_dim)
		evo_conf = config.embeddings_and_evoformer
		head_dict = {
			'masked_msa': (functools.partial(MaskedMSAHead, num_feat_2d=evo_conf.msa_channel), self.LOW),
			'distogram': (functools.partial(DistogramHead, num_feat_2d=evo_conf.pair_channel), self.LOW),
			'structure_module': (functools.partial(StructureModule, num_feat_1d=evo_conf.seq_channel, 
								num_feat_2d=evo_conf.pair_channel), self.HIGH),
			'predicted_lddt': (functools.partial(PredictedLDDTHead, num_feat_1d=evo_conf.seq_channel), self.LOW),
			'predicted_aligned_error': (functools.partial(PredictedAlignedErrorHead, num_feat_2d=evo_conf.pair_channel), self.LOW),
			'experimentally_resolved': (functools.partial(ExperimentallyResolvedHead, num_feat_1d=evo_conf.seq_channel), self.LOW)
		}
		
		self.head_order = [(head_name, head_dict[head_name][1]) for head_name, head_config in self.config.heads.items() if head_config.weight]
		self.head_order.sort(key=lambda x: x[1])
		self.heads = nn.ModuleList()
		for head_name, priority in self.head_order:
			head_config = self.config.heads[head_name]
			if not head_config.weight:
				continue
			head_factory, priority = head_dict[head_name]
			self.heads.append(head_factory(head_config, global_config))
	
	def load_weights_from_af2(self, data, rel_path: str='alphafold_iteration', ind:int=None):
		for (name, priority), module in zip(self.head_order, self.heads):
			if name != 'structure_module':
				load_name = f'{name}_head'
			else:
				load_name = name
			module.load_weights_from_af2(data, rel_path=f'{rel_path}/{load_name}', ind=ind)
		self.evoformer_module.load_weights_from_af2(data, rel_path=f'{rel_path}/evoformer')

	def forward(self, ensembled_batch, non_ensembled_batch, 
				is_training:bool=False, ensemble_representations:bool=False, compute_loss:bool=False):
		
		num_ensemble = torch.Tensor([ensembled_batch['seq_length'].size(0)])
		if not ensemble_representations:
			assert ensembled_batch['seq_length'].size(0) == 1
		
		batch0 = {k:v[0] for k,v in ensembled_batch.items()}
		batch0.update(non_ensembled_batch)
		representations = self.evoformer_module(batch0, is_training)
		
		# msa_representations = representations['msa']
		# del representations['msa']

		if ensemble_representations:
			raise(NotImplementedError())

		batch = batch0

		total_loss = 0.0
		ret = {'representations':representations}
		def loss(module, head_config, ret, name, filter_ret:bool=True):
			if filter_ret:
				value = ret[name]
			else:
				value = ret
			loss_output = module.loss(value, batch)
			ret[name].update(loss_output)
			loss = head_config.weight * ret[name]['loss']
			return loss
		
		for (name, priority), head in zip(self.head_order, self.heads):
			head_config = self.config.heads[name]
			ret[name] = head(representations, batch, is_training)
			if 'representations' in ret[name]:
				representations.update(ret[name].pop('representations'))
			if compute_loss:
				if name in ('predicted_aligned_error', 'predicted_lddt'):
					current_loss = loss(head, head_config, ret, name, filter_ret=False)
				else:
					current_loss = loss(head, head_config, ret, name, filter_ret=True)
				if(torch.isnan(current_loss) or torch.isinf(current_loss)):
					print(f"{name} loss is NaN. Skipping...")
					current_loss = current_loss.new_tensor(0., requires_grad=True)
				total_loss += current_loss
		
		return ret, total_loss
			

class AlphaFold(nn.Module):
	def __init__(self, config, target_dim:int, msa_dim:int, extra_msa_dim:int):
		super().__init__()
		self.config = config
		self.global_config = config.global_config
		self.impl = AlphaFoldIteration(	self.config, self.global_config,
										target_dim=target_dim, 
										msa_dim=msa_dim, 
										extra_msa_dim=extra_msa_dim)
	
	def load_weights_from_af2(self, data, rel_path: str='alphafold', ind:int=None):
		self.impl.load_weights_from_af2(data, rel_path=f'{rel_path}/alphafold_iteration')

	def forward(self, batch, 
				is_training:bool=False, compute_loss:bool=True,
				ensemble_representations:bool=False, 
				iter_num_recycling:torch.Tensor=None):

		batch_size, num_residues = batch['aatype'].shape
		
		def get_prev(ret):
			return {'prev_pos': ret['structure_module']['final_atom_positions'].detach(),
					'prev_msa_first_row': ret['representations']['msa_first_row'].detach(),
					'prev_pair': ret['representations']['pair'].detach()}

		def do_call(prev, recycle_idx, is_training:bool=False, compute_loss:bool=False):
			if self.config.resample_msa_in_recycling:
				num_ensemble = batch_size // (self.config.num_recycle + 1)
				def slice_recycle_idx(x):
					start, end = recycle_idx * num_ensemble, (recycle_idx+1) * num_ensemble
					start_corr = min(start, x.size(0) - num_ensemble)
					end_corr = min(x.size(0), end)
					return x[start_corr:end_corr,...]
				ensembled_batch = {k: slice_recycle_idx(v) for k, v in batch.items()}
			else:
				num_ensemble = batch_size
				ensembled_batch = batch
			
			non_ensembled_batch = prev
			return self.impl(ensembled_batch, non_ensembled_batch, 
							is_training=is_training, 
							ensemble_representations=ensemble_representations,
							compute_loss=compute_loss)

		if self.config.num_recycle:
			emb_config = self.config.embeddings_and_evoformer
			prev = {'prev_pos': batch['target_feat'].new_zeros(num_residues, residue_constants.atom_type_num, 3),
					'prev_msa_first_row':batch['target_feat'].new_zeros(num_residues, emb_config.msa_channel),
					'prev_pair':batch['target_feat'].new_zeros(num_residues, num_residues, emb_config.pair_channel)}
			if not(iter_num_recycling is None):
				num_iter = iter_num_recycling.item()
				num_iter = min(num_iter, self.config.num_recycle)
			else:
				num_iter = self.config.num_recycle

			for i in range(num_iter):
				with torch.no_grad():
					ret, _ = do_call(prev, recycle_idx=i, compute_loss=False, is_training=is_training)
					prev = get_prev(ret)
		else:
			prev = {}
			num_iter = 0
		
		ret, total_loss = do_call(prev=prev, recycle_idx=num_iter, compute_loss=compute_loss, is_training=is_training)
		return ret, total_loss
				

