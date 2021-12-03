import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Mapping
from alphafold.Common import residues_constants

def one_hot(x: torch.Tensor, bins: torch.Tensor):
	#Code taken from https://github.com/aqlaboratory/openfold/blob/03bb003a9d61ed0a0db66bb996f46b1754d7d821/openfold/utils/tensor_utils.py#L60
	#Initial AF2 impl: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1758
	reshaped_bins = bins.view(( (1,) * len(x.shape) ) + (len(bins),))
	diffs = x[..., None] - reshaped_bins
	am = torch.argmin(torch.abs(diffs), dim=-1)
	return nn.functional.one_hot(am, num_classes=len(bins)).to(dtype=x.dtype)

class InputEmbeddings(nn.Module):
	def __init__(self, config, global_config, target_dim: int, msa_dim: int):
		# Naming after this code:
		# https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1704
		super(InputEmbeddings, self).__init__()
		self.config = config
		self.global_config = global_config

		self.relpos_wind = config.max_relative_feature
		self.preprocess_1d = nn.Linear(target_dim, config.msa_channel)
		self.preprocess_msa = nn.Linear(msa_dim, config.msa_channel)
		self.left_single = nn.Linear(target_dim, config.pair_channel)
		self.right_single = nn.Linear(target_dim, config.pair_channel)
		self.pair_activations = nn.Linear(2*config.max_relative_feature + 1, config.pair_channel)
	
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer'):
		modules=[self.preprocess_1d, self.preprocess_msa, self.left_single, self.right_single, self.pair_activations]
		names=['preprocess_1d', 'preprocess_msa', 'left_single', 'right_single', 'pair_activiations']
		for module, name in zip(modules, names):
			w = data[f'{rel_path}/{name}']['weights']
			b = data[f'{rel_path}/{name}']['bias']
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w).transpose(0,1))
			module.bias.data.copy_(torch.from_numpy(b))

	def relpos(self, residue_index: torch.Tensor):
		# Algorithm 4
		# Here in the AF2 code: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1758
		d = residue_index[..., None] - residue_index[..., None, :]
		boundaries = torch.arange(start=-self.relpos_wind, end=self.relpos_wind+1, device=d.device)
		return self.pair_activations(one_hot(d, boundaries))

	def forward(self, batch:Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
		preprocess_1d = self.preprocess_1d(batch['target_feat'])
		preprocess_msa = self.preprocess_msa(batch['msa_feat'])

		#This code is: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1712
		expansion_shape = (-1, )*len(batch['target_feat'].shape[:-2]) + (batch['msa_feat'].shape[-3], -1, -1)
		preprocess_1d = preprocess_1d.unsqueeze(dim=-3).expand(expansion_shape)
		msa_act = preprocess_1d + preprocess_msa
		
		left_single = self.left_single(batch['target_feat'])
		right_single = self.right_single(batch['target_feat'])
		pair_act = left_single[..., None, :] + right_single[..., None, :, :]
		#This code is: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1764
		pair_act += self.relpos(batch['residue_index'].to(dtype=pair_act.dtype))

		return msa_act, pair_act

class RecycleEmbedding(nn.Module):
	def __init__(self, config, global_config):
		super(RecycleEmbedding, self).__init__()
		self.config = config
		self.global_config = global_config

		#Naming of the layers are:
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1730
		self.prev_pos_linear = nn.Linear(config.prev_pos.num_bins, config.pair_channel)
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1745
		self.prev_pair_norm = nn.LayerNorm(config.pair_channel)
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1736
		self.prev_msa_first_row_norm = nn.LayerNorm(config.msa_channel)

		self.bins = torch.linspace(config.prev_pos.min_bin, config.prev_pos.max_bin, config.prev_pos.num_bins)

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer'):
		w = data[f'{rel_path}/prev_pos_linear']['weights']
		b = data[f'{rel_path}/prev_pos_linear']['bias']
		print(f'Loading prev_pos_linear.weight: {w.shape} -> {self.prev_pos_linear.weight.size()}')
		print(f'Loading prev_pos_linear.bias: {b.shape} -> {self.prev_pos_linear.bias.size()}')
		self.prev_pos_linear.weight.data.copy_(torch.from_numpy(w).transpose(0,1))
		self.prev_pos_linear.bias.data.copy_(torch.from_numpy(b))

		modules=[self.prev_pair_norm, self.prev_msa_first_row_norm]
		names=['prev_pair_norm', 'prev_msa_first_row_norm']
		for module, name in zip(modules, names):
			w = data[f'{rel_path}/{name}']['scale']
			b = data[f'{rel_path}/{name}']['offset']
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w))
			module.bias.data.copy_(torch.from_numpy(b))
	
	def dgram_from_positions(self, x):
		#Code from here:
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1509
		if (self.bins.dtype != x.dtype) or (self.bins.device != x.device):
			self.bins = self.bins.to(dtype=x.dtype, device=x.device)

		lower_bins2 = self.bins*self.bins
		upper_bins2 = torch.cat([lower_bins2[1:], torch.tensor([1e8], dtype=x.dtype, device=x.device)], dim=-1)
		d = x.unsqueeze(dim=-2) - x.unsqueeze(dim=-3)
		d2 = d*d
		dist2 = torch.sum(d2, dim=-1, keepdim=True)
		dgram = (dist2>lower_bins2).to(dtype=x.dtype) * (dist2<upper_bins2).to(dtype=x.dtype)
		return dgram
	
	def pseudo_beta_fn(self, aatype, all_atom_positions, all_atom_masks):
		"""
		https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1541
		"""

		is_gly = torch.eq(aatype, residues_constants.restype_order['G'])
		ca_idx = residues_constants.atom_order['CA']
		cb_idx = residues_constants.atom_order['CB']
		pseudo_beta = torch.where(
			torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
			all_atom_positions[..., ca_idx, :],
			all_atom_positions[..., cb_idx, :])

		if all_atom_masks is not None:
			pseudo_beta_mask = torch.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
			pseudo_beta_mask = pseudo_beta_mask.to(dtype = torch.float32)
			return pseudo_beta, pseudo_beta_mask
		else:
			return pseudo_beta

	def forward(self, batch:Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
		msa_act = None
		pair_act = None
		if self.config.recycle_pos and 'prev_pos' in batch:
			prev_pseudo_beta = self.pseudo_beta_fn(batch['aatype'], batch['prev_pos'], None)
			dgram = self.dgram_from_positions(prev_pseudo_beta)
			pair_act = self.prev_pos_linear(dgram)
		
		if self.config.recycle_features:
			if 'prev_pair' in batch:
				pair_act += self.prev_pair_norm(batch['prev_pair'])
		
			if 'prev_msa_first_row' in batch:
				msa_act = self.prev_msa_first_row_norm(batch['prev_msa_first_row'])

		return msa_act, pair_act

class ExtraMSAEmbedding(nn.Module):
	def __init__(self, config, global_config, msa_dim: int):
		super(ExtraMSAEmbedding, self).__init__()
		self.config = config
		self.global_config = global_config
		self.extra_msa_activations = nn.Linear(msa_dim, config.extra_msa_channel)

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer'):
		w = data[f'{rel_path}/extra_msa_activations']['weights']
		b = data[f'{rel_path}/extra_msa_activations']['bias']
		print(f'Loading extra_msa_activations.weight: {w.shape} -> {self.extra_msa_activations.weight.size()}')
		print(f'Loading extra_msa_activations.bias: {b.shape} -> {self.extra_msa_activations.bias.size()}')
		self.extra_msa_activations.weight.data.copy_(torch.from_numpy(w).transpose(0,1))
		self.extra_msa_activations.bias.data.copy_(torch.from_numpy(b))

	def create_extra_msa_features(self, batch:Mapping[str, torch.Tensor]) -> torch.Tensor:
		"""
		https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L98
		"""
		msa_1hot = F.one_hot(batch['extra_msa'].to(dtype=torch.long), num_classes=23)
		msa_feat = [msa_1hot, batch['extra_has_deletion'].unsqueeze(dim=-1), batch['extra_deletion_value'].unsqueeze(dim=-1)]
		return torch.cat(msa_feat, dim=-1)

	def forward(self, batch:Mapping[str, torch.Tensor]) -> torch.Tensor:
		extra_msa_feat = self.create_extra_msa_features(batch)
		return self.extra_msa_activations(extra_msa_feat)
		