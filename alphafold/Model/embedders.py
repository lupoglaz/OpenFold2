import torch
from torch import nn
from typing import Tuple

def one_hot(x: torch.Tensor, bins: torch.Tensor):
	#Code taken from https://github.com/aqlaboratory/openfold/blob/03bb003a9d61ed0a0db66bb996f46b1754d7d821/openfold/utils/tensor_utils.py#L60
	#Initial AF2 impl: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1758
	reshaped_bins = bins.view(( (1,) * len(x.shape) ) + (len(bins),))
	diffs = x[..., None] - reshaped_bins
	am = torch.argmin(torch.abs(diffs), dim=-1)
	return nn.functional.one_hot(am, num_classes=len(bins)).to(dtype=x.dtype)

class InputEmbeddings(nn.Module):
	def __init__(self, target_feat_dim: int, msa_feat_dim: int, pair_emb_dim: int, msa_emb_dim: int, relpos_wind: int):
		# Naming after this code:
		# https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1704
		super(InputEmbeddings, self).__init__()
		self.preprocess_1d = nn.Linear(target_feat_dim, msa_emb_dim)
		self.preprocess_msa = nn.Linear(msa_feat_dim, msa_emb_dim)
		self.left_single = nn.Linear(target_feat_dim, pair_emb_dim)
		self.right_single = nn.Linear(target_feat_dim, pair_emb_dim)
		self.pair_activations = nn.Linear(2*relpos_wind + 1, pair_emb_dim)

		self.relpos_wind = relpos_wind
	
	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer'):
		modules=[self.preprocess_1d, self.preprocess_msa, self.left_single, self.right_single, self.pair_activations]
		names=['preprocess_1d', 'preprocess_msa', 'left_single', 'right_single', 'pair_activiations']
		for module, name in zip(modules, names):
			w = data[f'{rel_path}/{name}//weights']
			b = data[f'{rel_path}/{name}//bias']
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

	def forward(self, target_feat: torch.Tensor, residue_index: torch.Tensor, msa_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		preprocess_1d = self.preprocess_1d(target_feat)
		preprocess_msa = self.preprocess_msa(msa_feat)

		#This code is: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1712
		expansion_shape = (-1, )*len(target_feat.shape[:-2]) + (msa_feat.shape[-3], -1, -1)
		preprocess_1d = preprocess_1d.unsqueeze(dim=-3).expand(expansion_shape)
		msa_activations = preprocess_1d + preprocess_msa
		
		left_single = self.left_single(target_feat)
		right_single = self.right_single(target_feat)
		pair_activations = left_single[..., None, :] + right_single[..., None, :, :]
		#This code is: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1764
		pair_activations += self.relpos(residue_index.to(dtype=pair_activations.dtype))

		return msa_activations, pair_activations

class RecycleEmbedding:
	def __init__(self, msa_emb_dim: int, pair_emb_dim: int, min_bin: int, max_bin: int, num_bins: int):
		super(RecycleEmbedding, self).__init__()
		#Naming of the layers are:
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1730
		self.prev_pos_linear = nn.Linear(num_bins, pair_emb_dim)
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1745
		self.prev_pair_norm = nn.LayerNorm(pair_emb_dim)
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1736
		self.prev_msa_first_row_norm = nn.LayerNorm(msa_emb_dim)

		self.bins = torch.linspace(min_bin, max_bin, num_bins)

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer'):
		w = data[f'{rel_path}/prev_pos_linear//weights']
		b = data[f'{rel_path}/prev_pos_linear//bias']
		print(f'Loading prev_pos_linear.weight: {w.shape} -> {self.prev_pos_linear.weight.size()}')
		print(f'Loading prev_pos_linear.bias: {b.shape} -> {self.prev_pos_linear.bias.size()}')
		self.prev_pos_linear.weight.data.copy_(torch.from_numpy(w).transpose(0,1))
		self.prev_pos_linear.bias.data.copy_(torch.from_numpy(b))

		modules=[self.prev_pair_norm, self.prev_msa_first_row_norm]
		names=['prev_pair_norm', 'prev_msa_first_row_norm']
		for module, name in zip(modules, names):
			w = data[f'{rel_path}/{name}//scale']
			b = data[f'{rel_path}/{name}//offset']
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

	def forward(self, prev_pos_linear: torch.Tensor, prev_msa_first_row: torch.Tensor, prev_pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		msa_activations = self.prev_msa_first_row_norm(prev_msa_first_row)
		dgram = self.dgram_from_positions(prev_pos_linear)
		pair_activations = self.prev_pos_linear(dgram) + self.prev_pair_norm(prev_pair)

		return msa_activations, pair_activations

class ExtraMSAEmbedding:
	def __init__(self, msa_dim: int, msa_emb_dim: int):
		self.extra_msa_activations = nn.Linear(msa_dim, msa_emb_dim)

	def load_weights_from_af2(self, data, rel_path: str='alphafold/alphafold_iteration/evoformer'):
		w = data[f'{rel_path}/extra_msa_activations//weights']
		b = data[f'{rel_path}/extra_msa_activations//bias']
		print(f'Loading extra_msa_activations.weight: {w.shape} -> {self.extra_msa_activations.weight.size()}')
		print(f'Loading extra_msa_activations.bias: {b.shape} -> {self.extra_msa_activations.bias.size()}')
		self.extra_msa_activations.weight.data.copy_(torch.from_numpy(w).transpose(0,1))
		self.extra_msa_activations.bias.data.copy_(torch.from_numpy(b))

	def create_extra_msa_features(self, extra_msa: torch.Tensor) -> torch.Tensor:
		pass

	def forward(self, extra_msa: torch.Tensor) -> torch.Tensor:
		return self.extra_msa_activations(extra_msa)
		