import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Mapping
from alphafold.Common import residue_constants

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

	def _relative_encoding(self, residue_index: torch.Tensor):
		# Algorithm 4
		# Here in the AF2 code: https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1758
		#Different in AF21: https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/model/modules_multimer.py#L506
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