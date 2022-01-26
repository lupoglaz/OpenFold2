import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Dict


class DistogramHead(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1348
	"""
	def __init__(self, config, global_config, num_feat_2d:int) -> None:
		super(DistogramHead, self).__init__()
		self.config = config
		self.global_config = global_config

		self.half_logits = nn.Linear(num_feat_2d, config.num_bins)
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def load_weights_from_af2(self, data, rel_path: str='distogram_head', ind:int=None):
		modules=[self.half_logits]
		names=['half_logits']
		nums=[1]
		for module, name, num in zip(modules, names, nums):
			for i in range(num):
				if i==0:
					add_str = ''
				else:
					add_str = f'_{i}'
				if ind is None:
					w = data[f'{rel_path}/{name}{add_str}']['weights'].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias']
				else:
					w = data[f'{rel_path}/{name}{add_str}']['weights'][ind,...].transpose(-1,-2)
					b = data[f'{rel_path}/{name}{add_str}']['bias'][ind,...]
				if isinstance(module, nn.ModuleList):
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module[i].weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module[i].bias.size()}')
					module[i].weight.data.copy_(torch.from_numpy(w))
					module[i].bias.data.copy_(torch.from_numpy(b))
				else:
					print(f'Loading {name}{add_str}.weight: {w.shape} -> {module.weight.size()}')
					print(f'Loading {name}{add_str}.bias: {b.shape} -> {module.bias.size()}')
					module.weight.data.copy_(torch.from_numpy(w))
					module.bias.data.copy_(torch.from_numpy(b))

	def forward(self, representations:Dict[str,torch.Tensor], batch:Dict[str,torch.Tensor], is_training:bool=False):
		half_logits = self.half_logits(representations['pair'])
		logits = half_logits + half_logits.transpose(-2, -3)
		breaks = torch.linspace(start=self.config.first_break, end=self.config.last_break, steps=self.config.num_bins-1, device=logits.device, dtype=logits.dtype)
		return dict(logits=logits, bin_edges=breaks)

	def loss(self, value:Dict[str,torch.Tensor], batch:Dict[str,torch.Tensor])->Dict[str,torch.Tensor]:
		logits = value['logits']
		bin_edges = value['bin_edges']
		positions = batch['pseudo_beta']
		mask = batch['pseudo_beta_mask']
		

		assert logits.ndimension() == 3
		assert positions.size(-1) == 3

		sq_breaks = torch.square(bin_edges)
		dist2 = torch.sum(torch.square(positions.unsqueeze(dim=-2) - positions.unsqueeze(dim=-3)), dim=-1, keepdim=True)
		
		true_bins = torch.sum(dist2>sq_breaks, dim=-1)
		errors = self.loss_function(logits.view(-1, self.config.num_bins), true_bins.view(-1))
		square_mask = mask.unsqueeze(dim=-2) * mask.unsqueeze(dim=-1)
		errors = errors.view(positions.size(-2), positions.size(-2))
		
		avg_error = torch.sum(errors*square_mask, dim=(-2,-1))/(1e-8 + torch.sum(square_mask, dim=(-2,-1)))
		dist2 = dist2[...,0]
		return dict(loss=avg_error, true_dist=torch.sqrt(1e-6 + dist2))
