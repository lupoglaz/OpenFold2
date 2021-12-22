import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Dict

class MaskedMSAHead(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L962
	"""
	def __init__(self, config, global_config, num_feat_2d:int) -> None:
		super(MaskedMSAHead, self).__init__()
		self.config = config
		self.global_config = global_config

		self.logits = nn.Linear(num_feat_2d, config.num_output)
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def load_weights_from_af2(self, data, rel_path: str='masked_msa_head', ind:int=None):
		modules=[self.logits]
		names=['logits']
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
		del batch
		logits = self.logits(representations['msa'])
		return dict(logits=logits)

	def loss(self, value:Dict[str,torch.Tensor], batch:Dict[str,torch.Tensor])->Dict[str,torch.Tensor]:
		errors = self.loss_function(value['logits'], batch['true_msa'])
		loss = torch.sum(errors * batch['bert_mask'], dims=(-2,-1))/(1e-8 + torch.sum(batch['bert_mask'], dim=(-2,-1)))
		return {'loss':loss}