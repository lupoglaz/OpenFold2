import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Dict

from alphafold.Model.affine import QuatAffine

class PredictedAlignedErrorHead(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1111
	"""
	def __init__(self, config, global_config, num_feat_2d:int) -> None:
		super(PredictedAlignedErrorHead, self).__init__()
		self.config = config
		self.global_config = global_config

		self.logits = nn.Linear(num_feat_2d, config.num_bins)
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def load_weights_from_af2(self, data, rel_path: str='predicted_aligned_error_head', ind:int=None):
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
		act = representations['pair']
		logits = self.logits(act)
		breaks = torch.linspace(start=0, end=self.config.max_error_bin, steps=self.config.num_bins-1)
		return dict(logits=logits, breaks=breaks)

	def loss(self, value:Dict[str,torch.Tensor], batch:Dict[str,torch.Tensor])->Dict[str,torch.Tensor]:
		predicted_affine = QuatAffine.from_tensor(value['structure_module']['final_affines'])
		true_affine = QuatAffine.from_tensor(batch['backbone_affine_tensor'])
		mask = batch['backbone_affine_mask']
		square_mask = mask[:,None] * mask[None,:]
		num_bins = self.config.num_bins
		breaks = value['predicted_aligned_error']['breaks']
		logits = value['predicted_aligned_error']['logits']

		def _local_frame_points(affine):
			points = [x.unsqueeze(dim=-2) for x in affine.translation]
			return affine.invert_point(points, extra_dims=1)
		error_dist2_xyz = [torch.square(x-y) for x,y in zip(_local_frame_points(predicted_affine), _local_frame_points(true_affine))]
		error_dist2 = sum(error_dist2_xyz)
		error_dist2.detach()

		sq_breaks = torch.square(breaks)
		true_bins = torch.sum( (error_dist2[...,None] > sq_breaks).to(dtype=torch.int32), dim=-1)

		errors = self.loss_function(logits, true_bins)
		loss = torch.sum(errors*square_mask, dim=(-2,-1))/(1e-8 + torch.sum(square_mask, dim=(-2,-1)))

		if self.config.filter_by_resolution:
			loss *= ( (batch['resolution']>= self.config.min_resolution) & 
					(batch['resolution']< self.config.max_resolution)).to(dtype=torch.float32)
		return {'loss': loss}

