import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Dict

from alphafold.Common import residue_constants

def lddt(predicted_points:torch.Tensor,
		true_points:torch.Tensor,
		true_points_mask:bool,
		cutoff:float=15.0,
		per_residue:bool=False) -> torch.Tensor:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/lddt.py#L19
	"""
	assert predicted_points.ndimension() == 3
	assert predicted_points.size(-1) == 3
	assert true_points_mask.ndimension() == 1
	assert predicted_points.size(-1) == 1

	dmat_true = torch.sqrt(1e-10 + torch.sum(
		torch.square(true_points[:, :, None] - true_points[:, None, :]), dim=-1
	))
	dmat_predicted = torch.sqrt(1e-10 + torch.sum(
		torch.square(predicted_points[:, :, None] - predicted_points[:, None, :]), dim=-1
	))
	dists_to_score = ( (dmat_true<cutoff).to(dtype=torch.float32) * 
		true_points_mask * (true_points_mask.transpose(-1, -2)) *
		(1.0 - torch.eye(dmat_true.size(1)))
		)
	dist_l1 = torch.abs(dmat_true - dmat_predicted)
	score = 0.25*( 	(dist_l1 < 0.5).to(dtype=torch.float32) + 
					(dist_l1 < 1.0).to(dtype=torch.float32) +
					(dist_l1 < 2.0).to(dtype=torch.float32) +
					(dist_l1 < 4.0).to(dtype=torch.float32))
	reduce_dims = (-1,) if per_residue else (-1, -2)
	norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=reduce_dims))
	score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=reduce_dims))
	return score


class PredictedLDDTHead(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/modules.py#L1007
	"""
	def __init__(self, config, global_config, num_feat_1d:int) -> None:
		super(PredictedLDDTHead, self).__init__()
		self.config = config
		self.global_config = global_config

		self.input_layer_norm = nn.LayerNorm(num_feat_1d)
		self.act_0 = nn.Linear(num_feat_1d, config.num_channels)
		self.act_1 = nn.Linear(config.num_channels, config.num_channels)
		self.logits = nn.Linear(config.num_channels, config.num_bins)
		self.relu = nn.ReLU()

		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def load_weights_from_af2(self, data, rel_path: str='predicted_lddt_head', ind:int=None):
		modules=[self.act_0, self.act_1, self.logits]
		names=['act_0', 'act_1', 'logits']
		nums=[1, 1, 1]
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

		modules=[self.input_layer_norm]
		names=['input_layer_norm']
		for module, name in zip(modules, names):
			if ind is None:
				w = data[f'{rel_path}/{name}']['scale']
				b = data[f'{rel_path}/{name}']['offset']
			else:
				w = data[f'{rel_path}/{name}']['scale'][ind,...]
				b = data[f'{rel_path}/{name}']['offset'][ind,...]
			print(f'Loading {name}.weight: {w.shape} -> {module.weight.size()}')
			print(f'Loading {name}.bias: {b.shape} -> {module.bias.size()}')
			module.weight.data.copy_(torch.from_numpy(w))
			module.bias.data.copy_(torch.from_numpy(b))

	def forward(self, representations:Dict[str,torch.Tensor], batch:Dict[str,torch.Tensor], is_training:bool=False):
		act = representations['structure_module']
		act = self.input_layer_norm(act)
		act = self.relu(self.act_0(act))
		act = self.relu(self.act_1(act))
		logits = self.logits(act)
		return dict(logits=logits)

	def loss(self, value:Dict[str,torch.Tensor], batch:Dict[str,torch.Tensor])->Dict[str,torch.Tensor]:
		pred_all_atom_pos = value['structure_module']['final_atom_positions']
		true_all_atom_pos = batch['all_atom_positions']
		all_atom_mask = batch['all_atom_mask']
		num_bins = self.config.num_bins

		lddt_ca = lddt(
			predicted_points=pred_all_atom_pos[None,:,1,:],
			true_points=true_all_atom_pos[None,:,1,:],
			true_points_mask=all_atom_mask[None, :, 1:2].to(dtype=torch.float32),
			cutoff=15.0, per_residue=True)[0]
		lddt_ca = lddt_ca.detach()

		bin_index = torch.floor(lddt_ca*num_bins).to(dtype=torch.int32)
		bin_index = torch.minimum(bin_index, num_bins - 1)
		# lddt_ca_one_hot = F.one_hot(bin_index, num_classes=num_bins)

		logits = value['predicted_lddt']['logits']
		errors = self.loss_function(logits, bin_index)

		mask_ca = all_atom_mask[:, residue_constants.atom_order['CA']]
		mask_ca = mask_ca.to(dtype=torch.float32)
		loss = torch.sum(errors*mask_ca)/(torch.sum(mask_ca) + 1e-8)
		
		if self.config.filter_by_resolution:
			loss *= ( (batch['resolution']>= self.config.min_resolution) & 
					(batch['resolution']< self.config.max_resolution)).to(dtype=torch.float32)
		
		return {'loss':loss}