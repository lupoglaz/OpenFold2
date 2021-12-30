from typing import Sequence
import torch
import torch.nn as nn
import numpy as np
from typing import Mapping

def params_to_torch(params: Mapping[str, np.ndarray]) -> Mapping[str, Mapping[str, np.ndarray]]:
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/utils.py#L72
	"""
	torch_params = {}
	for path, array in params.items():
		scope, name = path.split('//')
		if scope not in torch_params:
			torch_params[scope] = {}
		torch_params[scope][name] = array
	return torch_params


def load_linear(data, modules, names:Sequence[int]=None, nums:Sequence[int]=None, rel_path: str='predicted_aligned_error_head', ind:int=None):
	if names is None:
		names = [module.name for module in modules]
	else:
		assert len(modules) == len(names)
	if nums is None:
		nums = [1 for name in names]
	else:
		assert len(nums) == len(names)

	for module, name, num in zip(modules, names, nums):
		for i in range(num):
			if i==0:
				add_str = ''
			else:
				add_str = f'_{i}'
			if rel_path is None:
				path = f'{name}{add_str}'
			else:
				path = f'{rel_path}/{name}{add_str}'
			if ind is None:
				print(torch.from_numpy(data[path]['weights']).size())
				w = torch.from_numpy(data[path]['weights']).transpose(-1,-2)
				b = torch.from_numpy(data[path]['bias'])
			else:
				w = torch.from_numpy(data[path]['weights'])[ind,...].transpose(-1,-2)
				b = torch.from_numpy(data[path]['bias'])[ind,...]
			
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

def load_params(data, parameters, names:Sequence[int]=None, rel_path: str='predicted_aligned_error_head', ind:int=None):
	assert len(parameters) == len(names)
	
	for param, name in zip(parameters, names):
		if rel_path is None:
			d = data[f'{name}']
		else:
			if ind is None:
				d = data[f'{rel_path}'][f'{name}']
			else:
				d = data[f'{rel_path}'][f'{name}'][ind,...]
		print(f'Loading {name}: {d.shape} -> {param.size()}')
		param.data.copy_(torch.from_numpy(d))