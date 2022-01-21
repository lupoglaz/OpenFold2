import torch
from typing import List

def permute_final_dims(tensor:torch.Tensor, idx:List[int]):
	"""
	https://github.com/aqlaboratory/openfold/blob/e1142cf3d2e47ae9d90e401b2bd10b6944a67bd3/openfold/utils/tensor_utils.py#L22
	"""
	zero_index = -1 * len(idx)
	first_idx = list(range(len(tensor.shape[:zero_index])))
	return tensor.permute(first_idx + [zero_index + i for i in idx])

def flatten_final_dims(tensor:torch.Tensor, num_dims:int):
	return tensor.reshape(tensor.shape[:-num_dims] + (-1,))

def tree_map(fn, tree):
	if isinstance(tree, dict):
		return {k: tree_map(fn, v) for k,v in tree.items()}
	if isinstance(tree, list):
		return [tree_map(fn, v) for v in tree]
	if isinstance(tree, tuple):
		return tuple([tree_map(fn, v) for v in tree])
	if isinstance(tree, torch.Tensor):
		return fn(tree)
	raise(ValueError(f'Not supported {type(tree)}'))

def batched_gather(data, inds, dim=0, no_batch_dims=0):
	"""https://github.com/aqlaboratory/openfold/blob/5037b3d029efcc9eeb3f5f33eedd4ecdd27ca962/openfold/utils/tensor_utils.py#L67"""
	ranges = []
	for i, s in enumerate(data.shape[:no_batch_dims]):
		r = torch.arange(s)
		r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
		ranges.append(r)

	remaining_dims = [
		slice(None) for _ in range(len(data.shape) - no_batch_dims)
	]
	remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
	ranges.extend(remaining_dims)
	return data[ranges]