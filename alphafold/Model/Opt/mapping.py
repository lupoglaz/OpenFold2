from tracemalloc import start
import torch
from typing import Any, Callable, Optional, Sequence, Union
from alphafold.Model.Utils.tensor_utils import tree_map
from functools import reduce

PYTREE = Any
PYTREE_TORCH_ARRAY = Any

class ShardIterator:
	def __init__(self, batched_args, shard_size:int, dim:int) -> None:
		self.dim = dim
		self.shard_size = shard_size
		self.shapes = [x.shape for x in batched_args]
		self.local_num_chunks = [(x.shape[dim] // shard_size) + int((x.shape[dim] % shard_size) != 0) for x in batched_args]
						
		self.chunked_args = []
		for tensor, n_chunks in zip(batched_args, self.local_num_chunks):
			self.chunked_args.append(list(torch.chunk(tensor, n_chunks, dim=dim)))
		
		self.flat_size = max([arg.size(dim) for arg in batched_args])
		
				
	def __iter__(self):
		self.local_idx = [-1 for _ in self.chunked_args]
		self.flat_slice = None
		return self
	
	def inc_local_idx(self):
		for i in reversed(range(len(self.local_idx))):
			if self.local_idx[i] < (self.local_num_chunks[i]-1):
				self.local_idx[i] += 1
				chunk_size = self.chunked_args[i][self.local_idx[i]].size(self.dim)
			else:
				return None, None
		if self.flat_slice is None:
			self.flat_slice = (0, chunk_size)
		else:
			self.flat_slice = (self.flat_slice[1], self.flat_slice[1] + chunk_size)
		return self.local_idx, self.flat_slice
		

	def __next__(self):
		self.local_idx, self.flat_slice = self.inc_local_idx()
		if not(self.local_idx is None):
			return self.flat_slice, [arg[idx] for arg, idx in zip(self.chunked_args, self.local_idx)]
		else:
			raise StopIteration

def inference_subbatch(	module:Callable[..., PYTREE_TORCH_ARRAY],
						subbatch_size: int,
						batched_args: Sequence[PYTREE_TORCH_ARRAY],
						nonbatched_args: Sequence[PYTREE_TORCH_ARRAY],
						low_memory: bool=True,
						input_subbatch_dim: int=0) -> PYTREE_TORCH_ARRAY:
	""" https://github.com/lupoglaz/alphafold/blob/d7a4bd4c3ab1c403d5e4d849d408782b546402dc/alphafold/model/mapping.py#L193	"""
	assert len(batched_args) > 0
	if not low_memory: 
		args = list(batched_args) + list(nonbatched_args)
		return module(*args)

	def run_module(batched_args):
		args = list(batched_args) + list(nonbatched_args)
		return module(*args)
	"""https://github.com/lupoglaz/alphafold/blob/d7a4bd4c3ab1c403d5e4d849d408782b546402dc/alphafold/model/mapping.py#L82"""
	
	output = None
	shards = ShardIterator(batched_args, subbatch_size, input_subbatch_dim)
	for flat_slice, shard in shards:
		print([sh.size() for sh in shard], flat_slice)
		output_chunk = run_module(shard)
		print(output_chunk.size(), flat_slice)
		if output is None:
			output_flat_shape = (shards.flat_size,) + output_chunk.shape[1:]
			if isinstance(output_chunk, torch.Tensor):
				output = output_chunk.new_zeros(output_flat_shape)
				# print('Output', output.size())
			else:
				output = [t.new_zeros(output_flat_shape) for t in output_chunk]
				# print('Output', [sh.size() for sh in output])
		
		if isinstance(output_chunk, torch.Tensor):
			output[flat_slice[0]:flat_slice[1]] = output_chunk
		else:
			for output_arg, output_chunk_arg in zip(output, output_chunk):
				output_arg[flat_slice[0]:flat_slice[1]] = output_chunk_arg
	
	return output

	
	



