from tracemalloc import start
import torch
from typing import Any, Callable, Optional, Sequence, Union
from alphafold.Model.Utils.tensor_utils import tree_map
from functools import reduce
import torch.nn.functional as F

PYTREE = Any
PYTREE_TORCH_ARRAY = Any

class ShardIterator:
	def __init__(self, batched_args, shard_size:int, dim:int) -> None:
		self.dim = dim
		self.shard_size = shard_size
		self.shapes = [x.shape[dim] for x in batched_args]
		self.flat_size = reduce(lambda x,y: x*y, [sz for sz in self.shapes])
		self.num_chunks = [(x.shape[dim] // shard_size) + int((x.shape[dim] % shard_size) != 0) for x in batched_args]
		self.flat_num_chunks = reduce(lambda x,y: x*y, [sz for sz in self.num_chunks])
		
		pad = [x.shape[dim] % shard_size != 0 for x in batched_args]
		# pad_sizes = [shard_size - (x.shape[dim] % shard_size) for x in batched_args]
		padded_args = []
		for arg, pad in zip(batched_args, pad):
			if pad:
				dim_indexes = list(range(len(arg.shape)))
				dim_indexes.reverse()
				pad_size = [(0, shard_size - (arg.shape[dim] % shard_size)) if i==0 else (0, 0) for i in dim_indexes]
				pad_size = reduce(lambda x,y: x+y, pad_size)
				padded_arg = F.pad(arg, pad_size, mode='constant', value=0.0)
				padded_args.append(padded_arg)
			else:
				padded_args.append(arg)

		self.padded_shapes = [x.shape[dim] for x in padded_args]
		self.padded_flat_size = reduce(lambda x,y: x*y, [sz for sz in self.padded_shapes])

		self.chunked_args = []
		for tensor, n_chunks in zip(padded_args, self.num_chunks):
			self.chunked_args.append(list(torch.chunk(tensor, n_chunks, dim=dim)))
		
		
		self.remainder_num_chunks = []
		self.remainder_size = []
		for i in range(len(self.num_chunks)-1):
			self.remainder_num_chunks.append(reduce(lambda x,y: x*y, [sz for sz in self.num_chunks[i+1:]]))
			self.remainder_size.append(reduce(lambda x,y: x*y, [arg.shape[dim] for arg in batched_args[i+1:]]))
		self.remainder_num_chunks.append(1)
		self.remainder_size.append(1)

	def __iter__(self):
		self.index = -1
		self.flat_slice = None
		return self
	
	def inc_idx(self):
		self.index += 1
		if self.index < self.flat_num_chunks:
			idx = self.index
			local_index = []
			for remainder in self.remainder_num_chunks:
				local_index.append(idx // remainder)
				idx = idx % remainder

			chunk_size = reduce(lambda x,y: x*y, [arg[idx].size(self.dim) for arg, idx in zip(self.chunked_args, local_index)])
			if self.flat_slice is None:
				self.flat_slice = (0, chunk_size)
			else:
				self.flat_slice = (self.flat_slice[1], self.flat_slice[1] + chunk_size)

			return local_index, self.flat_slice
		else:
			return None, None
		

	def __next__(self):
		local_idx, flat_slice = self.inc_idx()
		if not(local_idx is None):
			return flat_slice, [arg[idx] for arg, idx in zip(self.chunked_args, local_idx)]
		else:
			raise StopIteration

class SimpleShardIterator:
	def __init__(self, batched_args, shard_size:int, dim:int) -> None:
		self.dim = dim
		self.shard_size = shard_size
		self.shapes = [x.shape for x in batched_args]
		self.num_chunks = batched_args[0].shape[dim] // shard_size + int((batched_args[0].shape[dim] % shard_size) != 0)
		self.flat_size = batched_args[0].shape[dim]
		
		pad = batched_args[0].shape[dim] % shard_size != 0
		padded_args = []
		for arg in batched_args:
			if pad:
				dim_indexes = list(range(len(arg.shape)))
				dim_indexes.reverse()
				pad_size = [(0, shard_size - (arg.shape[dim] % shard_size)) if i==0 else (0, 0) for i in dim_indexes]
				pad_size = reduce(lambda x,y: x+y, pad_size)
				padded_arg = F.pad(arg, pad_size, mode='constant', value=0.0)
				padded_args.append(padded_arg)
			else:
				padded_args.append(arg)

		self.padded_shapes = [x.shape for x in padded_args]
		self.padded_flat_size = padded_args[0].shape[dim]
		
		self.chunked_args = []
		for tensor in padded_args:
			self.chunked_args.append(list(torch.chunk(tensor, self.num_chunks, dim=dim)))
					
		#All chunks are the same
		assert all([x.shape[dim]==batched_args[0].shape[dim] for x in batched_args])
		assert all([len(arg)==self.num_chunks for i, arg in enumerate(self.chunked_args)])
				
	def __iter__(self):
		self.index = -1
		self.flat_slice = None
		return self
	
	def inc_idx(self):
		if self.index < (self.num_chunks-1):
			self.index += 1
		else:
			return None, None
		
		chunk_size = self.chunked_args[0][self.index].size(self.dim)
		if self.flat_slice is None:
			self.flat_slice = (0, chunk_size)
		else:
			self.flat_slice = (self.flat_slice[1], self.flat_slice[1] + chunk_size)
		return self.index, self.flat_slice
		

	def __next__(self):
		self.index, self.flat_slice = self.inc_idx()
		if not(self.index is None):
			return self.flat_slice, [arg[self.index] for arg in self.chunked_args]
		else:
			raise StopIteration

def inference_subbatch(	module:Callable[..., PYTREE_TORCH_ARRAY],
						subbatch_size: int,
						batched_args: Sequence[PYTREE_TORCH_ARRAY],
						nonbatched_args: Sequence[PYTREE_TORCH_ARRAY],
						low_memory: bool=True,
						input_subbatch_dim: int=0,
						output_subbatch_dims: int=None) -> PYTREE_TORCH_ARRAY:
	""" https://github.com/lupoglaz/alphafold/blob/d7a4bd4c3ab1c403d5e4d849d408782b546402dc/alphafold/model/mapping.py#L193	
	This implementation is brittle when output_subbatch_dims != [0,1] or int. Won't work with other pairs of dims or triple dims, or even other dims indexes.
	But at least it work for OuterProductMean.
	"""
	assert len(batched_args) > 0
	if not low_memory: 
		args = list(batched_args) + list(nonbatched_args)
		return module(*args)
	
	def run_module(batched_args):
		args = list(batched_args) + list(nonbatched_args)
		return module(*args)

	"""https://github.com/lupoglaz/alphafold/blob/d7a4bd4c3ab1c403d5e4d849d408782b546402dc/alphafold/model/mapping.py#L82"""
	
	output = None
	
	if not(output_subbatch_dims is None):
		assert output_subbatch_dims[0] == 0
		assert output_subbatch_dims[1] == 1
		assert input_subbatch_dim == 0
		shards = ShardIterator(batched_args, subbatch_size, input_subbatch_dim)
	else:
		shards = SimpleShardIterator(batched_args, subbatch_size, input_subbatch_dim)

	for flat_slice, shard in shards:
		# print('shard', [arg.size() for arg in shard])
		output_chunk = run_module(shard)
		# print('output_chunk', output_chunk.size(), flat_slice)
		if output is None:
			if output_subbatch_dims is None:
				output_flat_padded_shape = (shards.padded_flat_size,) + output_chunk.shape[1:]
				output_flat_shape = (shards.flat_size,) + output_chunk.shape[1:]
				output_shape = (shards.flat_size,) + output_chunk.shape[1:]
			else:
				output_flat_padded_shape = (shards.padded_flat_size,) + output_chunk.shape[len(output_subbatch_dims):]
				output_flat_shape = (shards.flat_size,) + output_chunk.shape[len(output_subbatch_dims):]
				output_padded_shape = tuple(shards.padded_shapes) + output_chunk.shape[len(output_subbatch_dims):]
				output_shape = tuple(shards.shapes) + output_chunk.shape[len(output_subbatch_dims):]

			if isinstance(output_chunk, torch.Tensor):
				output = output_chunk.new_zeros(output_flat_padded_shape)
				# print('Output', output.size())
			else:
				raise(NotImplementedError())
				# output = [t.new_zeros(output_flat_shape) for t in output_chunk]
		
		if isinstance(output_chunk, torch.Tensor):
			output[flat_slice[0]:flat_slice[1]] = output_chunk.view(output[flat_slice[0]:flat_slice[1]].size())
		else:
			raise(NotImplementedError())
			# for output_arg, output_chunk_arg in zip(output, output_chunk):
			# 	output_arg[flat_slice[0]:flat_slice[1]] = output_chunk_arg
		# print(torch.cuda.memory_summary(device=None, abbreviated=False))

	if not(output_subbatch_dims is None):
		current_output_shape = tuple(shards.num_chunks) + (subbatch_size, subbatch_size) + output_chunk.shape[len(output_subbatch_dims):]
		output = output.view(current_output_shape)
		output = output.transpose(1,2).contiguous()
		output = output.view(shards.padded_shapes[0], shards.padded_shapes[0], -1)
		output = output[:output_shape[0], :output_shape[1], ...]
	else:
		output = output[:shards.flat_size, ...]

	return output.view(output_shape)

	
	



