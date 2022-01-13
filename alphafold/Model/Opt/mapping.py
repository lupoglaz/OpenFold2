import torch
from typing import Any, Callable, Optional, Sequence, Union

PYTREE = Any
PYTREE_TORCH_ARRAY = Any
def inference_subbatch(module:Callable[..., PYTREE_TORCH_ARRAY],
						subbatch_size: int,
						batched_args: Sequence[PYTREE_TORCH_ARRAY],
						nonbatched_args: Sequence[PYTREE_TORCH_ARRAY],
						low_memory: bool=True,
						input_subbatch_dim: int=0,
						output_subbatch_dim: Optional[int]=None) -> PYTREE_TORCH_ARRAY:
	""" https://github.com/lupoglaz/alphafold/blob/d7a4bd4c3ab1c403d5e4d849d408782b546402dc/alphafold/model/mapping.py#L193	"""
	assert len(batched_args) > 0
	if not low_memory: 
		args = list(batched_args) + list(nonbatched_args)
		return module(*args)

	if output_subbatch_dim is None:
		output_subbatch_dim = input_subbatch_dim
	def run_module(batched_args):
		args = list(batched_args) + list(nonbatched_args)
		return module(*args)
	sharded_module = sharded_apply(run_module, shard_size=subbatch_size, in_axes=input_subbatch_dim, out_axes=output_subbatch_dim)
	return sharded_module(*batched_args)

def sharded_apply(fun:Callable[..., PYTREE_TORCH_ARRAY],
					shard_size:Union[int, None]=1, 
					in_axes:Union[int, PYTREE]=0,
					out_axes:Union[int, PYTREE]=0) -> Callable[..., PYTREE_TORCH_ARRAY]:
	"""https://github.com/lupoglaz/alphafold/blob/d7a4bd4c3ab1c403d5e4d849d408782b546402dc/alphafold/model/mapping.py#L82"""
	pass



