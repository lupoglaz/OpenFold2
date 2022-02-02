# from torch.utils.checkpoint import CheckpointFunction as StdCheckpointFunction
from turtle import forward
import torch
from typing import Any, Iterable, List, Dict, Tuple, Callable
from collections.abc import Iterable
import torch.utils.checkpoint as chkpt

def recursive_walk(inputs: Iterable):
	if isinstance(inputs, tuple) or isinstance(inputs, list):
		for inp in inputs:
			yield from recursive_walk(inp)
	elif isinstance(inputs, dict):
		for key, inp in inputs.items():
			yield from recursive_walk(inp)
	else:
		yield inputs

def recursive_apply(fn: Callable, inputs: Iterable):
	if isinstance(inputs, tuple):
		return tuple([recursive_apply(fn, inp) for inp in inputs])
	elif isinstance(inputs, list):
		return [recursive_apply(fn, inp) for inp in inputs]
	elif isinstance(inputs, dict):
		return {key:recursive_apply(fn, inp) for key, inp in inputs.items()}
	else:
		return fn(inputs)

def recursive_flatten(inputs: Iterable, condition: Callable):
	output = []
	if isinstance(inputs, tuple):
		for inp in inputs:
			flat_inp = recursive_flatten(inp, condition)
			if not flat_inp is None:
				output += flat_inp
	elif isinstance(inputs, list):
		for inp in inputs:
			flat_inp = recursive_flatten(inp, condition)
			if not flat_inp is None:
				output += flat_inp
	elif isinstance(inputs, dict):
		for key, inp in inputs.items():
			flat_inp = recursive_flatten(inp, condition)
			if not flat_inp is None:
				output += flat_inp
	elif condition(inputs):
		output.append(inputs)
		return [inputs]
	return output

def detach_variable(inputs: Iterable):
	def detach(inp: torch.Tensor):
		x = inp.detach()
		x.requires_grad = inp.requires_grad
		return x
	return recursive_apply(detach, inputs)

def check_backward_validity(inputs: Iterable):
	flat_tensors = recursive_flatten(inputs, lambda x: isinstance(x, torch.Tensor))
	return any(map(lambda x: x.requires_grad, flat_tensors))

def get_device_states(*args):
	flat_tensors = recursive_flatten(args, lambda x: isinstance(x, torch.Tensor))
	gpu_tensors = filter(lambda x: x.is_cuda, flat_tensors)
	fwd_gpu_devices = map(lambda x: x.get_device(), gpu_tensors)
	def get_device_state(device):
		with torch.cuda.device(device):
			return torch.cuda.get_rng_state()
	fwd_gpu_states = {device:get_device_state(device) for device in fwd_gpu_devices}
	return fwd_gpu_states

def set_device_states(device_states):
	for device, state in device_states.items():
		with torch.cuda.device(device):
			torch.cuda.set_rng_state(state)

class TensorPlaceholder:
	def __init__(self, tensor_index) -> None:
		self.tensor_index = tensor_index
	def __repr__(self) -> str:
		return f"TensorPlaceholder({self.tensor_index})"

class CheckpointFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, run_function, preserve_rng_state, *args):
		check_backward_validity(args)
		ctx.run_function = run_function
		ctx.preserve_rng_state = preserve_rng_state
		ctx.has_autocast_in_fwd = torch.is_autocast_enabled()
		if preserve_rng_state:
			ctx.fwd_cpu_state = torch.get_rng_state()
			ctx.had_cuda_in_fwd = False
			if torch.cuda._initialized:
				ctx.had_cuda_in_fwd = True
				ctx.device_states = get_device_states(*args)
		
		def replace_tensors(arg):
			if torch.is_tensor(arg):
				return TensorPlaceholder(None)
			return arg

		tensor_inputs = []
		ctx.inputs = recursive_apply(replace_tensors, args)
		for input_arg, all_arg in zip(recursive_walk(ctx.inputs), recursive_walk(args)):
			if isinstance(input_arg, TensorPlaceholder):
				assert torch.is_tensor(all_arg)
				assert input_arg.tensor_index is None
				tensor_inputs.append(all_arg)
				input_arg.tensor_index = len(tensor_inputs) - 1
		ctx.save_for_backward(*tensor_inputs)
		with torch.no_grad():
			outputs = run_function(*args)
		for output in recursive_walk(outputs):
			if torch.is_tensor(output):
				output.requires_grad_(True)
		return outputs
	
	@staticmethod
	def backward(ctx, *args):
		print('bwd args', args)
		if not torch.autograd._is_checkpoint_valid():
			raise RuntimeError("Checkpointing is not compatible with .grad() or when an `inputs` parameter"
				" is passed to .backward(). Please use .backward() and do not pass its `inputs`"
				" argument.")
		def replace_placeholder(arg):
			if isinstance(arg, TensorPlaceholder):
				assert not(arg.tensor_index is None)
				return ctx.saved_tensors[arg.tensor_index]
			return arg
		inputs = recursive_apply(replace_placeholder, ctx.inputs)
		
		rng_devices = []
		if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
			rng_devices = ctx.device_states.keys()
		with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
			if ctx.preserve_rng_state:
				torch.set_rng_state(ctx.fwd_cpu_state)
				if ctx.had_cuda_in_fwd:
					set_device_states(ctx.device_states)
				detached_inputs = detach_variable(inputs)
				with torch.enable_grad(), torch.cuda.amp.autocast(ctx.has_autocast_in_fwd):
					outputs = ctx.run_function(*detached_inputs)
		if isinstance(outputs, torch.Tensor):
			outputs = (outputs,)
		
		outputs_with_grad = []
		args_with_grad = []
		for arg, output in zip(recursive_walk(args), recursive_walk(outputs)):
			if torch.is_tensor(output) and output.requires_grad:
				outputs_with_grad.append(output)
				args_with_grad.append(arg)
		
		if len(outputs_with_grad) == 0:
			raise RuntimeError("none of output has requires_grad=True,"
				" this checkpoint() is not necessary")
		torch.autograd.backward(outputs_with_grad, args_with_grad)
		grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)
		print(args)
		print(grads)
		return (None, None) + grads

def checkpoint(function, *args, **kwargs):
	preserve = kwargs.pop('preserve_rng_state', True)
	if kwargs:
		raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
	
	def replace_tensors(arg):
		if torch.is_tensor(arg):
			return TensorPlaceholder(None)
		return arg

	tensor_inputs = []
	non_tensor_inputs = recursive_apply(replace_tensors, args)
	for input_arg, all_arg in zip(recursive_walk(non_tensor_inputs), recursive_walk(args)):
		if isinstance(input_arg, TensorPlaceholder):
			assert torch.is_tensor(all_arg)
			assert input_arg.tensor_index is None
			tensor_inputs.append(all_arg)
			input_arg.tensor_index = len(tensor_inputs) - 1

	unfolded_args = tuple(tensor_inputs) + (non_tensor_inputs,)

	def function_fold(*args):
		non_tensor_input = args[-1]
		print(non_tensor_inputs)
		# assert not(non_tensor_inputs is None)
		
		def replace_placeholder(arg):
			if isinstance(arg, TensorPlaceholder):
				assert not(arg.tensor_index is None)
				return args[arg.tensor_index]
			return arg
		
		all_inputs = recursive_apply(replace_placeholder, non_tensor_inputs)[0]
		return function(all_inputs)
	print(non_tensor_inputs)
	return StdCheckpointFunction.apply(function_fold, preserve, *unfolded_args)


class TorchCheckpointFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, run_function, callback, preserve_rng_state, *args):
		print('Fwd', args)
		chkpt.check_backward_validity(args)
		ctx.callback = callback
		ctx.run_function = run_function
		ctx.preserve_rng_state = preserve_rng_state
		ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
		if preserve_rng_state:
			ctx.fwd_cpu_state = torch.get_rng_state()
			# Don't eagerly initialize the cuda context by accident.
			# (If the user intends that the context is initialized later, within their
			# run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
			# we have no way to anticipate this will happen before we run the function.)
			ctx.had_cuda_in_fwd = False
			if torch.cuda._initialized:
				ctx.had_cuda_in_fwd = True
				ctx.fwd_gpu_devices, ctx.fwd_gpu_states = chkpt.get_device_states(*args)

		# Save non-tensor inputs in ctx, keep a placeholder None for tensors
		# to be filled out during the backward.
		ctx.inputs = []
		ctx.tensor_indices = []
		tensor_inputs = []
		for i, arg in enumerate(args):
			if torch.is_tensor(arg):
				tensor_inputs.append(arg)
				ctx.tensor_indices.append(i)
				ctx.inputs.append(None)
			else:
				ctx.inputs.append(arg)

		ctx.save_for_backward(*tensor_inputs)

		with torch.no_grad():
			outputs = run_function(*args)
		return outputs

	@staticmethod
	def backward(ctx, *args):
		print('Bwd', args)
		if not torch.autograd._is_checkpoint_valid():
			raise RuntimeError(
				"Checkpointing is not compatible with .grad() or when an `inputs` parameter"
				" is passed to .backward(). Please use .backward() and do not pass its `inputs`"
				" argument.")
		# Copy the list to avoid modifying original list.
		inputs = list(ctx.inputs)
		tensor_indices = ctx.tensor_indices
		tensors = ctx.saved_tensors

		# Fill in inputs with appropriate saved tensors.
		for i, idx in enumerate(tensor_indices):
			inputs[idx] = tensors[i]

		# Stash the surrounding rng state, and mimic the state that was
		# present at this time during forward.  Restore the surrounding state
		# when we're done.
		rng_devices = []
		if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
			rng_devices = ctx.fwd_gpu_devices
		with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
			if ctx.preserve_rng_state:
				torch.set_rng_state(ctx.fwd_cpu_state)
				if ctx.had_cuda_in_fwd:
					chkpt.set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
			detached_inputs = chkpt.detach_variable(tuple(inputs))
			with torch.enable_grad(), torch.cuda.amp.autocast(ctx.had_autocast_in_fwd):
				outputs = ctx.run_function(*detached_inputs)
		
		if isinstance(outputs, torch.Tensor):
			outputs = (outputs,)

		# run backward() with only tensor that requires grad
		outputs_with_grad = []
		args_with_grad = []
		for i in range(len(outputs)):
			if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
				outputs_with_grad.append(outputs[i])
				args_with_grad.append(args[i])
		if len(outputs_with_grad) == 0:
			raise RuntimeError(
				"none of output has requires_grad=True,"
				" this checkpoint() is not necessary")
		torch.autograd.backward(outputs_with_grad, args_with_grad)

		# ctx.callback()

		grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
					  for inp in detached_inputs)
		# print(grads)
		return (None, None, None) + grads

def torch_checkpoint(function, callback, *args, **kwargs):
	preserve = kwargs.pop('preserve_rng_state', True)
	if kwargs:
		raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

	return TorchCheckpointFunction.apply(function, callback, preserve, *args)