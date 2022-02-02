import torch
from torch import nn
from alphafold.Model.Opt import checkpointing as chkpt
from alphafold.Tests.utils import convert, check_recursive
import argparse
from pathlib import Path
from typing import Dict
from functools import partial

from torch.utils.checkpoint import checkpoint

class SimpleIteration(nn.Module):
	def __init__(self, f_in, f_mid, f_out) -> None:
		super(SimpleIteration, self).__init__()
		self.op = nn.Sequential(
			nn.Linear(f_in, f_mid),
			nn.ReLU(),
			nn.Linear(f_mid, f_mid),
			nn.ReLU(),
			nn.LayerNorm(f_mid),
			nn.Linear(f_mid, f_out)
		)
	def forward(self, x):
		tensor = self.op(x)
		return tensor

class SimpleStack(nn.Module):
	def __init__(self, f_in, f_mid, f_out, num_iter:int=5, checkpoint:bool=False) -> None:
		super(SimpleStack, self).__init__()
		self.checkpoint = checkpoint

		self.input_norm = nn.LayerNorm(f_in)
		self.input_projection = nn.Linear(f_in, f_mid)
		
		self.output_projection = nn.Linear(f_mid, f_out)
		self.relu = nn.ReLU()
		self.iterations = nn.ModuleList([SimpleIteration(f_mid, 4*f_mid, f_mid) for i in range(num_iter)])

	def forward(self, x):
		x = self.relu(self.input_projection(self.input_norm(x)))
		def func(a, index=None):
			return self.iterations[index](a)
		if self.checkpoint:
			for i, _ in enumerate(self.iterations):
				x = checkpoint(partial(func, index=i), x)
		else:
			for simple_iter in self.iterations:
				x = simple_iter(x)
		x = self.output_projection(x)
		return x

def CheckpointingTest(args, batch_size:int=64):
	x = torch.randn(batch_size, 128, dtype=torch.float32, device='cuda')
	model_chk = SimpleStack(128, 256, 128, checkpoint=True).to(dtype=torch.float32, device='cuda')
	model = SimpleStack(128, 256, 128, checkpoint=False).to(dtype=torch.float32, device='cuda')
	for param_tensor in model.state_dict():
		model_chk.state_dict()[param_tensor].data.copy_(model.state_dict()[param_tensor].data)

	model.train()
	model_chk.train()
	model_chk.zero_grad()
	for param in model_chk.parameters():
		print(param.size(), param.requires_grad)
	
	y = model(x)
	loss = torch.square(y).mean()
	loss.backward()

	y_chk = model_chk(x)
	loss_chk = torch.square(y_chk).mean()
	loss_chk.backward()
	
	for param, param_chk in zip(model.parameters(), model_chk.parameters()):
		print(param.size(), '\t', param.grad is None, '\t', param_chk.grad is None)
	
	for param, param_chk in zip(model.parameters(), model_chk.parameters()):
		if not torch.allclose(param.grad, param_chk.grad):
			print(param.grad.mean(), param_chk.grad.mean(), torch.max(torch.abs(param.grad - param_chk.grad)))

class DictIteration(nn.Module):
	def __init__(self, f_in, f_mid, f_out) -> None:
		super(DictIteration, self).__init__()
		self.op = nn.Sequential(
			nn.Linear(f_in, f_mid),
			nn.ReLU(),
			nn.Linear(f_mid, f_mid),
			nn.ReLU(),
			nn.LayerNorm(f_mid),
			nn.Linear(f_mid, f_out)
		)
	def forward(self, batch:Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
		return {'a': self.op(batch['a']), 'b': self.op(batch['b'])}

class DictStack(nn.Module):
	def __init__(self, f_in, f_mid, f_out, num_iter:int=2, checkpoint:bool=False) -> None:
		super(DictStack, self).__init__()
		self.checkpoint = checkpoint

		self.input_norm = nn.LayerNorm(f_in)
		self.input_projection = nn.Linear(f_in, f_mid)
		
		self.output_projection = nn.Linear(f_mid, f_out)
		self.relu = nn.ReLU()
		self.iterations = nn.ModuleList([DictIteration(f_mid, 4*f_mid, f_mid) for i in range(num_iter)])

	def forward(self, batch:Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
		a = self.relu(self.input_projection(self.input_norm(batch['a'])))
		b = self.relu(self.input_projection(self.input_norm(batch['b'])))
		x = {'a': a, 'b': b}
		if self.checkpoint:
			for simple_iter in self.iterations:
				def iter(batch):
					return simple_iter(batch)
				x = chkpt.checkpoint(iter, x)
				print(x['a'].grad_fn)
		else:
			for simple_iter in self.iterations:
				x = simple_iter(x)
		return {'a':self.output_projection(x['a']), 'b':self.output_projection(x['b'])}

def DictCheckpointingTest(args, batch_size:int=64):
	x = {	'a':torch.randn(batch_size, 128, dtype=torch.float32, device='cuda', requires_grad=True),
			'b':torch.randn(batch_size, 128, dtype=torch.float32, device='cuda', requires_grad=True)
		}
	model_chk = DictStack(128, 256, 128, checkpoint=True).to(dtype=torch.float32, device='cuda')
	# model = DictStack(128, 256, 128, checkpoint=False).to(dtype=torch.float32, device='cuda')
	# optimizer = torch.optim.Adagrad(model_chk.parameters())
	# for param_tensor in model.state_dict():
	# 	model_chk.state_dict()[param_tensor].data.copy_(model.state_dict()[param_tensor].data)

	# model.train()
	model_chk.train()
	
	
	# y = model(x)
	# loss = torch.square(y['a']).mean() + torch.square(y['b']).mean()
	# loss.backward()
	
	# for param in model.parameters():
	# 	print(param.grad.sum())
	
	y_chk = model_chk(x)
	loss_chk = torch.square(y_chk['a'] + y_chk['b']).mean()
	loss_chk.backward()
	# optimizer.step()

	for param in model_chk.parameters():
		print(param.grad.sum())

	# for param, param_chk in zip(model.parameters(), model_chk.parameters()):
		# assert torch.allclose(param.grad, param_chk.grad)


def checkpointing_tests(args):
	inputs = {	'a': torch.zeros(10, 10, requires_grad=True, device='cuda'),
				'b': torch.zeros(10, 10, requires_grad=False, device='cuda'),
				'c': [torch.zeros(10, 10, requires_grad=False), 'str', torch.zeros(10, 10, requires_grad=True, device='cuda')],
				'd': (torch.zeros(10, 10, requires_grad=True), 100)
				}
	new_inputs = chkpt.detach_variable(inputs)
	def no_grad(x):
		if isinstance(x, torch.Tensor):
			x.requires_grad = False
		return x
	
	chkpt.recursive_apply(no_grad, new_inputs)
	assert chkpt.check_backward_validity(inputs) == True
	assert chkpt.check_backward_validity(new_inputs) == False

	# for inp in chkpt.tree_walk(inputs):
	# 	print(inp)
	print(chkpt.get_device_states(inputs))
	


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-log_dir', default='Log', type=str)
		
	args = parser.parse_args()

	CheckpointingTest(args)
	# DictCheckpointingTest(args)

	# checkpointing_tests(args)