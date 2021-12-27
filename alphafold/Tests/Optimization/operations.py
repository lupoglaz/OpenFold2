import torch
from torch import nn
# from alphafold.Model.Opt import checkpointing
from alphafold.Tests.Model.quaternion_test import convert, check_recursive
import argparse
from pathlib import Path
from typing import Dict

from torch.utils.checkpoint import checkpoint
import deepspeed as ds
# from deepspeed.checkpointing import checkpoint

class SimpleIteration(nn.Module):
	def __init__(self, f_in, f_mid, f_out) -> None:
		super(SimpleIteration, self).__init__()
		self.op = nn.Sequential(
			nn.Linear(f_in, f_mid),
			nn.ReLU(),
			nn.Linear(f_mid, f_mid),
			nn.ReLU(),
			# nn.LayerNorm(f_mid),
			nn.Linear(f_mid, f_out)
		)
	def forward(self, x):
		return self.op(x)

class SimpleStack(nn.Module):
	def __init__(self, f_in, f_mid, f_out, num_iter:int=10, checkpoint:bool=False) -> None:
		super(SimpleStack, self).__init__()
		self.checkpoint = checkpoint

		# self.input_norm = nn.LayerNorm(f_in)
		self.input_projection = nn.Linear(f_in, f_mid)
		
		self.output_projection = nn.Linear(f_mid, f_out)
		self.relu = nn.ReLU()
		self.iterations = nn.ModuleList([SimpleIteration(f_mid, 4*f_mid, f_mid) for i in range(num_iter)])

	def forward(self, x):
		x = self.relu(self.input_projection(x))#self.input_norm(x)))
		if self.checkpoint:
			for simple_iter in self.iterations:
				if ds.checkpointing.is_configured():
					x = ds.checkpointing.CheckpointFunction(lambda batch: simple_iter(batch), x)
				else:
					x = checkpoint(lambda batch: simple_iter(batch), x)
		else:
			for simple_iter in self.iterations:
				x = simple_iter(x)
		x = self.output_projection(x)
		return x

def CheckpointingTest(args, batch_size:int=64):
	# ds.checkpointing.configure()
	x = torch.randn(batch_size, 128, dtype=torch.float32, device='cuda')
	model_chk = SimpleStack(128, 256, 128, checkpoint=True).to(dtype=torch.float32, device='cuda')
	model = SimpleStack(128, 256, 128, checkpoint=False).to(dtype=torch.float32, device='cuda')
	for param_tensor in model.state_dict():
		model_chk.state_dict()[param_tensor].data.copy_(model.state_dict()[param_tensor].data)

	model.train()
	model_chk.train()
	
	y = model(x)
	loss = torch.square(y).mean()
	loss.backward()

	y_chk = model_chk(x)
	loss_chk = torch.square(y_chk).mean()
	loss_chk.backward()
	
	for param, param_chk in zip(model.parameters(), model_chk.parameters()):
		print(param.size(), '\t', param.grad is None, '\t', param_chk.grad is None)
	
	for param, param_chk in zip(model.parameters(), model_chk.parameters()):
		# print(param_chk.grad, param.grad)
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
	def __init__(self, f_in, f_mid, f_out, num_iter:int=10, checkpoint:bool=False) -> None:
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
				x = checkpoint(lambda batch: simple_iter(batch), x)
		else:
			for simple_iter in self.iterations:
				x = simple_iter(x)
		x = {'a':self.output_projection(x['a']), 'b':self.output_projection(x['b'])}
		return x



def DictCheckpointingTest(args, batch_size:int=64):
	x = {	'a':torch.randn(batch_size, 128, dtype=torch.float32, device='cuda'),
			'b':torch.randn(batch_size, 128, dtype=torch.float32, device='cuda')
		}
	# model_chk = DictStack(128, 256, 128, checkpoint=True).to(dtype=torch.float32, device='cuda')
	model = DictStack(128, 256, 128, checkpoint=False).to(dtype=torch.float32, device='cuda')
	# for param_tensor in model.state_dict():
	# 	model_chk.state_dict()[param_tensor].data.copy_(model.state_dict()[param_tensor].data)

	model.train()
	# model_chk.train()
	
	y = model(x)
	loss = torch.square(y['a']).mean() + torch.square(y['b']).mean()
	loss.backward()
	
	for param in model.parameters():
		print(param.grad.sum())

	# y_chk = model_chk(x)
	# loss_chk = torch.square(y_chk).mean()
	# loss_chk.backward()

	# for param, param_chk in zip(model.parameters(), model_chk.parameters()):
		# assert torch.allclose(param.grad, param_chk.grad)


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-log_dir', default='Log', type=str)
		
	args = parser.parse_args()

	CheckpointingTest(args)
	# DictCheckpointingTest(args)
