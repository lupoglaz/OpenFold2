import torch
from torch import nn

class AlphaFoldIteration(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
	
	def forward(self, ensembled_batch, non_ensembled_batch, is_training, 
				compute_loss=False, ensemble_representations=False, return_representations=False):
		raise Exception(NotImplemented)

class AlphaFold(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.impl = AlphaFoldIteration(self.config)

	def forward(self, batch, is_training, 
				compute_loss=False, ensemble_representations=False, return_representations=False):
		raise Exception(NotImplemented)
		# print(batch['aatype'])
		# batch_size, num_residues = batch['aatype']