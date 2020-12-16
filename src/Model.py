import math
import logging 
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import GSE3Res, GNormSE3, GConvSE3, GAvgPooling, GMaxPooling, get_basis_and_r

class SE3TConfig:
	num_layers = None
	num_nlayers = 1
	num_degrees = 4
	edge_dim = 1
	div = 1
	pooling = 'avg'
	n_heads = 1

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

class SE3Transformer(nn.Module):
	"""SE(3) equivariant GCN with attention: https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/models.py"""
	def __init__(self, config):
		super().__init__()
		# Build the network
		self.config = config
		
		self.fibers = {'in': Fiber(structure=[(2, 1)]),
					   'mid': Fiber(structure=[(4, 0), (2, 1), (1, 2)]),
					   'out': Fiber(structure=[(1, 1)])}

		blocks = self._build_gcn(self.fibers, 1)
		self.Gblock, self.FCblock = blocks
		# print(self.Gblock)
		# print(self.FCblock)

	def _build_gcn(self, fibers, out_dim):
		# Equivariant layers
		Gblock = []
		fin = fibers['in']
		for i in range(self.config.num_layers):
			Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.config.edge_dim, 
								  div=self.config.div, n_heads=self.config.n_heads))
			Gblock.append(GNormSE3(fibers['mid']))
			fin = fibers['mid']
		Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.config.edge_dim))

		# Pooling
		if self.config.pooling == 'avg':
			Gblock.append(GAvgPooling())
		elif self.config.pooling == 'max':
			Gblock.append(GMaxPooling())

		# FC layers
		FCblock = []
		FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
		FCblock.append(nn.ReLU(inplace=True))
		FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

		return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

	def configure_optimizers(self, train_config):
		optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer


	def forward(self, G, y):
		# Compute equivariant weight basis from relative positions
		print(G)
		basis, r = get_basis_and_r(G, self.config.num_degrees-1)

		# encoder (equivariant layers)
		h = {'1': G.ndata['f']}
		for layer in self.Gblock:
			h = layer(h, G=G, r=r, basis=basis)

		# for layer in self.FCblock:
		# 	h = layer(h)
		print(h)
		return h
