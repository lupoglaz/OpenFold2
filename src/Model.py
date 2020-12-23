import sys
import math
import logging 
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)
import dgl
from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import GSE3Res, GNormSE3, GConvSE3, GAvgPooling, GMaxPooling, get_basis_and_r, GConvSE3Partial

class SE3TConfig:
	num_layers = 4
	num_degrees = 3
	edge_dim = 2
	div = 1
	n_heads = 4

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

class TutorialTransformer(nn.Module):
	def __init__(self, config):
		super().__init__()
		# Build the network
		self.config = config

		self.fibers = {'in': Fiber(structure=[(4, 0)]),
					   'mid': Fiber(structure=[(3, 0)]),
					   'out': Fiber(structure=[(3, 0)])}
		self.block = GSE3Res(self.fibers['in'], self.fibers['mid'], edge_dim=self.config.edge_dim, 
								  div=self.config.div, n_heads=self.config.n_heads)
		self.norm = GNormSE3(self.fibers['mid'])
		self.out = GConvSE3(self.fibers['mid'], self.fibers['out'], self_interaction=True, edge_dim=self.config.edge_dim)
        
	def configure_optimizers(self, train_config):
		optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer

	def forward(self, G, y):
		basis, r = get_basis_and_r(G, self.config.num_degrees-1)
		h = {'0': G.ndata['f']}
		y = self.block(h, G=G, r=r, basis=basis)
		z = self.norm(y, G=G, r=r, basis=basis)
		w = self.out(z, G=G, r=r, basis=basis)
		print(w)
		sys.exit()

class SE3Transformer(nn.Module):
	"""SE(3) equivariant GCN with attention: https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/models.py"""
	def __init__(self, config):
		super().__init__()
		# Build the network
		self.config = config
		
		self.fibers = {'in': Fiber(structure=[(1, 1)]),
					   'mid': Fiber(structure=[(32, 0), (16, 1), (4, 2)]),
					   'out': Fiber(structure=[(1, 1)])}

		self.Gblock = self._build_gcn(self.fibers, 1)
		# self.loss = torch.nn.MSE()
		
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

		return nn.ModuleList(Gblock)

	def configure_optimizers(self, train_config):
		optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer

	def forward(self, G, targets):
		# Compute equivariant weight basis from relative positions
		basis, r = get_basis_and_r(G, self.config.num_degrees-1)

		# encoder (equivariant layers)
		h = {'1': G.ndata['f'][:,1:,:].squeeze().unsqueeze(dim=1)}
		for layer in self.Gblock:
			h = layer(h, G=G, r=r, basis=basis)
		
		#Unbatching graphs
		G.ndata['o'] = h['1']
		vec = torch.stack([g.ndata['o'] for g in dgl.unbatch(G)], dim=0).squeeze(dim=2)
		
		loss = None
		if not (targets is None):
			loss = torch.sqrt(torch.sum((vec - targets)**2, dim=-1) + 1E-5)
			loss = torch.mean(loss, dim=-1)

		return vec, loss
