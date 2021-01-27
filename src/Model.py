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

from .Cartesian2Spherical import Cartesian2Spherical
from .Basis import Basis
from .SphericalHarmonics import SphericalHarmonics
from .StructureModule import StructureModule

from TorchProteinLibrary.RMSD import Coords2RMSD

class SE3TConfig:
	embedding_dim = 32
	num_layers = 4
	num_degrees = 3
	edge_dim = 2
	div = 1
	n_heads = 4
	num_iter = 4

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)


class SE3TransformerIt(nn.Module):
	"""Iterative SE(3) equivariant GCN with attention: https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/models.py"""
	def __init__(self, config):
		super().__init__()
		# Build the network
		self.config = config
		
		self.fibers = {'in': Fiber(structure=[(self.config.embedding_dim, 0), (2, 1)]),
					   'mid': Fiber(structure=[(32, 0), (16, 1), (4, 2)]),
					   'out': Fiber(structure=[(self.config.embedding_dim, 0), (2, 1)])}

		self.Gblock = self._build_gcn(self.fibers, 1)
		self.sph_har = SphericalHarmonics(max_degree=2*(self.config.num_degrees))
		self.car2sph = Cartesian2Spherical()
		self.basis = Basis(max_degree=self.config.num_degrees)
		# self.loss = torch.nn.MSE()
		self.embedding = nn.Embedding(20, self.config.embedding_dim)
		self.structure = StructureModule()
		self.loss = Coords2RMSD()

	def _build_gcn(self, fibers, out_dim):
		# Equivariant layers
		Gblock = []
		fin = fibers['in']
		for i in range(self.config.num_layers):
			Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.config.embedding_dim, 
								  div=self.config.div, n_heads=self.config.n_heads))
			Gblock.append(GNormSE3(fibers['mid']))
			fin = fibers['mid']
		Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.config.embedding_dim))

		return nn.ModuleList(Gblock)

	def configure_optimizers(self, train_config):
		optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer

	def forward(self, G, target):		
		G.ndata['f'] = self.embedding(G.ndata['s'].squeeze())
		src, dst = G.all_edges()
		G.edata['w'] = G.ndata['f'][src] * G.ndata['f'][dst]
		
		for iter in range(self.config.num_iter):
			#Equivariant basis
			r_sp = self.car2sph(G.edata['d'])
			Y = self.sph_har(r_sp)
			basis = self.basis(Y)
			
			#First input, zeroing empty inputs
			if iter == 0:
				h = {'0': G.ndata['f'].unsqueeze(dim=-1),
					'1': torch.zeros(G.ndata['f'].size(0), 2, 3, dtype=G.ndata['f'].dtype, device=G.ndata['f'].device)
					}
			#Transformer blocks
			for layer in self.Gblock:
				h = layer(h, G=G, r=r_sp[:, self.car2sph.ind_radius].unsqueeze(dim=-1), basis=basis)
				
			#Updating coordinates
			src, dst = G.all_edges()
			G.ndata['x'] = G.ndata['x'] + h['1'][:,0,:]
			G.edata['d'] = G.ndata['x'][dst] - G.ndata['x'][src]
		
		#Converting vector fields into structure
		G.ndata['r'] = self.structure(h['1'])
		
		#Unbatching graphs
		batch_size = len(list(dgl.unbatch(G)))
		max_num_res = max([g.number_of_nodes() for g in dgl.unbatch(G)])
		num_atoms = torch.zeros(batch_size, dtype=torch.int, device=G.ndata['r'].device)
		max_num_atoms = 3*max_num_res
		structs = []
		for i, g in enumerate(dgl.unbatch(G)):
			num_res = g.ndata['r'].size(0)
			num_atoms[i] = 3*num_res
			struct = g.ndata['r'].contiguous().flatten().unsqueeze(dim=0)
			if num_atoms[i].item() < max_num_atoms:
				struct = torch.cat([struct, torch.zeros(1, (max_num_atoms-num_atoms[i].item())*3, dtype=struct.dtype, device=struct.device) ], dim=1)
			structs.append(struct)

		structs = torch.cat(structs, dim=0)
		
		loss = None
		if not (target is None):
			losses = self.loss(structs, target, num_atoms)

		return structs, losses
