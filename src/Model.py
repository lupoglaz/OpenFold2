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
	embedding_dim = 10
	num_degrees = 3
	edge_dim = 2
	div = 1
	n_heads = 2
	num_iter = 4

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

class MSAConfig:
	embd_pdrop = 0.1
	resid_pdrop = 0.1
	attn_pdrop = 0.1
	n_layer = 1
	n_head = 2
	n_embd = 10
	vocab_size = 21


	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)


class ColumnAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0

		self.key = nn.Linear(config.n_embd, config.n_embd)
		self.query = nn.Linear(config.n_embd, config.n_embd)
		self.value = nn.Linear(config.n_embd, config.n_embd)
		
		self.attn_drop = nn.Dropout(config.attn_pdrop)
		self.resid_drop = nn.Dropout(config.resid_pdrop)

		self.proj = nn.Linear(config.n_embd, config.n_embd)

		# self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size) )
		self.n_head = config.n_head

	def forward(self, x):
		B, M, N, C = x.size()
		x = x.transpose(1,2) #We are attending over columns of MSA
		
		k = self.key(x).view(B, N, M, self.n_head, C//self.n_head).transpose(2,3)
		q = self.query(x).view(B, N, M, self.n_head, C//self.n_head).transpose(2,3)
		v = self.value(x).view(B, N, M, self.n_head, C//self.n_head).transpose(2,3)
		
		att = (q @ k.transpose(-2, -1)) * (1.0 /math.sqrt(k.size(-1)))
		att = F.softmax(att, dim=-1)
		att = self.attn_drop(att)
		y = att @ v
				
		y = y.transpose(2,3).contiguous().view(B, N, M, C)
		y = self.resid_drop(self.proj(y))
		return y.transpose(1,2)

class RowAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0

		self.value = nn.Linear(config.n_embd, config.n_embd)
		
		self.attn_drop = nn.Dropout(config.attn_pdrop)
		self.resid_drop = nn.Dropout(config.resid_pdrop)

		self.proj = nn.Linear(config.n_embd, config.n_embd)

		# self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size) )
		self.n_head = config.n_head

	def forward(self, x, att):
		B, M, N, C = x.size()		
		v = self.value(x).view(B, M, N, self.n_head, C//self.n_head).transpose(2,3)

		att = att * (1.0 /math.sqrt(v.size(-1)))
		att = self.attn_drop(att)
		att = att.unsqueeze(dim=1).unsqueeze(dim=2)
		y = att @ v
						
		y = y.transpose(2,3).contiguous().view(B, M, N, C)
		y = self.resid_drop(self.proj(y))
		return y

class MSABlock(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.ln = nn.LayerNorm(config.n_embd)
		self.col_attn = ColumnAttention(config)
		self.row_attn = RowAttention(config)

		self.mlp = nn.Sequential(
			nn.Linear(config.n_embd, 4*config.n_embd),
			nn.GELU(),
			nn.Linear(4*config.n_embd, config.n_embd),
			nn.Dropout(config.resid_pdrop)
		)

	def forward(self, x, att):
		y = self.col_attn(self.ln(x))
		x = x + y
		x = x + self.row_attn(self.ln(x), att)
		x = x + self.mlp(self.ln(x))
				
		return x, y[:,0,:,:]

class GCNBlock(nn.Module):
	def __init__(self, config, fibers_in, fibers_mid, fibers_out):
		super().__init__()
		self.gse3res1 = GSE3Res(fibers_in, fibers_mid, edge_dim=config.embedding_dim, div=config.div, n_heads=config.n_heads)
		self.gse3res2 = GSE3Res(fibers_mid, fibers_mid, edge_dim=config.embedding_dim, div=config.div, n_heads=config.n_heads)
		self.gse3norm1 = GNormSE3(fibers_mid)
		self.gse3norm2 = GNormSE3(fibers_mid)
		self.gse3conv = GConvSE3(fibers_mid, fibers_out, self_interaction=True, edge_dim=config.embedding_dim)
	
	def forward(self, h, G, r, basis):
		h = self.gse3res1(h, G=G, r=r, basis=basis)
		h = self.gse3norm1(h, G=G, r=r, basis=basis)
		h = self.gse3res2(h, G=G, r=r, basis=basis)
		h = self.gse3norm2(h, G=G, r=r, basis=basis)
		h = self.gse3conv(h, G=G, r=r, basis=basis)
		return h

class SE3TransformerIt(nn.Module):
	"""Iterative SE(3) equivariant GCN with attention: https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/models.py"""
	def __init__(self, se3config, msaconfig):
		super().__init__()
		# Build the network
		self.se3config = se3config
		self.msaconfig = msaconfig
		
		self.fibers = {'in': Fiber(structure=[(self.se3config.embedding_dim + msaconfig.n_embd, 0), (2, 1)]),
					   'mid': Fiber(structure=[(32 + msaconfig.n_embd, 0), (6, 1), (4, 2)]),
					   'out': Fiber(structure=[(self.se3config.embedding_dim, 0), (2, 1)])}

		self.Gblock, self.Tblock = self._build_gcn(self.fibers, 1)
		self.sph_har = SphericalHarmonics(max_degree=2*(self.se3config.num_degrees))
		self.car2sph = Cartesian2Spherical()
		self.basis = Basis(max_degree=self.se3config.num_degrees)
		
		self.structure = StructureModule()
		self.loss = Coords2RMSD()
		self.relu = nn.ReLU()

		#amino-acid embeddings
		self.embedding = nn.Embedding(21, self.se3config.embedding_dim)
		self.msa_embedding = nn.Embedding(21, msaconfig.n_embd)
		
		#positional emebeddings
		self.msa_pos_embedding_x = nn.Embedding(80, msaconfig.n_embd)
		self.msa_pos_embedding_y = nn.Embedding(10, msaconfig.n_embd)
		self.x = torch.arange(0, 80, dtype=torch.long).unsqueeze(dim=0).repeat(10,1)	
		self.y = torch.arange(0, 10, dtype=torch.long).unsqueeze(dim=1).repeat(1,80)

	def _build_gcn(self, fibers, out_dim):
		# Equivariant layers
		Gblock = []
		Tblock = []
		for i in range(self.se3config.num_iter):
			Gblock.append(GCNBlock(self.se3config, fibers['in'], fibers['mid'], fibers['out']))
			Tblock.append(MSABlock(self.msaconfig))
				
		return nn.ModuleList(Gblock), nn.ModuleList(Tblock)

	def configure_optimizers(self, train_config):
		optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer

	def forward(self, msa, G, target):
		#Sequence embedding
		G.ndata['f'] = self.embedding(G.ndata['s'].squeeze())
		src, dst = G.all_edges()
		G.edata['w'] = G.ndata['f'][src] * G.ndata['f'][dst]

		#MSA embedding + positional embeddings
		self.x = self.x.to(device=msa.device)
		self.y = self.y.to(device=msa.device)
		pos_emb = self.msa_pos_embedding_x(self.x) + self.msa_pos_embedding_y(self.y)
		pos_emb = pos_emb.unsqueeze(dim=0)
		msa_emb = self.msa_embedding(msa) + pos_emb[:,:,:msa.size(2)]
		
		for iter in range(self.se3config.num_iter):
			#Equivariant basis
			r_sp = self.car2sph(G.edata['d'])
			Y = self.sph_har(r_sp)
			basis = self.basis(Y)
	
			#First input, zeroing empty inputs
			if iter == 0:
				zer = torch.zeros(G.ndata['f'].size(0), self.msaconfig.n_embd, 1, dtype=G.ndata['f'].dtype, device=G.ndata['f'].device)
				h = {'0': torch.cat([G.ndata['f'].unsqueeze(dim=-1), zer], dim=1),
					'1': torch.zeros(G.ndata['f'].size(0), 2, 3, dtype=G.ndata['f'].dtype, device=G.ndata['f'].device)
					}
			
			#Structural block
			h = self.Gblock[iter](h, G=G, r=r_sp[:, self.car2sph.ind_radius].unsqueeze(dim=-1), basis=basis)

			#Updating coordinates
			src, dst = G.all_edges()
			G.ndata['x'] = G.ndata['x'] + h['1'][:,0,:]
			G.edata['d'] = G.ndata['x'][dst] - G.ndata['x'][src]
			G.edata['att'] = self.relu(12.0 - torch.sqrt( (G.edata['d']*G.edata['d']).sum(dim=1) ))/12.0

			#Converting distances to row attention matrix
			batch_size = len(list(dgl.unbatch(G)))
			max_num_res = max([g.number_of_nodes() for g in dgl.unbatch(G)])
			att_mat = torch.zeros(batch_size, max_num_res, max_num_res, device=msa_emb.device, dtype=msa_emb.dtype)
			for i, g in enumerate(dgl.unbatch(G)):
				src, dst = g.all_edges()
				ind = torch.arange(0, g.edata['att'].size(0), dtype=torch.long, device=msa_emb.device)
				att_mat[i, src, dst] = g.edata['att'][ind]

			#MSA block
			msa_emb, node_emb = self.Tblock[iter](msa_emb, att_mat)
			node_feat = []
			for i, g in enumerate(dgl.unbatch(G)):
				ind = g.nodes()
				node_feat.append(node_emb[i, ind, :])
			node_feat = torch.cat(node_feat, dim=0).unsqueeze(dim=-1)
			h['0'] = torch.cat([h['0'], node_feat], dim=1)
						
		
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

		return losses, structs, num_atoms
