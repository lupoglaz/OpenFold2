import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import torch.nn as nn
import dgl
import random

from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import get_basis_and_r, get_basis
from equivariant_attention.from_se3cnn import utils_steerable

def make_neighbour_graph(coords, num_atoms, cutoff=1.0, f=None):
	batch_size = coords.size(0)
	max_num_atoms = coords.size(1)

	mask = torch.ones(batch_size, max_num_atoms, dtype=torch.bool)
	for i in range(batch_size):
		mask[i, num_atoms[i].item():] = False
	mask = mask.unsqueeze(dim=1)*mask.unsqueeze(dim=2)
	mask[:, torch.arange(0,max_num_atoms), torch.arange(0,max_num_atoms)] = False

	dr = coords.unsqueeze(dim=1) - coords.unsqueeze(dim=2)
	dist = torch.sqrt((dr*dr).sum(dim=-1))
	adj = (dist < cutoff).bitwise_and(mask)

	graphs = []
	for i in range(batch_size):
		edges = torch.nonzero(adj[i,:,:], as_tuple=False)
		src = torch.cat([edges[:,0], edges[:,1]], dim=0)
		dst = torch.cat([edges[:,1], edges[:,0]], dim=0)
		
		G = dgl.DGLGraph()
		G.add_nodes(num_atoms[i].item())
		G.add_edges(src, dst)
		
		G.ndata['x'] = coords[i, :num_atoms[i].item(), :]
		G.edata['d'] = coords[i,dst,:] - coords[i,src,:]

		if not(f is None):
			G.ndata['f'] = f[i, :num_atoms[i].item(), :]
		
		graphs.append(G)

	return dgl.batch(graphs)
		

if __name__=='__main__':
	
	max_num_atoms = 10
	batch_size = 3
	num_atoms = torch.zeros(batch_size, dtype=torch.long)
	r = torch.zeros(batch_size, max_num_atoms, 3, dtype=torch.float32)
	dr = torch.zeros(batch_size, max_num_atoms, 3, dtype=torch.float32)
	for i in range(batch_size):
		num_atoms[i] = random.randint(3, 10)
		r[i, :num_atoms[i].item(), :].copy_(3*torch.rand(num_atoms[i].item(), 3))
		dr[i, :num_atoms[i].item(), :].copy_(0.5*torch.rand(num_atoms[i].item(), 3))

	graphs = make_neighbour_graph(r, num_atoms, f=dr)
	graphs.ndata['x'].requires_grad_()
	graphs.edata['d'].requires_grad_()
	print(graphs)
	
