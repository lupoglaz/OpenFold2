import math
import torch
from torch.utils.data import Dataset, DataLoader
from equivariant_attention.from_se3cnn.utils_steerable import get_spherical_from_cartesian
import dgl
import numpy as np

def collate(samples):
	graphs, y = map(list, zip(*samples))
	batched_graph = dgl.batch(graphs)
	return batched_graph, torch.tensor(y)

class AtomDataset(Dataset):
	num_bonds = 2
	def __init__(self, data, block_size):
		r, v = data
		assert r.shape[0] == v.shape[0]
		assert r.shape[1] == v.shape[1]
		data_size, num_atoms = r.shape[0], r.shape[1]
		print(f'Data: {data_size} timesteps, {num_atoms} atoms')
		self.block_size = block_size
		self.num_atoms = num_atoms
		self.data_size = data_size
		self.data = data

	def connect_fully(self):
		adjacency = {}
		for i in range(self.num_atoms):
			for j in range(self.num_atoms):
				if i!=j:
					adjacency[(i,j)] = self.num_bonds - 1
		
		
		src = []
		dst = []
		w = []
		for edge, weight in adjacency.items():
			src.append(edge[0])
			dst.append(edge[1])
			w.append(weight)
		return np.array(src), np.array(dst), np.array(w)
	
	def to_one_hot(self, data, num_classes):
		one_hot = np.zeros(list(data.shape) + [num_classes])
		one_hot[np.arange(len(data)),data] = 1
		return one_hot

	def __getitem__(self, idx):
		r, v = self.data[0][idx,:,:].astype(np.float32), self.data[1][idx,:,:].astype(np.float32)
		m = np.ones((v.shape[0],1)).astype(np.float32)
		r_tgt, v_tgt = self.data[0][idx + self.block_size, :, :].astype(np.float32), self.data[1][idx + self.block_size, :, :].astype(np.float32)
		src, dst, w = self.connect_fully()
		w = self.to_one_hot(w, self.num_bonds).astype(np.float32)
		
		G = dgl.DGLGraph((src, dst))
		# v = get_spherical_from_cartesian(v).astype(np.float32)
		#Node features
		G.ndata['x'] = r
		G.ndata['f'] = np.expand_dims(np.concatenate([m,v],axis=1), axis=2)
		
		#Edge features
		G.edata['d'] = r[dst] - r[src]
		G.edata['w'] = w
		
		return G, (r_tgt - r).astype(np.float32)#get_spherical_from_cartesian(r_tgt - r).astype(np.float32)

	def __len__(self):
		return self.data_size - self.block_size