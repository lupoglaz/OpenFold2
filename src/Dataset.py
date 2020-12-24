import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
from torch.utils.data import Dataset, DataLoader
from equivariant_attention.from_se3cnn.utils_steerable import get_spherical_from_cartesian
import dgl
import numpy as np



def collate(samples):
	input, target = map(list, zip(*samples))
	input_graph = dgl.batch(input)
	target_graph = dgl.batch(target)
	return input_graph, target_graph

class AtomDataset(Dataset):
	num_bonds = 2
	def __init__(self, data, block_size):
		self.data = []
		for r,v in data:
			assert r.shape[0] == v.shape[0]
			assert r.shape[1] == v.shape[1]
			Nt = r.shape[0]
			num_atoms = r.shape[1]
			src, dst, w = self.connect_fully(num_atoms)
			for timestep in range(Nt-block_size):
				datapoint = r[timestep,:,:], v[timestep,:,:], r[timestep+block_size,:,:], src, dst, w
				self.data.append(datapoint)
		
		self.data_size = len(self.data)
		print(f'Data: {self.data_size} timesteps')

	def connect_fully(self, num_atoms):
		adjacency = {}
		for i in range(num_atoms):
			for j in range(num_atoms):
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
		r, v, r_tgt, src, dst, w = self.data[idx]
		m = np.ones((v.shape[0],1))
		w = self.to_one_hot(w, self.num_bonds)
		
		G = dgl.DGLGraph((src, dst))
		#Node features
		G.ndata['x'] = r.astype(np.float32)
		G.ndata['f'] = np.expand_dims(np.concatenate([m,v],axis=1), axis=2).astype(np.float32)
		
		#Edge features
		G.edata['d'] = (r[dst] - r[src]).astype(np.float32)
		G.edata['w'] = w.astype(np.float32)
		
		G_tgt = dgl.DGLGraph((src, dst))
		G_tgt.ndata['d'] = (r_tgt - r).astype(np.float32)
		return G, G_tgt

	def __len__(self):
		return self.data_size

if __name__=='__main__':
	import _pickle as pkl
	block_size = 128
	with open('../dataset/data.pkl', 'rb') as fin:
		data = pkl.load(fin)

	dataset = AtomDataset(data, block_size)

	stream = DataLoader(dataset, shuffle=True, pin_memory=True, 
						batch_size=1, num_workers=0, collate_fn=collate)

	for G_inp, G_tgt in stream:
		print(G_inp)
		print(G_tgt)
		break