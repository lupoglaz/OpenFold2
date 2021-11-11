import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import math
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import numpy as np
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1


from TorchProteinLibrary.Utils import ProteinStructure
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Angles2Coords

from Bio.PDB import Polypeptide as pol


def _tensor2str(tensor):
	return (tensor.numpy().astype(dtype=np.uint8).tobytes().split(b'\00')[0]).decode("utf-8")

def collate(samples):
	input_msa, input_prot, target_batch, secondary = map(list, zip(*samples))
	input_graph = dgl.batch(input_prot)
	
	max_M = max(map(lambda x: x.size(0), input_msa))
	max_N = max(map(lambda x: x.size(1), input_msa))
	collate_msa = torch.zeros(len(input_msa), max_M, max_N, dtype=torch.long)
	for i, msa in enumerate(input_msa):
		collate_msa[i,:msa.size(0),:msa.size(1)] = msa

	max_M = max(map(lambda x: x.size(0), secondary))
	max_N = max(map(lambda x: x.size(1), secondary))
	collate_secondary = torch.zeros(len(secondary), max_M, max_N, dtype=torch.bool)
	num_secondary = torch.zeros(len(secondary), dtype=torch.int)
	for i, sec in enumerate(secondary):
		collate_secondary[i,:sec.size(0),:sec.size(1)] = sec
		num_secondary[i] = sec.size(0)


	max_size = max([tgt.size(1) for tgt in target_batch])
	target_tensor = []
	for tgt in target_batch:
		if tgt.size(1)<max_size:
			target = torch.cat([tgt, torch.zeros(1, max_size-tgt.size(1), dtype=tgt.dtype, device=tgt.device)], dim=1)
			target_tensor.append(target)
		else:
			target_tensor.append(tgt)
	target_coords = torch.cat(target_tensor, dim=0)
	return collate_msa, input_graph, target_coords, collate_secondary, num_secondary

class MSADataset(Dataset):
	num_bonds = 2
	def __init__(self, list_path):
		self.data = []
		self.msa = []
		self.secondary = []
		dir_path = list_path.parents[0]
		with open(list_path, 'rt') as fin:
			for line in fin:
				sline = line.split()
				if len(sline) == 0:
					break
				self.data.append( dir_path.joinpath(Path(sline[0])).as_posix() ) 
				self.msa.append( dir_path.joinpath(Path(sline[0]).with_suffix('.msa')).as_posix() )
				self.secondary.append(dir_path.joinpath(Path(sline[0]).with_suffix('.th')).as_posix())
		
		self.data_size = len(self.data)
		self.p2c = PDB2CoordsUnordered()
		self.a2c = Angles2Coords()

		self.aa_dict = {}
		for num, aa in enumerate(list('-'+pol.aa1)):
			self.aa_dict[aa] = num


		print(f'Data: {self.data_size} proteins')
	
	def load_secondary(self, path):
		with open(path, 'rb') as fin:
			secondary = torch.load(fin)
		return secondary.view(secondary.size(0), secondary.size(1)*3)

	def load_msa(self, path):
		msa = []
		with open(path, 'rt') as fin:
			for line in fin:
				if len(line) == 0:
					break
				msa.append(line.split()[0])
		
		M = len(msa)
		N = len(msa[0])
		idx_msa = torch.zeros(M, N, dtype=torch.long)
		for i in range(M):
			for j in range(N):
				idx_msa[i,j] = self.aa_dict[msa[i][j]]
		
		return idx_msa

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
	
	def to_idx(self, data):
		idxs = torch.zeros(data.size(1), 1)
		seq = ''
		for i in range(data.size(1)):
			residue_3name = _tensor2str(data[0,i,:])
			idxs[i] = d3_to_index[residue_3name]
			seq = seq + dindex_to_1[idxs[i].item()]
		return idxs, seq

	def __getitem__(self, idx):
		#MSA
		msa = self.load_msa(self.msa[idx])
		secondary = self.load_secondary(self.secondary[idx])
		
		#Target conformation
		prot_tgt = self.p2c([self.data[idx]])
		#Extracting sequence, number of residues
		prot_tgt_CA = ProteinStructure(*prot_tgt).select_CA()
		num_res = prot_tgt_CA.resnames.size(1)
		residx, seq = self.to_idx(prot_tgt_CA.resnames)

		#Complete backbone
		prot_tgt_bkb = ProteinStructure(*prot_tgt).select_backbone()
		
		#Open initial conformation
		angles = torch.zeros(1, 8, len(seq), dtype=torch.float32)
		prot_init = self.a2c(angles, [seq])
		prot_init = ProteinStructure(*prot_init).select_CA()
		x_init = prot_init.coords.view(num_res, 3)
		
		src, dst, w = self.connect_fully(num_res)
		
		G = dgl.DGLGraph((src, dst))
		G.ndata['x'] = x_init.to(torch.float32)
		G.ndata['s'] = residx.to(torch.long)
		G.edata['d'] = (x_init[dst] - x_init[src]).to(torch.float32)
		G.edata['w'] = (torch.from_numpy(w)).to(torch.float32).unsqueeze(dim=-1)
		
		return msa, G, prot_tgt_bkb.coords.to(dtype=torch.float32), secondary

	def __len__(self):
		return self.data_size

if __name__=='__main__':
	import _pickle as pkl
	
	dataset = MSADataset(Path('../dataset/test/list.dat'))

	stream = DataLoader(dataset, shuffle=True, pin_memory=True, 
						batch_size=4, num_workers=0, collate_fn=collate)

	for MSA_inp, G_inp, G_tgt, secondary, num_secondary in stream:
		print(MSA_inp)
		print(G_inp)
		print(G_tgt)
		print(secondary)
		print(num_secondary)
		break

	batch_idx = 1
	for sec_idx in range(num_secondary[batch_idx].item()):
		sec = secondary[batch_idx, sec_idx, :]
		sec = sec.view(int(sec.size(0)/3), 3)
		str = ''
		for k in range(sec.size(0)):
			if sec[k,0]:
				str += '1'
			else:
				str += '0'
		print(str)