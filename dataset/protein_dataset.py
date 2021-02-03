import os
import sys

import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Angles2Coords, writePDB
from TorchProteinLibrary.Utils import ProteinStructure
from TorchProteinLibrary.FullAtomModel import Coords2Angles

import random
import argparse

class Patterns:
	def __init__(self):
		p2c = PDB2CoordsUnordered()
		fragment = p2c(["protein_samples/alpha.pdb"])
		alpha, length = Coords2Angles(*fragment)
				
		fragment = p2c(["protein_samples/alpha_link.pdb"])
		alpha_link, length = Coords2Angles(*fragment)
				
		fragment = p2c(["protein_samples/beta_sheet1.pdb"])
		beta, length = Coords2Angles(*fragment)
		
		self.patterns = {
			"alpha": alpha[:,:,1:4],
			"beta_left": beta[:,:,:8],
			"beta_right": beta[:,:,15:]
		}
		self.fragments = {
			"alpha_link": alpha_link[:,:,:7],
			# "beta_link": beta[:,:,8:15]
		}

		self.seq_fill = {
			"alpha": 'A',
			"beta_left": 'R',
			"beta_right": 'W',
			"alpha_link": 'G',
			"beta_link": 'F'
		}
		

	def add_pattern(self, name, length=10):
		sequence = ''.join([self.seq_fill[name] for i in range(length)])
		angles = torch.zeros(1, 8, length, dtype=torch.float, device='cpu')
		num_repeats = int(length / self.patterns[name].size(-1))
		angles.copy_(self.patterns[name].repeat(1, 1, num_repeats+1)[:,:,:length])
		return angles, sequence
	
	def add_fragment(self, name):
		length = self.fragments[name].size(-1)
		sequence = ''.join([self.seq_fill[name] for i in range(length)])
		angles = torch.zeros(1, 8, length, dtype=torch.float, device='cpu')
		angles.copy_(self.fragments[name])
		return angles, sequence

class Sampler:
	def __init__(self, patterns, 
					min_num_blocks=10, max_num_blocks=20,
					block_min_length=5, block_max_length=15):
		self.patterns = patterns
		self.min_num_blocks = min_num_blocks
		self.max_num_blocks = max_num_blocks
		self.block_min_length = block_min_length
		self.block_max_length = block_max_length
	
	def generate(self):
		angles = []
		sequence = ''
		block_types = ['pattern', 'fragment']
		num_blocks = random.randint(self.min_num_blocks, self.max_num_blocks)
		for i in range(num_blocks):
			block_length = random.randint(self.block_min_length, self.block_max_length)
			block_type = block_types[i%2]
			if block_type == 'pattern':
				pattern = random.choice(list(self.patterns.patterns.keys()))
				a, s = ptrn.add_pattern(name=pattern, length=block_length)
			elif block_type == 'fragment':
				link = random.choice(list(self.patterns.fragments.keys()))
				a, s = ptrn.add_fragment(name=link)
			angles.append(a)
			sequence = sequence + s
		angles = torch.cat(angles, dim=-1)
		sequence = [sequence]
		return angles, sequence

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-name', default='train', help='Dataset name', type=str)
	parser.add_argument('-size', default=100, help='Dataset size', type=int)
	args = parser.parse_args()

	import matplotlib 
	import matplotlib.pylab as plt
	import mpl_toolkits.mplot3d.axes3d as p3

	ptrn = Patterns()
	smpl = Sampler(ptrn, min_num_blocks=1, max_num_blocks=4)
	a2c = Angles2Coords()

	fig = plt.figure(figsize=plt.figaspect(0.3))
	with open(f'{args.name}/list.dat', 'wt') as fout:
		for i in range(args.size):
			angles, sequence = smpl.generate()
			prot = a2c(angles, sequence)
			writePDB(f'{args.name}/{i}.pdb', *prot)
			fout.write(f'{i}.pdb\n')

			if i < 3:
				ax = fig.add_subplot(1, 3, i+1, projection='3d')
				prot_ca = ProteinStructure(*prot).select_CA()
				atoms_plot = prot_ca.plot_coords(axis=ax)
	
	plt.show()
			