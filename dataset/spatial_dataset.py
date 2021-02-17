import os
import sys

import numpy as np
import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Angles2Coords, writePDB, CoordsTranslate, CoordsRotate, Coords2Center
from TorchProteinLibrary.Utils import ProteinStructure
from TorchProteinLibrary.FullAtomModel import Coords2Angles
from TorchProteinLibrary.RMSD import Coords2RMSD

from TorchProteinLibrary.Volume import RzRxRz

from matplotlib import pyplot as plt
from celluloid import Camera
from tqdm import tqdm

import random
import argparse

def getResNumAtoms(res_name):
	if res_name == "G":
		return 4
	elif res_name == "A":
		return 5
	elif res_name == "S":
		return 6
	elif res_name == "C":
		return 6
	elif res_name == "V":
		return 7
	elif res_name == "I":
		return 8
	elif res_name == "L":
		return 8
	elif res_name == "T":
		return 7
	elif res_name == "R":
		return 11
	elif res_name == "K":
		return 9
	elif res_name == "D":
		return 8
	elif res_name == "N":
		return 8
	elif res_name == "E":
		return 9
	elif res_name == "Q":
		return 9
	elif res_name == "M":
		return 8
	elif res_name == "H":
		return 10
	elif res_name == "P":
		return 7
	elif res_name == "F":
		return 11
	elif res_name == "Y":
		return 12
	elif res_name == "W":
		return 14

def getSeqNumAtoms(seq):
	num_atoms = 0
	for res in seq:
		num_atoms += getResNumAtoms(res)
	return num_atoms
	
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
			# "beta_left": beta[:,:,:6],
			# "beta_right": beta[:,:,15:]
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

		self.displacement_type = {
			'N': torch.tensor([[5, 5, 0]], dtype=torch.float32),
			'Q': torch.tensor([[5, -5, 0]], dtype=torch.float32)
		}
		self.rotation_type = {
			'N': torch.tensor([[1.6, 0.3, 0]], dtype=torch.float32),
			'Q': torch.tensor([[1.6, 0.3, 0]], dtype=torch.float32)
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
		self.linkers = []
		self.blocks = []
		self.blocks_masks = []
				
		block_ind = []
		linkers_ind = []
		frag_ind_seq = []
		glob_ind_coord = 0
		glob_ind_seq = 0

		angles = []
		sequence = ''
		block_types = ['pattern', 'fragment']
		num_blocks = random.randint(self.min_num_blocks, self.max_num_blocks)
		for i in range(num_blocks):
			block_length = random.randint(self.block_min_length, self.block_max_length)
			block_type = block_types[i%2]
			displ_type = random.randint(1,2)
			if block_type == 'pattern':
				pattern = random.choice(list(self.patterns.patterns.keys()))
				a, s = ptrn.add_pattern(name=pattern, length=block_length)
				
				#Amino-acid that signals the displacement of fragments
				displ_type = random.choice(list(self.patterns.displacement_type.keys()))
				s = displ_type + s[1:]

				self.blocks.append((a,[s]))
				block_ind.append((glob_ind_coord, glob_ind_coord + getSeqNumAtoms(s)))
				glob_ind_coord += getSeqNumAtoms(s)
				glob_ind_seq += len(s)

			elif block_type == 'fragment':
				link = random.choice(list(self.patterns.fragments.keys()))
				a, s = ptrn.add_fragment(name=link)
				
				self.linkers.append((a,[s]))
				linkers_ind.append((glob_ind_coord, glob_ind_coord + getSeqNumAtoms(s)))
				glob_ind_coord += getSeqNumAtoms(s)
				frag_ind_seq.append((glob_ind_seq, glob_ind_seq + len(s)))
				glob_ind_seq += len(s)

			angles.append(a)
			sequence = sequence + s
		
		total_num_atoms = getSeqNumAtoms(sequence)
		for i in range(len(self.blocks)):
			mask = torch.zeros(1, total_num_atoms, 3, dtype=torch.bool)
			ind_start, ind_end = block_ind[i]
			mask[0, ind_start:ind_end, :].fill_(True)
			self.blocks_masks.append(mask.clone())
		
		self.grad_mask = torch.ones(1, 8, len(sequence), dtype=torch.bool)
		for ind_start, ind_end in frag_ind_seq:
			self.grad_mask[0, :, ind_start:ind_end].fill_(False)
		
		angles = torch.cat(angles, dim=-1)
		sequence = [sequence]
		return angles, sequence

	def position(self, visualize=False):
		a2c = Angles2Coords()
		rzrxrz = RzRxRz()
		translate = CoordsTranslate()
		rotate = CoordsRotate()
		center = Coords2Center()
		
		if visualize:
			fig = plt.figure(figsize=plt.figaspect(1))
			ax = fig.add_subplot(1, 1, 1, projection='3d')

		T = torch.tensor([[0, 0, 0]], dtype=torch.float32)
		
		self.target_coords = []
		self.target_num_atoms = None
		self.target_mask = None
				
		for i, (angles, sequence) in enumerate(smpl.blocks):
			prot = a2c(angles, sequence)
			coords, num_atoms = prot[0], prot[-1]
			cent = center(coords, num_atoms)

			coords_cent = translate(coords, -cent, num_atoms)
			R = rzrxrz(self.patterns.rotation_type[sequence[0][0]]).contiguous()
			coords_rot = rotate(coords_cent, R, num_atoms)
			if i%2 == 1:
				R = rzrxrz(torch.tensor([[0, np.pi, 0]], dtype=torch.float32)).contiguous()
				coords_rot = rotate(coords_rot, R, num_atoms)

			
			coords_new = translate(coords_rot, T, num_atoms)
			
			self.target_coords.append(coords_new.clone())
			if self.target_num_atoms is None:
				self.target_num_atoms = torch.zeros_like(num_atoms)
			self.target_num_atoms = self.target_num_atoms + num_atoms
			if self.target_mask is None:
				self.target_mask = torch.zeros_like(self.blocks_masks[0])
			self.target_mask = self.target_mask + self.blocks_masks[i]

			T = T + self.patterns.displacement_type[sequence[0][0]]

			if visualize:
				prot = coords_new, *(prot[1:])
				prot_ca = ProteinStructure(*prot).select_CA()
				atoms_plot = prot_ca.plot_coords(axis=ax)
		
		if visualize:
			plt.savefig("example.png")

		self.target_coords = torch.cat(self.target_coords, dim=1)


	def optimize(self, angles, sequence, visualize=False):
		a2c = Angles2Coords()
		rmsd = Coords2RMSD()
		angles.requires_grad_()
		optimizer = torch.optim.Adam([angles], lr = 0.05)
		
		if visualize:
			fig = plt.figure(figsize=plt.figaspect(0.5))
			camera = Camera(fig)
			ax_prot = fig.add_subplot(1, 2, 1, projection='3d')
			ax_loss = fig.add_subplot(1, 2, 2)
			loss = []

		min_loss = 1000
		min_angles = None
		for epoch in range(300):
			optimizer.zero_grad()
			prot = a2c(angles, sequence)
			coords_src, num_atoms_src = prot[0], prot[-1]
			
			mask = self.target_mask.view(1, self.target_mask.size(1)*self.target_mask.size(2))
			sel_coords_src = coords_src.masked_select(mask).unsqueeze(dim=0)
			L = rmsd(sel_coords_src, self.target_coords, self.target_num_atoms).mean()
			L.backward()

			
			if L.item()<min_loss:
				min_angles = angles.detach().clone()
				min_loss = L.item()

			with torch.no_grad():
				angles.grad.masked_fill_(self.grad_mask, 0.0)

			optimizer.step()
			
			if visualize:
				loss.append(L.item())
				prot_to_plot = prot[0].detach(), *(prot[1:])
				prot_ca = ProteinStructure(*prot_to_plot).select_CA()
				atoms_plot = prot_ca.plot_coords(axis=ax_prot)
				ax_loss.plot(loss)
				camera.snap()
		
		if visualize:
			animation = camera.animate()
			animation.save('anim.gif')
		
		return min_angles, min_loss
	


if __name__=='__main__':
	from Bio.PDB import Polypeptide as pol
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-name', default='train', help='Dataset name', type=str)
	parser.add_argument('-size', default=100, help='Dataset size', type=int)
	args = parser.parse_args()

	import matplotlib 
	import matplotlib.pylab as plt
	import mpl_toolkits.mplot3d.axes3d as p3

	a2c = Angles2Coords()
	ptrn = Patterns()
	smpl = Sampler(ptrn, min_num_blocks=1, max_num_blocks=4, block_min_length=5, block_max_length=10)
	# angles, sequence = smpl.generate()
	# smpl.position(visualize=True)
	# min_angles, min_rmsd = smpl.optimize(angles, sequence, visualize=True)
	# sys.exit()
	
	fig = plt.figure(figsize=plt.figaspect(0.3))
	with open(f'{args.name}/list.dat', 'wt') as fout:
		for i in tqdm(range(args.size)):
			angles, sequence = smpl.generate()
			smpl.position(visualize=False)
			min_angles, min_rmsd = smpl.optimize(angles, sequence, visualize=False)
			
			prot = a2c(min_angles, sequence)
			writePDB(f'{args.name}/{i}.pdb', *prot)
			fout.write(f'{i}.pdb\n')

			if i < 3:
				ax = fig.add_subplot(1, 3, i+1, projection='3d')
				prot_ca = ProteinStructure(*prot).select_CA()
				atoms_plot = prot_ca.plot_coords(axis=ax)
	
	plt.savefig("dataset.png")

	
	

	
			