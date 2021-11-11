import os
import sys

import numpy as np
import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Angles2Coords, writePDB, CoordsTranslate, CoordsRotate, Coords2Center
from TorchProteinLibrary.Utils import ProteinStructure
from TorchProteinLibrary.FullAtomModel import Coords2Angles
from TorchProteinLibrary.RMSD import Coords2RMSD
from Bio.PDB import Polypeptide as pol
from TorchProteinLibrary.Volume import RzRxRz

from matplotlib import pyplot as plt
from celluloid import Camera
from tqdm import tqdm

import random
import argparse

from utils import getSeqNumAtoms, getResNumAtoms, writeMSA

class SeqPatterns:
	def __init__(self):
		self.seq_fill = {
			"alpha": 'A',
			"beta": 'R',
			"displ_1": 'N',
			"displ_2": 'Q'
		}
		self.msa_idx = {
			"alpha": 1,
			"beta": 2,
			"displ_1": 6,
			"displ_2": 7
		}

		self.rand_aa = [aa for aa in list(pol.aa1) if not (aa in self.seq_fill.values())]

	def get_pattern_types(self):
		return [name for name in self.seq_fill.keys() if name.find('displ')==-1]
	
	def get_displacement_types(self):
		return [name for name in self.seq_fill.keys() if name.find('displ')!=-1]

class StructPatterns:
	def __init__(self):
		p2c = PDB2CoordsUnordered()
		fragment = p2c(["protein_samples/alpha.pdb"])
		alpha, length = Coords2Angles(*fragment)
						
		fragment = p2c(["protein_samples/beta_sheet1.pdb"])
		beta, length = Coords2Angles(*fragment)

		self.angles_fill = {
			"alpha": alpha[:,:,1:4],
			"beta": beta[:,:,2:5]
		}
		self.displacement = {
			'displ_1': torch.tensor([[5, 5, 0]], dtype=torch.float32),
			'displ_2': torch.tensor([[5, -5, 0]], dtype=torch.float32)
		}
		self.rotation = {
			'alpha': torch.tensor([[1.6, 0.3, 0]], dtype=torch.float32),
			'beta': torch.tensor([[2.0, -1.0, 0.8]], dtype=torch.float32)
		}

class Sequence:
	def __init__(self, patterns):
		self.blocks = []
		self.patterns = patterns

	@classmethod
	def generate_sequence(cls,  patterns,
								min_num_blocks=10, max_num_blocks=20,
								block_min_length=5, block_max_length=15):		
		
		seq = cls(patterns)
		block_types = ['pattern', 'rand']
		num_blocks = random.randint(min_num_blocks, max_num_blocks)

		for i in range(num_blocks):
			block_length = random.randint(block_min_length, block_max_length)
			block_type = block_types[i%2]
			displ_type = random.choice(patterns.get_displacement_types())
			if block_type == 'pattern':
				pattern = random.choice(patterns.get_pattern_types())
				seq.add_pattern(displ_type, 1)
				seq.add_pattern(pattern, block_length)
				
			elif block_type == 'rand':
				seq.add_pattern(block_type, block_length)
		
		return seq
		
	def add_pattern(self, name, length=10):
		self.blocks.append((name, length))

	def get_seq_len(self, max_idx=None):
		if max_idx is None:
			return sum([length for name, length in self.blocks])
		else:
			return sum([length for ind, (name, length) in enumerate(self.blocks) if ind<max_idx])

	def get_block_displ(self, idx):
		d_name, d_length = self.blocks[idx-1]
		if d_name.find('displ')!=-1 and d_length==1:
			return d_name
		else:
			return None
	
	def get_block_span(self, idx):
		name, length = self.blocks[idx]
		seq_start = self.get_seq_len(max_idx=idx)
		seq_end = seq_start + length
		if name != 'rand':
			msa_idx = self.patterns.msa_idx[name]
		else:
			msa_idx = None
		return seq_start, seq_end, msa_idx

	def get_sequence(self):
		seq = ''
		for name, length in self.blocks:
			if name != 'rand':
				seq += ''.join([self.patterns.seq_fill[name] for i in range(length)])
			else:
				seq += ''.join([random.choice(self.patterns.rand_aa) for i in range(length)])
		return seq

	def get_msa(self, num_msa):
		msa = ['' for i in range(num_msa)]
		for name, length in self.blocks:
			for j in range(num_msa):
				if name != 'rand' and j == self.patterns.msa_idx[name]:
					msa[j] += ''.join([self.patterns.seq_fill[name] for i in range(length)])
				elif j==0:
					msa[j] += ''.join([random.choice(self.patterns.rand_aa) for i in range(length)])
				else:
					msa[j] += ''.join([random.choice(self.patterns.rand_aa+['-']) for i in range(length)])
		return msa

def plot_msa(ax, msa, seq):
	colors = np.zeros((len(msa), seq.get_seq_len()))
	for idx, (name, length) in enumerate(seq.blocks):
		seq_start, seq_end, msa_idx = seq.get_block_span(idx)
		if not(msa_idx is None):
			colors[msa_idx, seq_start:seq_end] = idx+1

	im = ax.imshow(colors)
	ax.set_xticks(np.arange(len(msa[0])))
	ax.set_yticks([])
	ax.set_xticklabels(np.arange(len(msa[0])))
	ax.set_yticklabels([])
	ax.set_title("MSA")

	for i in range(len(msa)):
		for j in range(len(msa[0])):
			text = ax.text(j, i, msa[i][j], ha="center", va="center", color="w")
	

class Structure:
	def __init__(self, seq: Sequence, patterns: StructPatterns):
		self.seq = seq
		self.patterns = patterns
		self.a2c = Angles2Coords()
		self.angles = torch.zeros(1, 8, seq.get_seq_len(), dtype=torch.float, device='cpu')
		for idx, (name, length) in enumerate(seq.blocks):
			seq_start, seq_end, msa_idx = seq.get_block_span(idx)
			if name == 'rand' or (name.find('displ') != -1):
				self.angles[:,:,seq_start:seq_end].normal_()
			else:
				pattern = patterns.angles_fill[name]
				num_repeats = int(length / pattern.size(-1))
				self.angles[:,:,seq_start:seq_end].copy_(pattern.repeat(1, 1, num_repeats+1)[:,:,:length])

	def position(self, sequence, visualize=False):
		rzrxrz = RzRxRz()
		translate = CoordsTranslate()
		rotate = CoordsRotate()
		center = Coords2Center()
		
		if visualize:
			fig = plt.figure(figsize=plt.figaspect(1))
			ax = fig.add_subplot(1, 1, 1, projection='3d')

		T = torch.tensor([[0, 0, 0]], dtype=torch.float32)
		
		self.target_coords = []
		self.target_num_atoms = torch.zeros(1, dtype=torch.int, device='cpu')
		self.target_mask = torch.zeros(1, getSeqNumAtoms(sequence), 3, dtype=torch.bool, device='cpu')
		self.grad_mask = torch.zeros(1, 8, len(sequence), dtype=torch.bool, device='cpu')
		self.secondary = []
		
		num_frags = 0
		for idx, (name, length) in enumerate(self.seq.blocks):
			if name == 'rand' or name.find('displ')!=-1: continue

			seq_start, seq_end, msa_idx = self.seq.get_block_span(idx)
			prot = self.a2c(self.angles[:,:,seq_start:seq_end], [sequence[seq_start:seq_end]])
			coords, num_atoms = prot[0], prot[-1]
			cent = center(coords, num_atoms)

			coords_cent = translate(coords, -cent, num_atoms)
			R = rzrxrz(self.patterns.rotation[name]).contiguous()
			coords_rot = rotate(coords_cent, R, num_atoms)
			if num_frags%2 == 1:
				R = rzrxrz(torch.tensor([[0, np.pi, 0]], dtype=torch.float32)).contiguous()
				coords_rot = rotate(coords_rot, R, num_atoms)
			
			coords_new = translate(coords_rot, T, num_atoms)
			self.target_coords.append(coords_new.clone())
			self.target_num_atoms = self.target_num_atoms + num_atoms
			atom_start = getSeqNumAtoms(sequence[:seq_start])
			atom_end = atom_start + getSeqNumAtoms(sequence[seq_start:seq_end])
			self.target_mask[0, atom_start:atom_end, :] = True
			self.grad_mask[0,:, seq_start:seq_end] = True

			secondary = torch.zeros(1, len(sequence)*3, 3, dtype=torch.bool, device='cpu')
			secondary[0, seq_start*3:seq_end*3, :] = True
			self.secondary.append(secondary.clone())
			
			T = T + self.patterns.displacement[self.seq.get_block_displ(idx)]

			num_frags += 1

			if visualize:
				prot = coords_new, *(prot[1:])
				prot_ca = ProteinStructure(*prot).select_CA()
				atoms_plot = prot_ca.plot_coords(axis=ax)

		if visualize:
			plt.show()

		self.target_coords = torch.cat(self.target_coords, dim=1)
		self.secondary = torch.cat(self.secondary, dim=0)

	def optimize(self, sequence, visualize=False):
		rmsd = Coords2RMSD()
		self.angles.requires_grad_()
		optimizer = torch.optim.Adam([self.angles], lr = 0.05)
		
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
			prot = self.a2c(self.angles, [sequence])
			coords_src, num_atoms_src = prot[0], prot[-1]
			
			mask = self.target_mask.view(1, self.target_mask.size(1)*self.target_mask.size(2))
			sel_coords_src = coords_src.masked_select(mask).unsqueeze(dim=0)
			L = rmsd(sel_coords_src, self.target_coords, self.target_num_atoms).mean()
			L.backward()
			
			if L.item()<min_loss:
				min_angles = self.angles.detach().clone()
				min_loss = L.item()

			with torch.no_grad():
				self.angles.grad.masked_fill_(self.grad_mask, 0.0)

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
			animation.save("struct.mp4")
		
		self.angles.requires_grad_(False)
		self.angles.copy_(min_angles)
		return min_loss

	def get_contacts(self, sequence, cutoff=8):
		prot = self.a2c(self.angles, [sequence])
		prot_ca = ProteinStructure(*prot).select_CA().get()
		coords = prot_ca[0].view(len(sequence), 3)
		dist = coords.unsqueeze(dim=-2) - coords.unsqueeze(dim=-3)
		dist = torch.sqrt(torch.sum(dist*dist, dim=-1))

		diag = torch.arange(0,dist.size(0), dtype=torch.long)
		dist[diag, diag] = cutoff + 1.0
		return dist < cutoff


class MSA:
	def __init__(self, msa: torch.Tensor, contacts: torch.Tensor, mask: torch.Tensor):
		self.msa = msa
		self.contacts = contacts
		self.mask = mask

	@classmethod
	def generate(cls, seq: Sequence, contacts: torch.Tensor, num_msa: int=10):
		msa_init = seq.get_msa(num_msa)

		mask = torch.ones(len(msa_init), len(msa_init[0]), dtype=torch.bool, device='cpu')
		for idx, (name, length) in enumerate(seq.blocks):
			seq_beg, seq_end, msa_idx = seq.get_block_span(idx)
			mask[msa_idx, seq_beg:seq_end] = False

		msa = torch.zeros(len(msa_init), len(msa_init[0]), dtype=torch.long, device='cpu')
		for i, sequence in enumerate(msa_init):
			for j, aa in enumerate(sequence):
				if msa_init[i][j] == '-': 
					msa[i, j] = len(pol.aa1)
					mask[i, j] = False
					continue
				msa[i,j] = pol.d1_to_index[aa]
		
		return cls(msa, contacts, mask)

	def get_msa(self):
		msa = []
		for i in range(self.msa.size(0)):
			seq = ''
			for j in range(self.msa.size(1)):
				if self.msa[i, j] == len(pol.aa1):
					seq += '-'
					continue
				seq += pol.index_to_one(self.msa[i,j].item())
			msa.append(seq)
		return msa
	
	def E(self, msa, de=1.0):
		eq = (msa.unsqueeze(dim=-1) == msa.unsqueeze(dim=-2)).to(torch.float)
		E = torch.sum(self.contacts.unsqueeze(dim=0).to(torch.float) * ( 1.0 - 2.0 * de * eq.to(torch.float)), dim=[-2,-1])
		return E.mean()

	def step(self, max_try=10):
		def select_ij():
			while(True):
				i = random.randint(0, self.msa.size(0)-1)
				j = random.randint(0, self.msa.size(1)-1)
				if self.mask[i,j]:
					return i, j
		
		for try_idx in range(max_try):
			i, j = select_ij()
			indexes = torch.arange(0, self.contacts.size(0), dtype=torch.long)
			eq = (self.msa[i,:].unsqueeze(dim=-1) != self.msa[i,:].unsqueeze(dim=-2))
			pairs = indexes.masked_select(self.contacts[j,:]*eq[j,:]*self.mask[i,:])
			if pairs is None or pairs.size(0)==0:
				continue
			k = pairs[random.randint(0, pairs.size(0)-1)]
			new_msa = self.msa.clone()
			new_msa[i,j] = self.msa[i,k].item()
			return new_msa
		
	def MCMC(self, num_steps, acc_threshold=0.1, T=1.0, visualize=False, seq=None):
		if visualize:
			fig = plt.figure(figsize=plt.figaspect(0.3))
			camera = Camera(fig)
			gs = fig.add_gridspec(2, 2)
			ax_msa = fig.add_subplot(gs[0,:])
			ax_e = fig.add_subplot(gs[1,0])
			ax_acc = fig.add_subplot(gs[1,1])
		
		energy = []
		acc = []
		acc_rate = []
		acc_t = 50
		for step in range(num_steps):
			E_init = self.E(self.msa)
			energy.append(E_init.item())
			new_msa = self.step()
			if not(new_msa is None):
				E_new = self.E(new_msa)
				if torch.exp((E_new - E_init)/T) < (1.0 - torch.rand(1)):
					self.msa = new_msa.clone()
					acc.append(1)
				else:
					acc.append(0)
			else:
				acc.append(0)

			acc_rate.append(np.mean(acc[max(0, len(acc)-acc_t):]))
			
			if acc_rate[-1] < acc_threshold and step>acc_t:
				return True

			if visualize and step%10==0:
				plot_msa(ax_msa, self.get_msa(), seq)
				ax_e.plot(energy)
				ax_e.set_xlabel('Step')
				ax_e.set_ylabel('Energy')
				
				ax_acc.plot(acc_rate)
				ax_acc.set_xlabel('Step')
				ax_acc.set_ylabel('Acc rate')
				camera.snap()
		
		if visualize:
			animation = camera.animate()
			animation.save("mcmc.mp4")
		
		return False

def contacts_example():
	seq = Sequence.generate_sequence(SeqPatterns(), 
									min_num_blocks=5, max_num_blocks=10,
									block_min_length=5, block_max_length=15)

	struct = Structure(seq, StructPatterns())
	sequence = seq.get_sequence()
	print(sequence, getSeqNumAtoms(sequence))
	cont_init = struct.get_contacts(sequence)
	prot_init = Angles2Coords()(struct.angles, [sequence])

	struct.position(sequence, visualize=False)
	print(struct.angles[0,0,:])
	print(struct.target_mask)
	struct.optimize(sequence, visualize=False)

	cont_opt = struct.get_contacts(sequence)
	prot_opt = Angles2Coords()(struct.angles, [sequence])

	fig = plt.figure(figsize=plt.figaspect(0.3))
	ax_prot = fig.add_subplot(2, 2, 1, projection='3d')
	prot_ca = ProteinStructure(*prot_init).select_CA()
	atoms_plot = prot_ca.plot_coords(axis=ax_prot)
	plt.subplot(2, 2, 2)
	plt.imshow(cont_init)	

	ax_prot = fig.add_subplot(2, 2, 3, projection='3d')
	prot_ca = ProteinStructure(*prot_opt).select_CA()
	atoms_plot = prot_ca.plot_coords(axis=ax_prot)
	plt.subplot(2, 2, 4)
	plt.imshow(cont_opt)
	plt.tight_layout()
	plt.show()			

def mcmc_example():
	seq = Sequence.generate_sequence(SeqPatterns(), 
									min_num_blocks=5, max_num_blocks=10,
									block_min_length=5, block_max_length=15)
	sequence = seq.get_sequence()
	struct = Structure(seq, StructPatterns())
	struct.position(sequence, visualize=False)
	struct.optimize(sequence, visualize=False)

	cont_opt = struct.get_contacts(sequence)
	msa = MSA.generate(seq, cont_opt, 10)
	
	msa_init = msa.get_msa()
	msa.MCMC(5000, visualize=True, seq=seq)
	msa_final = msa.get_msa()



if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-name', default='train', help='Dataset name', type=str)
	parser.add_argument('-size', default=100, help='Dataset size', type=int)
	parser.add_argument('-mcmc', default=True, help='MCMC optimize', type=bool)
	parser.add_argument('-secondary', default=False, help='Include secondary structure regions', type=bool)
	args = parser.parse_args()
	
	contacts_example()
	mcmc_example()
	sys.exit()

	with open(f'{args.name}/list.dat', 'wt') as fout:
		i = 0
		with tqdm(total=args.size) as pbar:
			while i < args.size:
				seq = Sequence.generate_sequence(SeqPatterns(), 
										min_num_blocks=1, max_num_blocks=8, 
										block_min_length=5, block_max_length=10)
				sequence = seq.get_sequence()
				
				struct = Structure(seq, StructPatterns())
				struct.position(sequence)
				struct.optimize(sequence)
				
				cont_opt = struct.get_contacts(sequence, cutoff=8)
				msa_gen = MSA.generate(seq, cont_opt, 10)
				if args.mcmc:
					converged = msa_gen.MCMC(5000)
					if not converged:
						continue
					else:
						i += 1
						pbar.update(1)
				else:
					i += 1
					pbar.update(1)
				msa = msa_gen.get_msa()

			
				prot = struct.a2c(struct.angles, [msa[0]])
				writePDB(f'{args.name}/{i}.pdb', *prot)
				writeMSA(f'{args.name}/{i}.msa', msa)
				if args.secondary:
					with open(f'{args.name}/{i}.th', 'wb') as fsec:
						torch.save(struct.secondary, fsec)
				fout.write(f'{i}.pdb\n')
	
