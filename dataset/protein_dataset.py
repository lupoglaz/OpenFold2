import os
import sys

import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Angles2Coords
from TorchProteinLibrary.Utils import ProteinStructure
from TorchProteinLibrary.FullAtomModel import Coords2Angles

class Patterns:
	def __init__(self):
		p2c = PDB2CoordsUnordered()
		fragment = p2c(["protein_samples/alpha.pdb"])
		alpha, length = Coords2Angles(*fragment)
		self.alpha = alpha[0,:,1:4]

		fragment = p2c(["protein_samples/alpha_link.pdb"])
		alpha, length = Coords2Angles(*fragment)
		self.alpha_link = alpha[0,:,:7]
		
		fragment = p2c(["protein_samples/beta_sheet1.pdb"])
		beta, length = Coords2Angles(*fragment)
		self.beta = beta
		self.beta_left = beta[0,:,:8]
		self.beta_link = beta[0,:,8:15]
		self.beta_right = beta[0,:,15:]
		print(length)

	def apply_alpha(self, length):
		num_repeats = int(length / self.alpha.size(-1))
		return self.alpha.repeat(1, num_repeats+1)[:,:length]


if __name__=='__main__':
	ptrn = Patterns()

	a2c = Angles2Coords()
	sequence = [''.join(['A' for i in range(27)])]
	angles = torch.randn(1, 8, len(sequence[0]), dtype=torch.float, device='cpu')
	angles[0,:,:10] = ptrn.apply_alpha(10)
	angles[0,:,10:17] = ptrn.alpha_link
	angles[0,:,17:27] = ptrn.apply_alpha(10)
	
	prot = a2c(angles, sequence)
	prot_ca = ProteinStructure(*prot).select_CA()
	atoms_plot = prot_ca.plot_coords()