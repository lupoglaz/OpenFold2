import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import torch.nn as nn
import dgl
import random
import numpy as np

from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import get_basis_and_r, get_basis
from equivariant_attention.from_se3cnn import utils_steerable

class Basis(nn.Module):
	def __init__(self, max_degree=3):
		super().__init__()
		self.max_degree = max_degree
		
		
		self.Q_J = {}
		for d_in in range(max_degree+1):
			for d_out in range(max_degree+1):
				for J in range(abs(d_in-d_out), d_in+d_out+1):
					# Get spherical harmonic projection matrices
					self.Q_J[(d_in, d_out, J)] = utils_steerable._basis_transformation_Q_J(J, d_in, d_out).float().T

	def forward(self, Y):
		device = Y[0].device
		basis = {}
		for d_in in range(self.max_degree+1):
			for d_out in range(self.max_degree+1):
				K_Js = []
				for J in range(abs(d_in-d_out), d_in+d_out+1):
					
					if self.Q_J[(d_in, d_out, J)].device != device:
						self.Q_J[(d_in, d_out, J)] = self.Q_J[(d_in, d_out, J)].to(device)

					# Create kernel from spherical harmonics
					K_J = torch.matmul(Y[J], self.Q_J[(d_in, d_out, J)])
					K_Js.append(K_J)

				# Reshape so can take linear combinations with a dot product
				size = (-1, 1, 2*d_out+1, 1, 2*d_in+1, 2*min(d_in,d_out)+1)
				basis[f'{d_in},{d_out}'] = torch.stack(K_Js, -1).view(*size)

		return basis

if __name__=='__main__':
	batch_size = 100
	max_degree = 3
	r_sp = torch.randn(batch_size, 3).requires_grad_()
	r_sp_ = torch.zeros_like(r_sp).copy_(r_sp)
	r_sp_[:,2] = np.pi - r_sp[:,2]
	Y = utils_steerable.precompute_sh(r_sp_, 2*max_degree)
	for l in Y.keys():
		print('l:', Y[l].size())
		Y[l].requires_grad_()

	b = Basis(max_degree)
	
	my_basis = b(Y)
	basis = get_basis(Y,max_degree)
	print(basis.keys())
	for key in basis.keys():
		print('Err=', torch.sum(torch.abs(basis[key] - my_basis[key])).item())

	print(my_basis['0,0'].squeeze())