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

import numpy as np
from scipy.special import binom, factorial

class SphericalHarmonics(nn.Module):
	ind_radius = 0
	ind_phi = 1
	ind_theta = 2
	def __init__(self, max_degree=3):
		super().__init__()
		self.max_degree = max_degree

		self.B = {}
		
		for l in range(0, max_degree+1):
			self.B[l] = torch.zeros(2*l+1, l+1)
			for m_ind, m in enumerate(range(-l, l+1)):
				mp = np.abs(m)
				for k in range(mp, l+1):
					p1 = binom(l,k)*binom((l+k-1)/2, l) * factorial(k)/factorial(k-mp) *(2**l)
					p2 = np.sqrt( (factorial(l-mp)*(2*l+1))/(factorial(l+mp)*2.0*np.pi) )
					if m==0:
						p2 /= np.sqrt(2)
					self.B[l][m_ind, k] = ((-1)**m)*p1*p2
					# self.B[l][m_ind, k] = p1*p2
			# print(self.B[l])

	def forward(self, d):
		cost = torch.cos(d[:,self.ind_theta])
		sint = torch.sin(d[:,self.ind_theta])
		
		y_lt0 = []
		y_gt0 = []
		for m in range(1,self.max_degree+1):
			y_lt0.append(torch.sin(m*d[:,self.ind_phi]).unsqueeze(dim=1))
			y_gt0.append(torch.cos(m*d[:,self.ind_phi]).unsqueeze(dim=1))

		y_lt0 = torch.cat(y_lt0, dim=1).flip(dims=[1])
		y_gt0 = torch.cat(y_gt0, dim=1)
		ym = torch.cat([y_lt0, torch.ones_like(cost).unsqueeze(dim=1), y_gt0], dim=1).unsqueeze(dim=-1).repeat(1,1,self.max_degree+1)
		# print('ym', ym)

		costkm = []
		for m in range(-self.max_degree, self.max_degree+1):
			if self.max_degree>abs(m):
				costk = torch.cat([torch.ones_like(cost).unsqueeze(dim=-1), cost.unsqueeze(dim=-1).repeat(1, self.max_degree - abs(m))], dim=-1)
			else: #Preventing 0-chunk gradient propagation
				costk = torch.ones_like(cost).unsqueeze(dim=-1)
			costk = torch.cumprod(costk, dim=-1)
			if abs(m)>0:
				a = torch.zeros_like(cost).unsqueeze(dim=-1).repeat(1, abs(m))
				costk = torch.cat([a, costk], dim=-1)
			costkm.append(costk.unsqueeze(dim=1))
		costkm = torch.cat(costkm, dim=1)
		# print('costkm', costkm)
				
		sint_mg0 = torch.sqrt(1-cost*cost).unsqueeze(dim=1).repeat(1, self.max_degree)
		sint_mg0 = torch.cumprod(sint_mg0, dim=1)
		sint_ml0 = sint_mg0.flip(dims=[1])
		sintm = torch.cat([sint_ml0, torch.ones_like(sint).unsqueeze(dim=1), sint_mg0], dim=1).unsqueeze(dim=-1).repeat(1,1,self.max_degree+1)
		# print('sintm', sintm)

		A = ym * costkm * sintm
		# print('A', A)
		Y = {}
		for l in range(self.max_degree+1):
			Aprime = A[:, int((2*self.max_degree+1)/2)-l:int((2*self.max_degree+1)/2)+l+1, :l+1]
			if self.B[l].device != Aprime.device:
				self.B[l] = self.B[l].to(Aprime.device)
			Y[l] = (self.B[l] * Aprime).sum(dim=2)
			# print(self.B[l])
		
		return Y
				


if __name__=='__main__':
	batch_size = 100
	max_degree = 6
	sph = SphericalHarmonics(max_degree=max_degree)
	r_sp = torch.randn(batch_size, 3).requires_grad_()
	r_sp_ = torch.zeros_like(r_sp).copy_(r_sp)
	r_sp_[:,sph.ind_theta] = np.pi - r_sp[:,sph.ind_theta]
	Y = utils_steerable.precompute_sh(r_sp_, max_degree)
	Y_my = sph(r_sp)
	for l in Y.keys():
		print('Err=', torch.sum(torch.abs(Y[l]-Y_my[l])).item())