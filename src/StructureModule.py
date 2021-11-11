import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
from math import cos, sin
import torch
import torch.nn as nn
import dgl
import random

from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import get_basis_and_r, get_basis
from equivariant_attention.from_se3cnn import utils_steerable

import numpy as np
from scipy.special import binom, factorial

class RotationMatrix(nn.Module):
	'''
	https://en.wikipedia.org/wiki/Cayley_transform
	input: 3-vector field per rotation matrix
	'''
	def __init__(self):
		super().__init__()
		self.I = torch.zeros(1,3,3)
		for i in range(3): self.I[0,i,i] = 1.0
	
	def forward(self, input):
		batch_size = input.size(0)
		assert input.size(1) == (3)
		if self.I.device != input.device:
			self.I = self.I.to(input.device)
		if self.I.dtype != input.dtype:
			self.I = self.I.to(input.dtype)
		
		I = self.I.repeat(batch_size, 1, 1)
		
		#trace is a scalar field
		# I = self.delta*input[:,0]/3.0
				
		#anti-symmetric part
		zero = torch.zeros_like(input[:,0])
		A1 = torch.stack([zero, -input[:,0], -input[:,1]], dim=1)
		A2 = torch.stack([input[:,0], zero, -input[:,2]], dim=1)
		A3 = torch.stack([input[:,1], input[:,2], zero], dim=1)
		A = torch.stack([A1, A2, A3], dim=2)
		
		# Cayley's transform
		return (I - A) @ torch.inverse(I + A)

		# #symmetric part
		# S1 = torch.stack([input[:,4], input[:,6], input[:,7]], dim=1)
		# S2 = torch.stack([input[:,6], input[:,5], input[:,8]], dim=1)
		# S3 = torch.stack([input[:,7], input[:,8], -input[:,4]-input[:,5]], dim=1)
		# S = 0.5*torch.stack([S1, S2, S3], dim=2)
		
		# return I + A + S



def get_B(phi, psi, R):
	return torch.tensor([[cos(psi), sin(phi)*sin(psi), cos(phi)*sin(psi), R*cos(psi)],
						[0, cos(phi), -sin(phi), 0],
						[-sin(psi), sin(phi)*cos(psi), cos(phi)*cos(psi), -R*sin(psi)],
						[0, 0, 0, 1]])

class StructureModule(nn.Module):
	'''
	Input: Two vector fields one rotation, one translation
	'''
	def __init__(self):
		super().__init__()
		self.coords = torch.tensor([[0.0, 0.0, 0.0, 1.0], #C-alpha
									[0.0, 0.0, 0.0, 1.0], #C
									[0.0, 0.0, 0.0, 1.0]  #N
									])
		R_CA_C = 1.525
		R_C_N = 1.330
		R_N_CA = 1.460

		CA_C_N = (math.pi - 2.1186)
		C_N_CA = (math.pi - 1.9391)
		N_CA_C = (math.pi - 2.061)

		B_CA_C = get_B(0.0, N_CA_C, R_CA_C)
		B_C_N = get_B(0.0, CA_C_N, R_C_N)
		self.coords[1,:] = B_CA_C @ self.coords[0,:]
		self.coords[2,:] = B_C_N @ self.coords[1,:]
		self.coords = self.coords[:,:3].unsqueeze(dim=0).transpose(1,2)
		
		self.rotMat = RotationMatrix()
	
	def forward(self, input):
		batch_size = input.size(0)
		assert input.size(1) == (2) and input.size(2) == (3)
		if self.coords.device != input.device:
			self.coords = self.coords.to(input.device)
		if self.coords.dtype != input.dtype:
			self.coords = self.coords.to(input.dtype)
		
		coords = self.coords.repeat(batch_size, 1, 1)
		R = self.rotMat(input[:,1,:])
		x = R @ self.coords

		dx = input[:,0,:].unsqueeze(dim=-1).repeat(1,1,3)
		
		return (x + dx).transpose(1,2)

def test_rotations():
	input = torch.randn(1,3)
	rm = RotationMatrix()
	R = rm(input)
	print(torch.det(R))
	print(R @ (R.transpose(1,2)))

if __name__=='__main__':
	struct = StructureModule()
	input = torch.randn(1, 2, 3)
	x = struct(input)
	print(x)
	print(x[0,0,:])
