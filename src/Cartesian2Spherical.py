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

class Cartesian2Spherical(nn.Module):
	ind_radius = 0
	ind_phi = 1
	ind_theta = 2
	cartesian_x = 0
	cartesian_y = 1
	cartesian_z = 2

	def __init__(self):
		super().__init__()

	def forward(self, r_ct):	
		# get projected radius in xy plane
		r_xy = r_ct[:, self.cartesian_x] ** 2 + r_ct[:, self.cartesian_y] ** 2 + 1E-5
		# get overall radius
		r_sp_rad = torch.sqrt(r_xy + r_ct[:, self.cartesian_z]**2)
		# get second angle
		# version 'elevation angle defined from Z-axis down'
		r_sp_theta = torch.atan2(torch.sqrt(r_xy), r_ct[:, self.cartesian_z])
		# get angle in x-y plane
		r_sp_phi = torch.atan2(r_ct[:, self.cartesian_y], r_ct[:, self.cartesian_x])
		return torch.stack([r_sp_rad, r_sp_phi, r_sp_theta], dim=1)

class Cartesian2SphericalFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, r_ct):
		ind_radius = 0
		ind_phi = 1
		ind_theta = 2
		cartesian_x = 0
		cartesian_y = 1
		cartesian_z = 2
		r_sp = torch.zeros_like(r_ct)
		# get projected radius in xy plane
		r_xy = r_ct[:, cartesian_x] ** 2 + r_ct[:, cartesian_y] ** 2
		# get overall radius
		r_sp[:, ind_radius] = torch.sqrt(r_xy + r_ct[:, cartesian_z]**2)
		# get second angle
		# version 'elevation angle defined from Z-axis down'
		r_sp[:, ind_theta] = torch.atan2(torch.sqrt(r_xy), r_ct[:, cartesian_z])
		# get angle in x-y plane
		r_sp[:, ind_phi] = torch.atan2(r_ct[:, cartesian_y], r_ct[:, cartesian_x])

		ctx.save_for_backward(r_sp)
		return r_sp
	
	@staticmethod
	def backward(ctx, dr_sp):
		ind_radius = 0
		ind_phi = 1
		ind_theta = 2
		cartesian_x = 0
		cartesian_y = 1
		cartesian_z = 2
		r_sp, = ctx.saved_tensors
		dr_ct = torch.zeros_like(dr_sp)
		stheta = torch.sin(r_sp[:, ind_theta])
		ctheta = torch.cos(r_sp[:, ind_theta])
		sphi = torch.sin(r_sp[:, ind_phi])
		cphi = torch.cos(r_sp[:, ind_phi])
		dr_ct[:, cartesian_x] = stheta*cphi*dr_sp[:,ind_radius] + ctheta*cphi*dr_sp[:,ind_theta] - sphi*dr_sp[:,ind_phi]
		dr_ct[:, cartesian_y] = stheta*sphi*dr_sp[:,ind_radius] + ctheta*sphi*dr_sp[:,ind_theta] + cphi*dr_sp[:,ind_phi]
		dr_ct[:, cartesian_z] = ctheta*dr_sp[:,ind_radius] - stheta*dr_sp[:,ind_theta]
		return dr_ct


def basis_change_test():
	print('Testing cartesian to spherical basis')
	max_num_atoms = 10
	batch_size = 3
	num_atoms = torch.zeros(batch_size, dtype=torch.long)
	r = torch.zeros(batch_size, max_num_atoms, 3, dtype=torch.float32)
	for i in range(batch_size):
		num_atoms[i] = random.randint(3, 10)
		r[i, :num_atoms[i].item(), :].copy_(3*torch.rand(num_atoms[i].item(), 3))
		
	graphs = make_neighbour_graph(r, num_atoms)
	graphs.ndata['x'].requires_grad_()
	graphs.edata['d'].requires_grad_()
	

	d_sp = Cartesian2Spherical.apply(graphs.edata['d'])
	L = (d_sp[:,0] + 10*torch.cos(d_sp[:,1]) + torch.sin(d_sp[:,2])).mean()
	L.backward()
	grad_one = torch.zeros_like(graphs.edata['d'].grad).copy_(graphs.edata['d'].grad)

	graphs.edata['d'].grad.zero_()
	d_sp = Cartesian2SphericalModule()(graphs.edata['d'])
	L = (d_sp[:,0] + 10*torch.cos(d_sp[:,1]) + torch.sin(d_sp[:,2])).mean()
	L.backward()
	grad_two = torch.zeros_like(graphs.edata['d'].grad).copy_(graphs.edata['d'].grad)
	# print(torch.abs(grad_one - grad_two)/torch.abs(grad_one+1E-3))
	sys.exit()

	eps = 0.0001
	with torch.no_grad():
		for i in range(graphs.edata['d'].size(0)):
			d = graphs.edata['d'][i,:]
			if torch.sum(d*d).item()<1E-5:
				continue
			for j in range(graphs.edata['d'].size(1)):
				d_cache_p = torch.zeros_like(graphs.edata['d']).copy_(graphs.edata['d'])
				d_cache_p[i,j] += eps
				d_cache_m = torch.zeros_like(graphs.edata['d']).copy_(graphs.edata['d'])
				d_cache_m[i,j] -= eps
				d_sp1p = Cartesian2Spherical.apply(d_cache_p)
				d_sp1m = Cartesian2Spherical.apply(d_cache_m)
				L1p = (d_sp1p[:,0] + 10*torch.cos(d_sp1p[:,1]) + torch.sin(d_sp1p[:,2])).mean()
				L1m = (d_sp1m[:,0] + 10*torch.cos(d_sp1m[:,1]) + torch.sin(d_sp1m[:,2])).mean()
				grad = (L1p - L1m)/(2*eps)
				err = abs(grad.item() - graphs.edata['d'].grad[i,j].item())/abs(grad.item()+eps)
				print(i, j, err, graphs.edata['d'].grad[i,j].item(), grad.item())

if __name__ == '__main__':
    pass
