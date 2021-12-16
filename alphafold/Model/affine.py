import functools
import collections
from typing import Tuple

import torch
import numpy as np

Vecs = collections.namedtuple('Vecs', ['x', 'y', 'z'])
Rots = collections.namedtuple('Rots', [	'xx', 'xy', 'xz',
										'yx', 'yy', 'yz',
										'zx', 'zy', 'zz'])
Rigids = collections.namedtuple('Rigids', ['rot', 'trans'])

def quat_to_rot(quaternion):
	q0, q1, q2, q3 = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
	return [[q0*q0+q1*q1-q2*q2-q3*q3, 	2*(q1*q2-q0*q3), 		2*(q0*q2+q1*q3)],
			[2*(q1*q2+q0*q3), 		q0*q0-q1*q1+q2*q2-q3*q3, 	2*(q2*q3-q0*q1)],
			[2*(q1*q3-q0*q2), 			2*(q0*q1+q2*q3), 	q0*q0-q1*q1-q2*q2+q3*q3]]

def apply_rot_to_vec(rot, vec, unstack:bool=False):
	if unstack:
		x, y, z = [vec[:, i] for i in range(3)]
	else:
		x, y, z = vec

	return [rot[0][0]*x + rot[0][1]*y + rot[0][2]*z,
			rot[1][0]*x + rot[1][1]*y + rot[1][2]*z,
			rot[2][0]*x + rot[2][1]*y + rot[2][2]*z]

def apply_inverse_rot_to_vec(rot, vec):
	x, y, z = vec
	return [rot[0][0]*x + rot[1][0]*y + rot[2][0]*z,
			rot[0][1]*x + rot[1][1]*y + rot[2][1]*z,
			rot[0][2]*x + rot[1][2]*y + rot[2][2]*z]


class QuatAffine(object):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/quat_affine.py#L181
	"""
	def __init__(self, 	quaternion:torch.Tensor, translation:torch.Tensor, rotation:torch.Tensor=None, 
						normalize:bool=True, unstack_inputs:bool=False) -> None:
		super().__init__()
		if not(quaternion is None):
			assert quaternion.shape[-1] == 4
			if normalize:
				quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True)
		if unstack_inputs:
			if not(rotation is None):
				rotation = [torch.moveaxis(x, -1, 0) for x in torch.moveaxis(rotation, -2, 0)]
			translation = torch.moveaxis(translation, -1, 0)
		if rotation is None:
			rotation = quat_to_rot(quaternion)

		self.quaternion = quaternion
		self.rotation = [list(row) for row in rotation]
		self.translation = list(translation)
		# print(self.quaternion.dtype, self.rotation[0][0].dtype,self.translation[0].dtype)

		assert all(len(row) == 3 for row in self.rotation)
		assert len(self.translation) == 3

	@classmethod
	def from_tensor(cls, tensor:torch.Tensor, normalize:bool=False):
		return cls(tensor[...,:4], [tensor[...,4], tensor[...,5], tensor[...,6]], normalize=normalize)
	
	def to_tensor(self):
		return torch.cat([self.quaternion]+[x.unsqueeze(dim=-1) for x in self.translation], dim=-1)

	def scale_translation(self, position_scale):
		return QuatAffine(	quaternion=self.quaternion, 
							translation=[x * position_scale for x in self.translation],
							rotation=[[x for x in row] for row in self.rotation], 
							normalize=False)
	
	def apply_rotation_tensor_fn(self, tensor_fn):
		return QuatAffine(	quaternion=tensor_fn(self.quaternion),
							translation=[x for x in self.translation],
							rotation=[[tensor_fn(x) for x in row] for row in self.rotation], 
							normalize=False)
	
	def apply_to_point(self, point, extra_dims=0):
		# r = self.rotation
		# t = self.translation
		r, t = [], []
		for t_i in self.translation:
			t_iu = t_i
			for _ in range(extra_dims):
				t_iu = t_iu.unsqueeze(dim=-1)
			t.append(t_iu)
		for r_i in self.rotation:
			r_vec = []
			for r_ij in r_i:
				r_iju = r_ij
				for _ in range(extra_dims):
					r_iju = r_iju.unsqueeze(dim=-1)
				r_vec.append(r_iju)
			r.append(r_vec)
		r_p = apply_rot_to_vec(r, point)
		return [r_p[0]+t[0], r_p[1]+t[1], r_p[2]+t[2]]

	def invert_point(self, transformed_point, extra_dims=0):
		# r = self.rotation
		# t = self.translation
		# for _ in range(extra_dims):
		# 	r = r.unsqueeze(dim=-1)
		# 	t = t.unsqueeze(dim=-1)
		r, t = [], []
		for t_i in self.translation:
			t_iu = t_i
			for _ in range(extra_dims):
				t_iu = t_iu.unsqueeze(dim=-1)
			t.append(t_iu)
		for r_i in self.rotation:
			r_vec = []
			for r_ij in r_i:
				r_iju = r_ij
				for _ in range(extra_dims):
					r_iju = r_iju.unsqueeze(dim=-1)
				r_vec.append(r_iju)
			r.append(r_vec)
		r_p = [transformed_point[0]-t[0], transformed_point[1]-t[1], transformed_point[2]-t[2]]
		return apply_inverse_rot_to_vec(r, r_p)

	def to_rigids(self) -> Rigids:
		r = self.rotation
		t = self.translation
		return Rigids(	Rots(	r[0][0], r[0][1], r[0][2],
								r[1][0], r[1][1], r[1][2],
								r[2][0], r[2][1], r[2][2]), 
						Vecs(t[0], t[1], t[2]))