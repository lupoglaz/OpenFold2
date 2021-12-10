from numpy import broadcast
import torch
from torch import nn
from typing import Sequence, Tuple
import numpy as np

class InvariantPointAttention(nn.Module):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/folding.py#L37
	"""
	def __init__(self, config, global_config, num_res:int, num_seq:int, dist_epsilon:float=1e-8) -> None:
		super(InvariantPointAttention, self).__init__()
		self.config = config
		self.global_config = global_config
		self._dist_epsilon = dist_epsilon

		self.num_res = num_res
		self.num_head = self.config.num_head
		self.num_scalar_qk = self.config.num_scalar_qk
		self.num_scalar_v = self.config.num_scalar_v
		self.num_point_qk = self.config.num_point_qk
		self.num_point_v = self.config.num_point_v

		scalar_variance = max(self.num_scalar_qk, 1) * 1.0
		point_variance = max(self.num_point_qk, 1) * 9.0/2.0
		num_logit_terms = 3
		scalar_weights = np.sqrt(1.0/num_logit_terms*scalar_variance)
		self.point_weights = np.sqrt(1.0/num_logit_terms*point_variance)
		attention2d_weights = np.sqrt(1.0/num_logit_terms)

		self.q_scalar = nn.Linear(num_res, self.num_head * self.num_scalar_qk)
		self.kv_scalar = nn.Linear(num_res, self.num_head*(self.num_scalar_v + self.num_scalar_qk))
		self.q_point_local = nn.Linear(num_res, self.num_head * 3 * self.num_point_qk)
		self.kv_point_local = nn.Linear(num_res, self.num_head * 3 * (self.num_point_qk + self.num_point_v))
		self.trainable_point_weights = nn.Parameter(torch.ones(self.num_head))
		self.attention_2d = nn.Linear(num_seq, self.num_head)

	def forward(self, inputs_1d:torch.Tensor, inputs_2d:torch.Tensor, mask:torch.Tensor, affine) -> None:
		q_scalar = self.q_scalar(inputs_1d)
		q_scalar = q_scalar.view(self.num_res, self.num_head, self.num_scalar_qk)

		kv_scalar = self.kv_scalar(inputs_1d)
		kv_scalar = kv_scalar.view(self.num_res, self.num_head, self.num_scalar_v + self.num_scalar_qk)
		k_scalar, v_scalar = kv_scalar.split(self.num_scalar_qk, dim=-1)

		q_point_local = self.q_point_local(inputs_1d)
		q_point_local = q_point_local.split(3, dim=-1)
		q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
		q_point = [x.view(self.num_res, self.num_head, self.num_point_qk) for x in q_point_global]

		kv_point_local = self.kv_point_local(inputs_1d)
		kv_point_local = kv_point_local.split(3, dim=-1)
		kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
		kv_point_global = [x.view(self.num_res, self.num_head, self.num_point_qk + self.num_point_v) for x in kv_point_global]
		k_point, v_point = list(zip(*[x.split(self.num_point_qk, dim=-1) for x in kv_point_global]))

		point_weights = self.trainable_point_weights.unsqueeze(dim=1) * self.point_weights

		v_point = [x.transpose(-2, -3) for x in v_point]
		q_point = [x.transpose(-2, -3) for x in q_point]
		k_point = [x.transpose(-2, -3) for x in k_point]
		dist2 = []

