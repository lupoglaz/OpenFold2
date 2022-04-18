import torch
from torch import nn

from typing import List, Mapping, Tuple
from alphafold.Data.pipeline import FeatureDict
import ml_collections
import numpy as np
import copy

NUM_RES = "num residues placeholder"
NUM_SEQ = "length msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

ATOM_TYPE_NUM = 37

from alphafold.Model import data_transforms as transf
from alphafold.Model import protein


class AlphaFoldMultimerFeatures(nn.Module):
	def __init__(self, config, device:torch.device=None, is_training:bool=False, dtype=torch.float32):
		super().__init__()
		self.config = config
		self.is_training = is_training
		self.device = device
		self.dtype = dtype

	def np_to_tensor_dict(self, np_example: FeatureDict) -> FeatureDict:
		#Features dictionary to tensor
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/proteins_dataset.py#L145
		
		tensor_dict = {}
		for k, v in np_example.items():
			if v.dtype == np.object_:
				tensor_dict[k] = torch.from_numpy(np.fromstring(v))
			else:
				tensor_dict[k] = torch.from_numpy(v)
				if not(self.device is None):
					tensor_dict[k] = tensor_dict[k].to(device=self.device)

		return tensor_dict

	def forward(self, raw_features: FeatureDict, random_seed: int=0) -> FeatureDict:
		tensor_dict = self.np_to_tensor_dict(raw_features)
		