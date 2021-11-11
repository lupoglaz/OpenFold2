import torch
from torch import nn

from typing import List, Mapping, Tuple
import ml_collections
import numpy as np
import copy

NUM_RES = "num residues placeholder"
NUM_SEQ = "length msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

ATOM_TYPE_NUM = 37


class AlphaFoldFeatures(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.protein_features = {
			#### Static features of a protein sequence ####
			"aatype": (torch.float32, [NUM_RES, 21]),
			"between_segment_residues": (torch.int64, [NUM_RES, 1]),
			"deletion_matrix": (torch.float32, [NUM_SEQ, NUM_RES, 1]),
			"domain_name": (torch.uint8, [1]),
			"msa": (torch.int64, [NUM_SEQ, NUM_RES, 1]),
			"num_alignments": (torch.int64, [NUM_RES, 1]),
			"residue_index": (torch.int64, [NUM_RES, 1]),
			"seq_length": (torch.int64, [NUM_RES, 1]),
			"sequence": (torch.uint8, [1]),
			"all_atom_positions": (torch.float32, [NUM_RES, ATOM_TYPE_NUM, 3]),
			"all_atom_mask": (torch.int64, [NUM_RES, ATOM_TYPE_NUM]),
			"resolution": (torch.float32, [1]),
			"template_domain_names": (torch.uint8, [NUM_TEMPLATES]),
			"template_sum_probs": (torch.float32, [NUM_TEMPLATES, 1]),
			"template_aatype": (torch.float32, [NUM_TEMPLATES, NUM_RES, 22]),
			"template_all_atom_positions": (torch.float32, [NUM_TEMPLATES, NUM_RES, ATOM_TYPE_NUM, 3]),
			"template_all_atom_masks": (torch.float32, [NUM_TEMPLATES, NUM_RES, ATOM_TYPE_NUM, 1]),
		}

	def make_data_config(self, num_res: int) -> Tuple[ml_collections.ConfigDict, List[str]]:
		"""Makes a data config for the input pipeline."""
		cfg = copy.deepcopy(self.config.data)

		feature_names = cfg.common.unsupervised_features
		if cfg.common.use_templates:
			feature_names += cfg.common.template_features

		with cfg.unlocked():
			cfg.eval.crop_size = num_res

		return cfg, feature_names

	def np_to_tensor_dict(self, np_example, features):
		#Features dictionary to tensor
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/proteins_dataset.py#L145
		
		required_features = ["aatype", "sequence", "seq_length"]
		feature_names = list(set(features) | set(required_features))
		features_metadata = {name: self.protein_features[name] for name in feature_names}

		tensor_dict = {}
		for k,v in np_example.items():
			if k == 'domain_name':
				tensor_dict[k] = v
				continue
			if v.dtype == np.object:
				tensor_dict[k] = torch.from_numpy(np.fromstring(v))
			else:
				tensor_dict[k] = torch.from_numpy(v)

			print(k, tensor_dict[k].size())

		#Reshaping tensors
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/proteins_dataset.py#L57
		

		return tensor_dict

	def forward(self, raw_features, random_seed):
		#Processing features
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/model.py#L88
		#features.np_example_to_features
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/features.py#L76

		raw_features = dict(raw_features)
		num_res = int(raw_features['seq_length'][0])
		cfg, feature_names = self.make_data_config(num_res)
		
		if 'deletion_matrix_int' in raw_features:
			raw_features['deletion_matrix'] = (raw_features.pop('deletion_matrix_int').astype(np.float32))

		tensor_dict = self.np_to_tensor_dict(raw_features, feature_names)

		#Process tensors to get model input
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/input_pipeline.py#L125
		processed_batch = input_pipeline.process_tensors_from_config(tensor_dict, cfg)
		