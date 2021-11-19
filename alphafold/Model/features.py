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
			#"template_domain_names": (torch.uint8, [NUM_TEMPLATES]),
			#"template_sum_probs": (torch.float32, [NUM_TEMPLATES, 1]),
			#"template_aatype": (torch.float32, [NUM_TEMPLATES, NUM_RES, 22]),
			#"template_all_atom_positions": (torch.float32, [NUM_TEMPLATES, NUM_RES, ATOM_TYPE_NUM, 3]),
			#"template_all_atom_masks": (torch.float32, [NUM_TEMPLATES, NUM_RES, ATOM_TYPE_NUM, 1]),
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

	def np_to_tensor_dict(self, np_example: FeatureDict, features: ml_collections.ConfigDict) -> FeatureDict:
		#Features dictionary to tensor
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/proteins_dataset.py#L145
		
		required_features = ["aatype", "sequence", "seq_length"]
		feature_names = list(set(features) | set(required_features))
		features_metadata = {name: self.protein_features[name] for name in feature_names}

		tensor_dict = {}
		for k, v in np_example.items():
			if v.dtype == np.object_:
				tensor_dict[k] = torch.from_numpy(np.fromstring(v))
			else:
				tensor_dict[k] = torch.from_numpy(v)

		#Reshaping tensors
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/proteins_dataset.py#L57
		

		return tensor_dict

	def forward(self, raw_features: FeatureDict, random_seed: int=0) -> FeatureDict:
		#Processing features
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/model.py#L88
		#features.np_example_to_features
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/features.py#L76
		
		raw_features = dict(raw_features)
		num_res = int(raw_features['seq_length'][0])
		cfg, feature_names = self.make_data_config(num_res)
		mode_cfg = cfg['eval']
		
		if 'deletion_matrix_int' in raw_features:
			raw_features['deletion_matrix'] = (raw_features.pop('deletion_matrix_int').astype(np.float32))

		tensor_dict = self.np_to_tensor_dict(raw_features, feature_names)

		#Process tensors to get model input
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/input_pipeline.py#L125
		
		tensor_dict = transf.cast_to_64bit_ints(tensor_dict)
		tensor_dict = transf.correct_msa_restypes(tensor_dict)
		tensor_dict = transf.squeeze_features(tensor_dict)
		tensor_dict = transf.randomly_replace_msa_with_unknown(tensor_dict, 0.0)
		tensor_dict = transf.make_seq_mask(tensor_dict)
		tensor_dict = transf.make_msa_mask(tensor_dict)
		tensor_dict = transf.make_hhblits_profile(tensor_dict)
		
		if cfg.common.use_templates:
			raise NotImplementedError()

		#Ensembled features
		#https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/tf/input_pipeline.py#L64
		if 'no_recycling_iters' in tensor_dict:
			num_recycling = int(tensor_dict['no_recycling_iters'])
		else:
			num_recycling = cfg.common.num_recycle

		# TODO: Stack recycled stuff!!
		#https://github.com/aqlaboratory/openfold/blob/8d1119dff18506588604949d2cc81997d63cd911/openfold/data/input_pipeline.py#L200
		ensemble = []
		for i in range(num_recycling+1):
			tensor_dict_ens = copy.deepcopy(tensor_dict)
			if "max_distillation_msa_clusters" in mode_cfg:
				tensor_dict_ens = transf.sample_msa_distillation(tensor_dict_ens, mode_cfg.max_distillation_msa_clusters)

			if cfg.common.reduce_msa_clusters_by_max_templates:
				pad_msa_clusters = cfg.eval.max_msa_clusters - cfg.eval.max_templates
			else:
				pad_msa_clusters = cfg.eval.max_msa_clusters
			max_msa_clusters = pad_msa_clusters
			
			msa_seed = None
			if(not cfg.common.resample_msa_in_recycling):
				msa_seed = random_seed
			
			tensor_dict_ens = transf.sample_msa(tensor_dict_ens, max_msa_clusters, keep_extra=True, seed=msa_seed)

			if "masked_msa" in cfg.common:
				tensor_dict_ens = transf.make_masked_msa(tensor_dict_ens, cfg.common.masked_msa, mode_cfg.masked_msa_replace_fraction)
			
			if cfg.common.msa_cluster_features:
				tensor_dict_ens = transf.nearest_neighbor_clusters(tensor_dict_ens)
				tensor_dict_ens = transf.summarize_clusters(tensor_dict_ens)
			
			if cfg.common.max_extra_msa:
				tensor_dict_ens = transf.crop_extra_msa(tensor_dict_ens, cfg.common.max_extra_msa)
			else:
				tensor_dict_ens = transf.delete_extra_msa(tensor_dict_ens)

			tensor_dict_ens = transf.make_msa_feat(tensor_dict_ens)
			
			crop_feats = dict(mode_cfg.feat)
			if mode_cfg.fixed_size:
				tensor_dict_ens = transf.select_feat(tensor_dict_ens, crop_feats)
				tensor_dict_ens = transf.random_crop_to_size(
								tensor_dict_ens, mode_cfg.crop_size, mode_cfg.max_templates, 
								crop_feats, mode_cfg.subsample_templates, seed=random_seed+1)
				tensor_dict_ens = transf.make_fixed_size(
								tensor_dict_ens, crop_feats, pad_msa_clusters, 
								cfg.common.max_extra_msa, mode_cfg.crop_size, mode_cfg.max_templates)
			else:
				tensor_dict_ens = transf.crop_templates(tensor_dict_ens, mode_cfg.max_templates)

			ensemble.append(tensor_dict_ens)
		
		ensembled_dict = {}
		for feat in ensemble[0].keys():
			ensembled_dict[feat] = torch.stack([dict_i[feat] for dict_i in ensemble], dim=0)
		
		return ensembled_dict
		