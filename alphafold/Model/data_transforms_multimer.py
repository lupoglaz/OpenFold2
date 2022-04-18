import torch
from torch import distributions
import torch.nn.functional as F
from alphafold.Common import residue_constants
from functools import reduce
from operator import add
import numpy as np
from .config import NUM_RES, NUM_EXTRA_SEQ, NUM_TEMPLATES, NUM_MSA_SEQ
from torch.distributions.gumbel import Gumbel

def make_msa_profile(feature_dict):
	assert 'msa_mask' in feature_dict.keys()
	assert 'msa' in feature_dict.keys()
	msa_one_hot = F.one_hot(feature_dict['msa'].long(), 22)
	one_hot_mask = feature_dict['msa_mask'][:, :, None]
	return torch.sum(one_hot_mask * msa_one_hot, dim=0) / (torch.sum(one_hot_mask, dim=0) + 1e-8)

def sample_msa(feature_dict, max_seq):
	assert 'msa_mask' in feature_dict.keys()
	assert 'msa' in feature_dict.keys()
	
	msa_mask = feature_dict['msa_mask']
	logits = (torch.clip(torch.sum(msa_mask, dim=-1), min=0.0, max=1.0) - 1.0)*1e6
	
	if 'cluster_bias_mask' in feature_dict:
		cluster_bias_mask = feature_dict['cluster_bias_mask']
	else:
		zeros = feature_dict['msa'].new_zeros(feature_dict['msa'].shape[0] - 1)
		cluster_bias_mask = F.pad(zeros, (1, 0), mode='constant', value=1.0)

	logits += cluster_bias_mask * 1e6
	index_order = gumbel_argsort_sample_idx(logits)
	sel_idx = index_order[:max_seq]
	extra_idx = index_order[max_seq:]

	for key in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
		if key in feature_dict:
			feature_dict['extra_' + key] = feature_dict[key][extra_idx]
			feature_dict[key] = feature_dict[key][sel_idx]
		else:
			raise Exception(f'{key} not found in features')
	
	return feature_dict
	

def make_masked_msa(feature_dict, config):
	def shaped_categorical(probs, epsilon=1e-10):
		ds = probs.shape
		num_classes = ds[-1]
		distribution = torch.distributions.categorical.Categorical(torch.reshape(probs+epsilon, [-1, num_classes]))
		counts = distribution.sample()
		return counts.reshape(ds[:-1])

	msa = feature_dict["msa"]
	random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=torch.float32, device=msa.device)
	categorical_probs = (config.uniform_prob * random_aa + 
						config.profile_prob * feature_dict["msa_profile"] + 
						config.same_prob * F.one_hot(msa.long(), 22))
	
	pad_shapes = list(reduce(add, [(0,0) for _ in range(len(categorical_probs.shape))]))
	pad_shapes[1] = 1
	
	mask_prob = 1.0 - config.profile_prob - config.same_prob - config.uniform_prob
	assert mask_prob >= 0

	categorical_probs = F.pad(categorical_probs, pad_shapes, value=mask_prob)
	mask_position = torch.rand(msa.shape, device=msa.device) < config.replace_fraction
	mask_position = mask_position.int() * feature_dict['msa_mask'].int()
	
	bert_msa = shaped_categorical(categorical_probs)
	bert_msa = torch.where(mask_position==1, bert_msa, msa.long())
	bert_msa *= feature_dict['msa_mask'].long()

	feature_dict["bert_mask"] = mask_position.to(torch.float32)
	feature_dict["true_msa"] = feature_dict["msa"]
	feature_dict["msa"] = bert_msa.to(torch.float32)

	return feature_dict

def nearest_neighbor_clusters(feature_dict, gap_agreement_weight=0.0):
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/model/modules_multimer.py#L160"""
	weights = torch.cat([torch.ones(21, device=feature_dict["msa"].device), 
						gap_agreement_weight*torch.ones(1, device=feature_dict["msa"].device), 
						torch.zeros(1, device=feature_dict["msa"].device)], dim=0)
	
	msa_one_hot = F.one_hot(feature_dict['msa'].long(), 23)
	extra_msa_one_hot = F.one_hot(feature_dict['extra_msa'].long(), 23)
	msa_one_hot_masked = feature_dict['msa_mask'][:,:,None] * msa_one_hot
	extra_one_hot_masked = feature_dict['extra_msa_mask'][:,:,None] * extra_msa_one_hot
	
	agreement = torch.einsum('mrc, nrc->nm', extra_one_hot_masked, weights * msa_one_hot_masked)
	cluster_assignment = F.softmax(1e3 * agreement, dim=0)
	cluster_assignment *= torch.einsum('mr, nr -> mn', feature_dict['msa_mask'], feature_dict['extra_msa_mask'])

	cluster_count = torch.sum(cluster_assignment, dim=-1) + 1.0
	msa_sum = torch.einsum('nm, mrc->nrc', cluster_assignment, extra_one_hot_masked)
	msa_sum += msa_one_hot_masked
	feature_dict['cluster_profile'] = msa_sum / cluster_count[:, None, None]

	del_sum = torch.einsum('nm, mc->nc', cluster_assignment, feature_dict['extra_msa_mask'] * feature_dict['extra_deletion_matrix'])
	del_sum += feature_dict['deletion_matrix']
	feature_dict['cluster_deletion_mean'] = del_sum / cluster_count[:, None]
	
	return feature_dict


def create_msa_feat(feature_dict):
	msa_1hot = F.one_hot(feature_dict['msa'].long(), 23)
	deletion_matrix = feature_dict['deletion_matrix']
	has_deletion = torch.clip(deletion_matrix, min=0., max=1.)[..., None]
	deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / np.pi))[..., None]

	deletion_mean_value = (torch.arctan(feature_dict['cluster_deletion_mean'] / 3.) *(2. / np.pi))[..., None]

	msa_feat = [
		msa_1hot,
		has_deletion,
		deletion_value,
		feature_dict['cluster_profile'],
		deletion_mean_value
	]

	return torch.cat(msa_feat, dim=-1)

def create_extra_msa_feature(feature_dict, num_extra_msa):
	extra_msa = feature_dict['extra_msa'][:num_extra_msa]
	deletion_matrix = feature_dict['extra_deletion_matrix'][:num_extra_msa]
	msa_1hot = F.one_hot(extra_msa.long(), 23)
	has_deletion = torch.clip(deletion_matrix, min=0., max=1.)[..., None]
	deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / np.pi))[..., None]
	extra_msa_mask = feature_dict['extra_msa_mask'][:num_extra_msa]
	return torch.cat([msa_1hot, has_deletion, deletion_value], dim=-1), extra_msa_mask



def gumbel_argsort_sample_idx(logits):
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/model/modules_multimer.py#L93
	Samples with replacement from a distribution given by 'logits'.
	This uses Gumbel trick to implement the sampling an efficient manner. For a
	distribution over k items this samples k times without replacement, so this
	is effectively sampling a random permutation with probabilities over the
	permutations derived from the logprobs.
	Args:
	key: prng key.
	logits: Logarithm of probabilities to sample from, probabilities can be
		unnormalized.
	Returns:
	Sample from logprobs in one-hot form.
	"""
	z = Gumbel(torch.zeros_like(logits), torch.ones_like(logits)).sample()
	perm = torch.argsort(z, dim=-1)
	return torch.flip(perm, dims=(0,))