import torch
from torch import distributions
from alphafold.Common import residue_constants
from functools import reduce
from operator import add
import numpy as np
from .config import NUM_RES, NUM_EXTRA_SEQ, NUM_TEMPLATES, NUM_MSA_SEQ
import itertools

MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]

def cast_to_64bit_ints(protein):
	for k, v in protein.items():
		if v.dtype == torch.int32:
			protein[k] = v.type(torch.int64)
	return protein

def correct_msa_restypes(protein):
	new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
	new_order = torch.tensor([new_order_list]*protein['msa'].size(1), dtype=torch.long).transpose(0,1)
	protein['msa'] = torch.gather(new_order, 0, protein['msa'])

	for k in protein:
		if "profile" in k:
			#https://github.com/aqlaboratory/openfold/blob/99e3562882cc58b9cc5d46fee08afdf47eda0a09/openfold/data/data_transforms.py#L116
			raise NotImplementedError()
	return protein

def squeeze_features(protein):
	protein['aatype'] = torch.argmax(protein['aatype'], dim=-1)
	for k in [
		"domain_name", "msa", "num_alignments", "seq_length", "sequence", "superfamily", "deletion_matrix", "resolution", 
		"between_segment_residues", "residue_index", "template_all_atom_mask"
	]:
		if k in protein:
			final_dim = protein[k].shape[-1]
			if isinstance(final_dim, int) and final_dim == 1:
				protein[k] = torch.squeeze(protein[k], dim=-1)
	
	for k in ["sequence_length", "num_alignments"]:
		if k in protein:
			protein[k] = protein[k][0]
	
	return protein

def randomly_replace_msa_with_unknown(protein, replace_proportion):
	x_idx = 20
	gap_idx = 21

	msa_mask = torch.rand(protein['msa'].shape) < replace_proportion
	msa_mask = torch.logical_and(msa_mask, protein['msa'] != gap_idx)
	protein['msa'] = torch.where(msa_mask, torch.ones_like(protein['msa'])*x_idx, protein['msa'])

	aatype_mask = torch.rand(protein['aatype'].shape) < replace_proportion
	aatype_mask = torch.logical_and(aatype_mask, protein['aatype'] != gap_idx)
	protein['aatype'] = torch.where(aatype_mask, torch.ones_like(protein['aatype'])*x_idx, protein['aatype'])
	
	return protein

def make_seq_mask(protein):
	protein['seq_mask'] = torch.ones(protein['aatype'].shape, dtype=torch.float32)
	return protein

def make_msa_mask(protein):
	protein['msa_mask'] = torch.ones(protein['msa'].shape, dtype=torch.float32)
	protein['msa_row_mask'] = torch.ones(protein['msa'].shape[0], dtype=torch.float32)
	return protein

def make_one_hot(x, num_classes):
		x_one_hot = torch.zeros(*x.shape, num_classes)
		x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
		return x_one_hot

def make_hhblits_profile(protein):
	if 'hhblits_profile' in protein:
		return protein
	msa_one_hot = make_one_hot(protein['msa'], 22)
	protein['hhblits_profile'] = torch.mean(msa_one_hot, dim=0)
	return protein

def make_atom14_masks(protein):
	restype_atom14_to_atom37 = []
	restype_atom37_to_atom14 = []
	restype_atom14_mask = []
	for rt in residue_constants.restypes:
		atom_names = residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[rt]]
		restype_atom14_to_atom37.append([ (residue_constants.atom_order[name] if name else 0) for name in atom_names])
		atom_name_to_idx14 = {name:i for i,name in enumerate(atom_names)}
		restype_atom37_to_atom14.append([(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in residue_constants.atom_types])
		restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])
	restype_atom14_to_atom37.append([0]*14)
	restype_atom37_to_atom14.append([0]*37)
	restype_atom14_mask.append([0]*14)
	restype_atom14_to_atom37 = torch.tensor(restype_atom14_to_atom37, dtype=torch.int32, device=protein['aatype'].device)
	restype_atom37_to_atom14 = torch.tensor(restype_atom37_to_atom14, dtype=torch.int32, device=protein['aatype'].device)
	restype_atom14_mask = torch.tensor(restype_atom14_mask, dtype=torch.float32, device=protein['aatype'].device)
	
	protein_aatype = protein['aatype'].to(dtype=torch.long)
	protein['residx_atom14_to_atom37'] = restype_atom14_to_atom37[protein_aatype].long()
	protein['atom14_atom_exists'] = restype_atom14_mask[protein_aatype]
	protein['residx_atom37_to_atom14'] = restype_atom37_to_atom14[protein_aatype].long()

	restype_atom37_mask = torch.zeros(21,37, dtype=torch.float32, device=protein['aatype'].device)
	for restype, restype_letter in enumerate(residue_constants.restypes):
		restype_name = residue_constants.restype_1to3[restype_letter]
		atom_names = residue_constants.residue_atoms[restype_name]
		for atom in atom_names:
			atom_type = residue_constants.atom_order[atom]
			restype_atom37_mask[restype, atom_type] = 1
	protein['atom37_atom_exists'] = restype_atom37_mask[protein_aatype]
	return protein


def sample_msa(protein, max_seq, keep_extra, seed=None):
	num_seq = protein["msa"].shape[0]
	g = torch.Generator(protein["msa"].device)
	if not(seed is None):
		g.manual_seed(seed)
	shuffled = torch.randperm(num_seq - 1, generator=g) + 1
	index_order = torch.cat([torch.tensor([0]), shuffled], dim=0)
	num_sel = min(max_seq, num_seq)
	sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq-num_sel])

	for k in MSA_FEATURE_NAMES:
		if k in protein:
			if keep_extra:
				protein[f'extra_{k}'] = torch.index_select(protein[k], 0, not_sel_seq)
			protein[k] = torch.index_select(protein[k], 0, sel_seq)
	
	return protein

def sample_msa_distillation(protein, max_seq):
	if protein["is_distillation"] == 1:
		protein = sample_msa(protein, max_seq=max_seq, keep_extra=False)
	return protein

def make_masked_msa(protein, config, replace_fraction):
	def shaped_categorical(probs, epsilon=1e-10):
		ds = probs.shape
		num_classes = ds[-1]
		distribution = torch.distributions.categorical.Categorical(torch.reshape(probs+epsilon, [-1, num_classes]))
		counts = distribution.sample()
		return counts.reshape(ds[:-1])

	random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=torch.float32)
	categorical_probs = (config.uniform_prob * random_aa + 
						config.profile_prob * protein["hhblits_profile"] + 
						config.same_prob*make_one_hot(protein["msa"], 22))
	
	pad_shapes = list(reduce(add, [(0,0) for _ in range(len(categorical_probs.shape))]))
	pad_shapes[1] = 1
	
	mask_prob = 1.0 - config.profile_prob - config.same_prob - config.uniform_prob
	assert mask_prob >= 0

	categorical_probs = torch.nn.functional.pad(categorical_probs, pad_shapes, value=mask_prob)
	mask_position = torch.rand(protein['msa'].shape) < replace_fraction
	bert_msa = shaped_categorical(categorical_probs)
	bert_msa = torch.where(mask_position, bert_msa, protein['msa'])

	protein["bert_mask"] = mask_position.to(torch.float32)
	protein["true_msa"] = protein["msa"]
	protein["msa"] = bert_msa

	return protein

	

def nearest_neighbor_clusters(protein, gap_agreement_weight=0.0):
	weights = torch.cat([torch.ones(21), gap_agreement_weight*torch.ones(1), torch.zeros(1)], dim=0)
	msa_one_hot = make_one_hot(protein['msa'], 23)
	sample_one_hot = protein['msa_mask'][:,:,None] * msa_one_hot
	extra_msa_one_hot = make_one_hot(protein['extra_msa'], 23)
	extra_one_hot = protein['extra_msa_mask'][:,:,None] * extra_msa_one_hot

	num_seq, num_res, _ = sample_one_hot.shape
	extra_num_seq, _, _ = extra_one_hot.shape

	agreement = torch.matmul(
		torch.reshape(extra_one_hot, [extra_num_seq, num_res*23]),
		torch.reshape(sample_one_hot*weights, [num_seq, num_res*23]).transpose(0,1)
	)
	
	protein['extra_cluster_assignment'] = torch.argmax(agreement, dim=1).to(torch.long)
	return protein
	

def unsorted_segmented_sum(data, segment_ids, num_segments):
	assert (segment_ids.ndimension()==1 and segment_ids.shape[0]==data.shape[0])
	segment_ids = segment_ids.view(segment_ids.shape[0], *((1,)*len(data.shape[1:])))
	segment_ids = segment_ids.expand(data.shape)
	shape = [num_segments] + list(data.shape[1:])
	tensor = torch.zeros(*shape).scatter_add_(0, segment_ids, data.to(dtype=torch.float32))
	tensor = tensor.to(dtype=data.dtype)
	return tensor

def summarize_clusters(protein):
	num_seq = protein['msa'].shape[0]
	
	def cumsum(x):
		return unsorted_segmented_sum(x, protein['extra_cluster_assignment'], num_seq)

	mask = protein['extra_msa_mask']
	mask_counts = 1e-6 + protein['msa_mask'] + cumsum(mask)

	msa_sum = cumsum(mask[:,:,None] * make_one_hot(protein['extra_msa'], 23))
	msa_sum += make_one_hot(protein['msa'], 23)
	protein['cluster_profile'] = msa_sum / mask_counts[:,:,None]
	del msa_sum

	del_sum = cumsum(mask * protein['extra_deletion_matrix'])
	del_sum += protein['deletion_matrix']
	protein['cluster_deletion_mean'] = del_sum / mask_counts
	del del_sum

	return protein


def crop_extra_msa(protein, max_extra_msa):
	num_seq = protein["extra_msa"].shape[0]
	num_sel = min(max_extra_msa, num_seq)
	sel_ind = torch.randperm(num_seq)[:num_sel]
	for k in MSA_FEATURE_NAMES:
		if f'extra_{k}' in protein:
			protein[f'extra_{k}'] = torch.index_select(protein[f'extra_{k}'], 0, sel_ind)
	return protein

def delete_extra_msa(protein):
	for k in MSA_FEATURE_NAMES:
		if f'extra_{k}' in protein:
			del protein[f'extra_{k}']

def make_msa_feat(protein):
	has_break = torch.clip(protein['between_segment_residues'].to(dtype=torch.float32), 0, 1)
	aatype_one_hot = make_one_hot(protein['aatype'], 21)
	target_feat = [has_break.unsqueeze(dim=-1), aatype_one_hot]

	msa_one_hot = make_one_hot(protein['msa'], 23)
	has_deletion = torch.clip(protein['deletion_matrix'], 0, 1)
	deletion_value = torch.atan(protein['deletion_matrix']/3.0) * (2.0/np.pi)

	msa_feat = [msa_one_hot, has_deletion.unsqueeze(dim=-1), deletion_value.unsqueeze(dim=-1)]

	if 'cluster_profile' in protein:
		deletion_mean_value = torch.atan(protein['cluster_deletion_mean']/3.0) * (2.0/np.pi)
		msa_feat.extend([protein['cluster_profile'], deletion_mean_value.unsqueeze(dim=-1)])
	if 'extra_deletion_matrix' in protein:
		protein['extra_has_deletion'] = torch.clip(protein['extra_deletion_matrix'], 0.0, 1.0)
		protein['extra_deletion_value'] = torch.atan(protein['extra_deletion_matrix']/3.0) * (2.0/np.pi)

	protein['msa_feat'] = torch.cat(msa_feat, dim=-1)
	protein['target_feat'] = torch.cat(target_feat, dim=-1)
	return protein

def select_feat(protein, feature_list):
	return {k: v for k, v in protein.items() if k in feature_list}

def random_crop_to_size(protein, crop_size, max_templates, shape_schema, subsample_templates=False, seed=None):
	dev = protein['seq_length'].device
	g = torch.Generator(device=dev)
	if not(seed is None):
		g.manual_seed(seed)
	
	seq_length = protein['seq_length'][0].item()

	if 'template_mask' in protein:
		num_templates = protein['template_mask'].shape[-1]
	else:
		num_templates = 0
	subsample_templates = subsample_templates and num_templates
	
	num_res_crop_size = min(int(seq_length), crop_size)
	def _randint(lower, upper):
		return int(torch.randint(lower, upper+1, (1,), device=dev, generator=g).item())
	
	if subsample_templates:
		templates_crop_start = _randint(0, num_templates)
		templates_select_indices = torch.randperm(num_templates, device=dev)
		num_templates_crop_size = min(num_templates - templates_crop_start, max_templates)
	else:
		templates_crop_start = 0
		num_templates_crop_size = 0
		
	
	n = seq_length - num_res_crop_size
	if ('use_clamped_fape' in protein) and (protein['use_clamped_fape'] == 1.):
		right_anchor = n
	else:
		x = _randint(0, n)
		right_anchor = n - x

	num_res_crop_start = _randint(0, right_anchor)
	for k, v in protein.items():
		if not(k in shape_schema) or (('template' not in k) and (NUM_RES not in shape_schema[k])):
			continue
		if k.startswith('template') and subsample_templates:
			v = v[templates_select_indices]
		slices = []
		for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
			is_num_res = (dim_size == NUM_RES)
			if i == 0 and k.startswith('template') and subsample_templates:
				crop_size = num_templates_crop_size
				crop_start = templates_crop_start
			else:
				crop_start = num_res_crop_start if is_num_res else 0
				crop_size = num_res_crop_size if is_num_res else dim

			slices.append(slice(crop_start, crop_start+crop_size))
		protein[k] = v[slices]

	protein['seq_length'] = protein['seq_length'].new_tensor(num_res_crop_size)
	return protein

def make_fixed_size(protein, shape_schema, msa_cluster_size, extra_msa_size, num_res=0, num_templates=0):
	pad_size_map = {
		NUM_RES: num_res,
		NUM_MSA_SEQ: msa_cluster_size,
		NUM_EXTRA_SEQ: extra_msa_size,
		NUM_TEMPLATES: num_templates
	}
	for k, v in protein.items():
		if k == 'extra_cluster_assignment' or k.startswith('template_'): continue
		shape = list(v.shape)
		schema = shape_schema[k]
		assert len(shape) == len(schema), f'{k}: {shape} vs {schema}'
		pad_size = [pad_size_map.get(s2, None) or s1 for s1, s2 in zip(shape, schema)]
		padding = [(0, p-v.shape[i]) for i, p in enumerate(pad_size)]
		padding.reverse()
		padding = list(itertools.chain(*padding))
		if padding:
			protein[k] = torch.nn.functional.pad(v, padding)
			protein[k] = torch.reshape(protein[k], pad_size)
	
	return protein


def crop_templates(protein, max_templates):
	for k, v in protein.items():
		if k.startswith('template_'):
			protein[k] = v[:max_templates]
	return protein[k]



				