import torch
from alphafold.Common import residues_constants

def cast_to_64bit_ints(protein):
	for k, v in protein.items():
		if v.dtype == torch.int32:
			protein[k] = v.type(torch.int64)
	return protein

def correct_msa_restypes(protein):
	new_order_list = residues_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
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

def make_hhblits_profile(protein):
	def make_one_hot(x, num_classes):
		x_one_hot = torch.zeros(*x.shape, num_classes)
		x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
		return x_one_hot

	if 'hhblits_profile' in protein:
		return protein
	msa_one_hot = make_one_hot(protein['msa'], 22)
	protein['hhblits_profile'] = torch.mean(msa_one_hot, dim=0)
	return protein

def make_atom14_masks(protein):
	raise NotImplementedError()
				