import torch
from alphafold.Common import residues_constants

def correct_msa_restypes(protein):
	new_order_list = residues_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
	new_order = torch.tensor(new_order_list, dtype=torch.long)
	print(protein['msa'].size())
	protein['msa'] = torch.gather(protein['msa'], 0, new_order)
	print(new_order_list)