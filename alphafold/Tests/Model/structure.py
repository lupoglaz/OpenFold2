import torch
import argparse
from pathlib import Path
from alphafold.Model import model_config
import numpy as np
from alphafold.Tests.utils import check_recursive, load_data, get_total_alloc, mem_to_str
from alphafold.Model.protein import atom37_to_frames, atom37_to_torsion_angles, make_backbone_frames, frame_aligned_point_error
from alphafold.Model import affine

def frame_aligned_point_error_test(args):
	(activations1, activations2, frames_mask, pos_mask), res = load_data(args, 'frame_aligned_point_error_test')
	qa1 = affine.QuatAffine.from_tensor(activations1)
	rigs1 = qa1.to_rigids()
	qa2 = affine.QuatAffine.from_tensor(activations2)
	rigs2 = qa2.to_rigids()

	this_res = frame_aligned_point_error(rigs1, rigs2, frames_mask, rigs1.trans, rigs2.trans, pos_mask, 1.0)
	check_recursive(res, this_res)

def atom37_to_frames_test(args):
	(aatype, all_atom_positions, all_atom_mask), res = load_data(args, 'atom37_to_frames_test')
	this_res = atom37_to_frames(aatype.to(torch.long), all_atom_positions, all_atom_mask)
	check_recursive(res, this_res)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
	args = parser.parse_args()

	# frame_aligned_point_error_test(args)
	atom37_to_frames_test(args)