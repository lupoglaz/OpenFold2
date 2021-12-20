from typing import Dict
import torch
import argparse
from pathlib import Path
import pickle
import numpy as np
from alphafold.Model import model_config
import numpy as np

from alphafold.Model.structure import InvariantPointAttention, MultiRigidSidechain, FoldIteration
from alphafold.Model.protein import torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos
from alphafold.Model import affine
from alphafold.Model.affine import QuatAffine
from alphafold.Tests.Model.quaternion_test import load_data as quat_load_data
from alphafold.Tests.Model.quaternion_test import convert, check_recursive

def load_data(args, filename):
	with open(Path(args.debug_dir)/Path(f'{filename}.pkl'), 'rb') as f:
		fnargs, params, res = pickle.load(f)
	torch_args = convert(fnargs)
	return convert(fnargs), params, convert(res)

def InvariantPointAttentionTest(args, config, global_config):
	print('InvariantPointAttentionTest')
	feat, params, res = load_data(args, 'InvariantPointAttention')
	conf = config.model.heads.structure_module
	
	attn = InvariantPointAttention(	conf, global_config, 
									num_res=feat['inputs_1d'].shape[-2], 
									num_seq=feat['inputs_2d'].shape[-3], 
									num_feat_1d=feat['inputs_1d'].shape[-1],
									num_feat_2d=feat['inputs_2d'].shape[-1])
	attn.load_weights_from_af2(params, rel_path='invariant_point_attention')
	
	qa = QuatAffine.from_tensor(feat['activations'].to(dtype=torch.float32))
	this_res = attn(inputs_1d = feat['inputs_1d'], inputs_2d = feat['inputs_2d'], mask=feat['mask'], affine=qa)
	print(check_recursive(this_res, res))

def test_torsion_angles_to_frames(args):
	print('test_torsion_angles_to_frames')
	(activations, aatype, torsion_angles_sin_cos), res = quat_load_data(args, 'test_torsion_angles_to_frames')
	rigs = QuatAffine.from_tensor(activations).to_rigids()
	this_res = torsion_angles_to_frames(aatype=aatype, backb_to_global=rigs, torsion_angles_sin_cos=torsion_angles_sin_cos)
	this_res = affine.rigids_to_tensor_flat12(this_res)
	print(check_recursive(this_res, res))

def test_frames_and_literature_positions_to_atom14_pos(args):
	print('test_frames_and_literature_positions_to_atom14_pos')
	(activations, aatype, torsion_angles_sin_cos), res = quat_load_data(args, 'test_frames_and_literature_positions_to_atom14_pos')
	rigs = QuatAffine.from_tensor(activations).to_rigids()
	all_frames = torsion_angles_to_frames(aatype=aatype, backb_to_global=rigs, torsion_angles_sin_cos=torsion_angles_sin_cos)
	this_res = frames_and_literature_positions_to_atom14_pos(aatype=aatype, all_frames_to_global=all_frames)
	this_res = affine.vecs_to_tensor(this_res)
	print(check_recursive(this_res, res))

def MultiRigidSidechainTest(args, config, global_config):
	print('MultiRigidSidechainTest')
	feat, params, res = load_data(args, 'MultiRigidSidechain')
	conf = config.model.heads.structure_module.sidechain

	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param)
	
	attn = MultiRigidSidechain(	conf, global_config, 
								repr_dim=feat['representations_list'][0].shape[-1], 
								num_repr=len(feat['representations_list'])
								)
	attn.load_weights_from_af2(params, rel_path='rigid_sidechain')
	
	qa = QuatAffine.from_tensor(feat['activations'].to(dtype=torch.float32))
	this_res = attn(affine = qa, representations_list = feat['representations_list'], aatype=feat['aatype'])
	this_res['atom_pos'] = affine.vecs_to_tensor(this_res['atom_pos'])
	this_res['frames'] = affine.rigids_to_tensor_flat12(this_res['frames'])
	print(check_recursive(this_res, res))

def FoldIterationTest(args, config, global_config):
	print('FoldIterationTest')
	feat, params, res = load_data(args, 'FoldIteration')
	conf = config.model.heads.structure_module

	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param)
	
	attn = FoldIteration(conf, global_config, 
						num_res=feat['static_feat_2d'].shape[-3], 
						num_seq=feat['static_feat_2d'].shape[-2], 
						num_feat_1d=feat['activations']['act'].shape[-1],
						num_feat_2d=feat['static_feat_2d'].shape[-1],
						# num_feat_2d= feat['activations']['act'].shape[-1]
						)
						
	attn.load_weights_from_af2(params, rel_path='fold_iteration')
	
	this_res = attn(**feat)
	this_res[1]['sc']['atom_pos'] = affine.vecs_to_tensor(this_res[1]['sc']['atom_pos'])
	this_res[1]['sc']['frames'] = affine.rigids_to_tensor_flat12(this_res[1]['sc']['frames'])
	print(check_recursive(this_res, res))

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config
	

	# InvariantPointAttentionTest(args, config, global_config)
	# test_torsion_angles_to_frames(args)
	# test_frames_and_literature_positions_to_atom14_pos(args)
	# MultiRigidSidechainTest(args, config, global_config)
	FoldIterationTest(args, config, global_config)