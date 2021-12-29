from typing import Dict
import torch
import argparse
from pathlib import Path
import pickle
import numpy as np
from alphafold.Model import model_config
import numpy as np

from alphafold.Model.Heads.structure import InvariantPointAttention, MultiRigidSidechain, FoldIteration, StructureModule
from alphafold.Model.Heads.lddt import PredictedLDDTHead
from alphafold.Model.Heads.resolved import ExperimentallyResolvedHead
from alphafold.Model.Heads.masked_msa import MaskedMSAHead
from alphafold.Model.Heads.distogram import DistogramHead
from alphafold.Model.Heads.aligned_error import PredictedAlignedErrorHead
from alphafold.Model.protein import torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos
from alphafold.Model import affine
from alphafold.Model.affine import QuatAffine
from alphafold.Tests.utils import load_data, check_recursive


def InvariantPointAttentionTest(args, config, global_config):
	print('InvariantPointAttentionTest')
	feat, params, res = load_data(args, 'InvariantPointAttention')
	conf = config.model.heads.structure_module
	
	attn = InvariantPointAttention(	conf, global_config, 
									num_res=feat['inputs_1d'].shape[-2], 
									num_feat_1d=feat['inputs_1d'].shape[-1],
									num_feat_2d=feat['inputs_2d'].shape[-1])
	attn.load_weights_from_af2(params, rel_path='invariant_point_attention')
	
	qa = QuatAffine.from_tensor(feat['activations'].to(dtype=torch.float32))
	this_res = attn(inputs_1d = feat['inputs_1d'], inputs_2d = feat['inputs_2d'], mask=feat['mask'], affine=qa)
	print(check_recursive(this_res, res))

def test_torsion_angles_to_frames(args):
	print('test_torsion_angles_to_frames')
	(activations, aatype, torsion_angles_sin_cos), res = load_data(args, 'test_torsion_angles_to_frames')
	rigs = QuatAffine.from_tensor(activations).to_rigids()
	this_res = torsion_angles_to_frames(aatype=aatype, backb_to_global=rigs, torsion_angles_sin_cos=torsion_angles_sin_cos)
	this_res = affine.rigids_to_tensor_flat12(this_res)
	print(check_recursive(this_res, res))

def test_frames_and_literature_positions_to_atom14_pos(args):
	print('test_frames_and_literature_positions_to_atom14_pos')
	(activations, aatype, torsion_angles_sin_cos), res = load_data(args, 'test_frames_and_literature_positions_to_atom14_pos')
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
						num_feat_1d=feat['activations']['act'].shape[-1],
						num_feat_2d=feat['static_feat_2d'].shape[-1],
						# num_feat_2d= feat['activations']['act'].shape[-1]
						)
						
	attn.load_weights_from_af2(params, rel_path='fold_iteration')
	
	this_res = attn(**feat)
	this_res[1]['sc']['atom_pos'] = affine.vecs_to_tensor(this_res[1]['sc']['atom_pos'])
	this_res[1]['sc']['frames'] = affine.rigids_to_tensor_flat12(this_res[1]['sc']['frames'])
	print(check_recursive(this_res, res))

def StructureModuleTest(args, config, global_config):
	print('StructureModuleTest')
	feat, params, res = load_data(args, 'StructureModule')
	conf = config.model.heads.structure_module
	representations = feat['representations']
	batch = feat['batch']
	
	attn = StructureModule(conf, global_config, 
						num_res=representations['single'].shape[-2],
						num_feat_1d=representations['single'].shape[-1],
						num_feat_2d=representations['pair'].shape[-1]
						)
						
	attn.load_weights_from_af2(params, rel_path='structure_module')
	
	this_res = attn(representations, batch)
	print(check_recursive(this_res, res))

def PredictedLDDTHeadTest(args, config, global_config):
	print('PredictedLDDTHeadTest')
	feat, params, res = load_data(args, 'PredictedLDDTHead')
	conf = config.model.heads.predicted_lddt
	representations = feat['representations']
	batch = feat['batch']
	
	attn = PredictedLDDTHead(conf, global_config, 
						num_feat_1d=representations['single'].shape[-1]
						)
						
	attn.load_weights_from_af2(params, rel_path='predicted_lddt_head')
	
	this_res = attn(representations, batch)
	print(check_recursive(this_res, res))

def ExperimentallyResolvedHeadTest(args, config, global_config):
	print('ExperimentallyResolvedHeadTest')
	feat, params, res = load_data(args, 'ExperimentallyResolvedHead')
	conf = config.model.heads.experimentally_resolved
	representations = feat['representations']
	batch = feat['batch']
	
	attn = ExperimentallyResolvedHead(conf, global_config, 
									num_feat_1d=representations['single'].shape[-1]
									)
						
	attn.load_weights_from_af2(params, rel_path='experimentally_resolved_head')
	
	this_res = attn(representations, batch)
	print(check_recursive(this_res, res))

def MaskedMSAHeadTest(args, config, global_config):
	print('MaskedMSAHeadTest')
	feat, params, res = load_data(args, 'MaskedMSAHead')
	conf = config.model.heads.masked_msa
	representations = feat['representations']
	batch = feat['batch']
	
	attn = MaskedMSAHead(conf, global_config, 
						num_feat_2d=representations['msa'].shape[-1]
						)
						
	attn.load_weights_from_af2(params, rel_path='masked_msa_head')
	
	this_res = attn(representations, batch)
	print(check_recursive(this_res, res))

def DistogramHeadTest(args, config, global_config):
	print('DistogramHeadTest')
	feat, params, res = load_data(args, 'DistogramHead')
	conf = config.model.heads.distogram
	representations = feat['representations']
	batch = feat['batch']
	
	attn = DistogramHead(conf, global_config, 
						num_feat_2d=representations['pair'].shape[-1]
						)
						
	attn.load_weights_from_af2(params, rel_path='distogram_head')
	
	this_res = attn(representations, batch)
	print(check_recursive(this_res, res))

def PredictedAlignedErrorHeadTest(args, config, global_config):
	print('PredictedAlignedErrorHeadTest')
	feat, params, res = load_data(args, 'PredictedAlignedErrorHead')
	conf = config.model.heads.predicted_aligned_error
	representations = feat['representations']
	batch = feat['batch']
	
	attn = PredictedAlignedErrorHead(conf, global_config, 
						num_feat_2d=representations['pair'].shape[-1]
						)
						
	attn.load_weights_from_af2(params, rel_path='predicted_aligned_error_head')
	
	this_res = attn(representations, batch)
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
	MultiRigidSidechainTest(args, config, global_config)
	# FoldIterationTest(args, config, global_config)
	# StructureModuleTest(args, config, global_config)
	# PredictedLDDTHeadTest(args, config, global_config)
	# ExperimentallyResolvedHeadTest(args, config, global_config)
	# MaskedMSAHeadTest(args, config, global_config)
	# DistogramHeadTest(args, config, global_config)
	PredictedAlignedErrorHeadTest(args, config, global_config)