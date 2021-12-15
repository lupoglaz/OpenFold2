from typing import Dict
import torch
import argparse
from pathlib import Path
import pickle
import numpy as np
from alphafold.Model import model_config
import numpy as np

from alphafold.Model.structure import InvariantPointAttention
from alphafold.Model.affine import QuatAffine
from alphafold.Tests.Model.module_test import load_data, check_success

def InvariantPointAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'InvariantPointAttention')
	conf = config.model.heads.structure_module

	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param)
	
	attn = InvariantPointAttention(	conf, global_config, 
									num_res=feat['inputs_1d'].shape[-2], 
									num_seq=feat['inputs_2d'].shape[-3], 
									num_feat_1d=feat['inputs_1d'].shape[-1],
									num_feat_2d=feat['inputs_2d'].shape[-1])
	attn.load_weights_from_af2(params, rel_path='invariant_point_attention')
	
	qa = QuatAffine.from_tensor(feat['activations'])
	this_res = attn(inputs_1d = feat['inputs_1d'], inputs_2d = feat['inputs_2d'], mask=feat['mask'], affine=qa)
	for key in activations.keys():
		print(key)
		check_success(this_res[key], res[key])

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config
	

	InvariantPointAttentionTest(args, config, global_config)