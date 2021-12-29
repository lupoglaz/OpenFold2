import argparse
from pathlib import Path
import pickle
import numpy as np
from ...Data import pipeline

import torch
from alphafold.Tests.utils import convert, check_recursive, load_data
from alphafold.Model.alphafold import AlphaFoldIteration, EmbeddingsAndEvoformer, AlphaFold
from alphafold.Model import model_config


def AlphaFoldIterationTest(args, config, global_config):
	ensembled_batch, non_ensembled_batch, params, res = load_data(args, 'AlphaFoldIteration')
	conf = config.model
	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param + '  ' + str(params[key][param].shape))
	for key in non_ensembled_batch.keys():
		print(key, non_ensembled_batch[key].shape)

	conf.embeddings_and_evoformer.recycle_pos = False
	conf.embeddings_and_evoformer.recycle_features = False
	conf.embeddings_and_evoformer.template.enabled = False
	conf.embeddings_and_evoformer.evoformer_num_block = 1
	conf.embeddings_and_evoformer.extra_msa_stack_num_block = 1
	conf.num_recycle = 0
	conf.resample_msa_in_recycling = False
	global_config.deterministic = True

	attn = AlphaFoldIteration(conf, global_config,
								num_res=non_ensembled_batch['target_feat'].shape[-2],
								target_dim=non_ensembled_batch['target_feat'].shape[-1], 
								msa_dim=non_ensembled_batch['msa_feat'].shape[-1],
								extra_msa_dim=25)
	attn.load_weights_from_af2(params, rel_path='alphafold_iteration')
	with torch.no_grad():
		this_res = attn(ensembled_batch, non_ensembled_batch, is_training=False)
	check_recursive(res, this_res)

def AlphaFoldTest(args, config):
	batch, params, res = load_data(args, 'AlphaFold')
	conf = config.model
	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param + '  ' + str(params[key][param].shape))
	for key in batch.keys():
		print(key, batch[key].shape)

	conf.embeddings_and_evoformer.recycle_pos = False
	conf.embeddings_and_evoformer.recycle_features = False
	conf.embeddings_and_evoformer.template.enabled = False
	conf.embeddings_and_evoformer.evoformer_num_block = 1
	conf.embeddings_and_evoformer.extra_msa_stack_num_block = 1
	conf.num_recycle = 0
	conf.resample_msa_in_recycling = False
	global_config.deterministic = True

	attn = AlphaFold(conf,
					num_res=batch['target_feat'].shape[-2],
					target_dim=batch['target_feat'].shape[-1], 
					msa_dim=batch['msa_feat'].shape[-1],
					extra_msa_dim=25)
	attn.load_weights_from_af2(params, rel_path='alphafold')
	with torch.no_grad():
		this_res = attn(batch, is_training=False)
	print(this_res)
	check_recursive(res, this_res)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/AlphaFold2Output', type=str)
		
	args = parser.parse_args()
	
	# params = np.load(Path(args.data_dir)/Path('params')/Path(f'params_{args.model_name}.npz'))
	# for k in params.keys():
	# 	print(k)
	# proc_features_path = Path(args.output_dir)/Path('T1024')/Path('proc_features.pkl')
	# with open(proc_features_path, 'rb') as f:
	# 	af2_proc_features = pickle.load(f)
	
	config = model_config(args.model_name)
	global_config = config.model.global_config
	AlphaFoldIterationTest(args, config, global_config)
	# AlphaFoldTest(args, config)


	
	
	