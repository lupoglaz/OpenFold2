import torch
import argparse
from pathlib import Path
import pickle
import numpy as np
from alphafold.Model import model_config
import numpy as np

from alphafold.Model.msa import Attention, MSARowAttentionWithPairBias, MSAColumnAttention, GlobalAttention, MSAColumnGlobalAttention

def load_data(args, filename):
	with open(Path(args.debug_dir)/Path(f'{filename}.pkl'), 'rb') as f:
		feat, params, res = pickle.load(f)
	
	for k in feat.keys():
		feat[k] = torch.from_numpy(feat[k])
	
	if isinstance(res, tuple):
		res = tuple([torch.from_numpy(res_i) for res_i in res])
	else:
		res = torch.from_numpy(res)
	
	return feat, params, res

def check_success(this_res, res):
	err = torch.abs(this_res - res)
	max_err = torch.max(err).item()
	mean_err = torch.mean(err).item()
	print(f'Max error = {max_err}, mean error = {mean_err}')
	print(f'Success = {(max_err < 1e-7) and (mean_err < 1e-8)}')

def AttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'Attention')
	# for param in params['attention'].keys():
	# 	print(param)
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	# conf.gating = False
	attn = Attention(conf, global_config, output_dim=256, key_dim=feat['q_data'].shape[-1], value_dim=feat['m_data'].shape[-1])
	attn.load_weights_from_af2(params['attention'], None)
	this_res = attn(q_data=feat['q_data'], m_data=feat['m_data'], bias=feat['bias'], nonbatched_bias=feat['nonbatched_bias'])
	
	check_success(this_res, res)

def MSARowAttentionWithPairBiasTest(args, config, global_config):
	feat, params, res = load_data(args, 'MSARowAttentionWithPairBias')
	# for key in params.keys():
	# 	print(key)
	# 	for param in params[key].keys():
	# 		print('\t' + param)
	conf = config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	# conf.gating = False
	attn = MSARowAttentionWithPairBias(conf, global_config, pair_dim=feat['pair_act'].shape[-1], msa_dim=feat['msa_act'].shape[-1])
	attn.load_weights_from_af2(params, rel_path='msa_row_attention_with_pair_bias')
	this_res = attn(feat['msa_act'], feat['msa_mask'], feat['pair_act'])
	
	check_success(this_res, res)

def MSAColumnAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'MSAColumnAttention')
	conf = config.model.embeddings_and_evoformer.evoformer.msa_column_attention
	attn = MSAColumnAttention(conf, global_config, msa_dim=feat['msa_act'].shape[-1])

	attn.load_weights_from_af2(params, rel_path='msa_column_attention')
	this_res = attn(feat['msa_act'], feat['msa_mask'])
	
	check_success(this_res, res)

def GlobalAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'GlobalAttention')
			
	conf = config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	attn = GlobalAttention(conf, global_config, output_dim=256, key_dim=feat['q_data'].shape[-1], value_dim=feat['m_data'].shape[-1])
	attn.load_weights_from_af2(params['attention'], None)
	this_res = attn(q_data=feat['q_data'], m_data=feat['m_data'], q_mask=feat['q_mask'], bias=feat['bias'])
	
	check_success(this_res, res)

def MSAColumnGlobalAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'MSAColumnGlobalAttention')
	conf = config.model.embeddings_and_evoformer.evoformer.msa_column_attention
	attn = MSAColumnGlobalAttention(conf, global_config, msa_dim=feat['msa_act'].shape[-1])

	attn.load_weights_from_af2(params, rel_path='msa_column_global_attention')
	this_res = attn(feat['msa_act'], feat['msa_mask'])
	
	check_success(this_res, res)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config

	# AttentionTest(args, config, global_config)
	# MSARowAttentionWithPairBiasTest(args, config, global_config)
	# MSAColumnAttentionTest(args, config, global_config)
	# GlobalAttentionTest(args, config, global_config)
	# MSAColumnGlobalAttentionTest(args, config, global_config)

	
	