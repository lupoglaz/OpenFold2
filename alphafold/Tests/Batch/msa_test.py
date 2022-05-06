from typing import Dict
import torch
import argparse
from pathlib import Path
import pickle
import numpy as np
from alphafold.Model import model_config
import numpy as np
from collections import namedtuple

from alphafold.Model.msa import Attention, GlobalAttention, MSARowAttentionWithPairBias, MSAColumnAttention, MSAColumnGlobalAttention
from alphafold.Model.Opt.msa import AttentionOpt, GlobalAttentionOpt, MSARowAttentionWithPairBiasOpt, MSAColumnAttentionOpt, MSAColumnGlobalAttentionOpt
from alphafold.Model.Opt.fastfold_msa import AttentionFF, MSAColumnAttentionFF, MSARowAttentionWithPairBiasFF
from alphafold.Model.Opt.batch_msa import AttentionFFB, MSARowAttentionWithPairBiasFFB, MSAColumnAttentionFFB, MSAColumnGlobalAttentionOptB

from alphafold.Tests.utils import check_recursive, load_data, get_total_alloc, mem_to_str

from pytorch_memlab import LineProfiler
from pytorch_memlab import MemReporter


def MSARowAttentionWithPairBiasTest(args, config, global_config, is_training = False):
	feat, params, res = load_data(args, 'MSARowAttentionWithPairBias')
	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param)
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	conf.dropout_rate = 0.0
	attn_single = MSARowAttentionWithPairBiasFF(conf, global_config, pair_dim=feat['pair_act'].shape[-1], msa_dim=feat['msa_act'].shape[-1])
	attn_single.load_weights_from_af2(params, rel_path='msa_row_attention_with_pair_bias')
	attn_batch = MSARowAttentionWithPairBiasFFB(conf, global_config, pair_dim=feat['pair_act'].shape[-1], msa_dim=feat['msa_act'].shape[-1])
	attn_batch.load_weights_from_af2(params, rel_path='msa_row_attention_with_pair_bias')
	
	attn_single.cuda()
	attn_batch.cuda()
	batch_size = 8
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)#[:63,:,:]
	feat['pair_act'] = feat['pair_act'].to(device='cuda', dtype=torch.float32)#[:63,:,:]
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)#[:63,:]
	batch_msa_act = feat['msa_act'][None, ...].repeat(batch_size, 1, 1, 1)
	batch_pair_act = feat['pair_act'][None, ...].repeat(batch_size, 1, 1, 1)
	batch_msa_mask = feat['msa_mask'][None, ...].repeat(batch_size, 1, 1)
	print(feat['pair_act'].size(), batch_pair_act.size())
	print(feat['msa_act'].size(), batch_msa_act.size())
	print(feat['msa_mask'].size(), batch_msa_mask.size())

	res_single = attn_single(feat['msa_act'], feat['msa_mask'], feat['pair_act'], is_training=is_training)	
	res_batch = attn_batch(batch_msa_act, batch_msa_mask, batch_pair_act, is_training=is_training)
	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-5
	
	

def MSAColumnAttentionTest(args, config, global_config, is_training = False):
	feat, params, res = load_data(args, 'MSAColumnAttention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_column_attention
	conf.dropout_rate = 0.0
	attn_batch = MSAColumnAttentionFFB(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_batch.load_weights_from_af2(params, rel_path='msa_column_attention')
	attn_single = MSAColumnAttentionFF(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_single.load_weights_from_af2(params, rel_path='msa_column_attention')
	
	attn_batch.cuda()
	attn_single.cuda()
	batch_size = 8
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)
	batch_msa_act = feat['msa_act'][None, ...].repeat(batch_size, 1, 1, 1)
	batch_msa_mask = feat['msa_mask'][None, ...].repeat(batch_size, 1, 1)

	res_batch = attn_batch(batch_msa_act, batch_msa_mask, is_training=is_training)
	res_single = attn_single(feat['msa_act'], feat['msa_mask'], is_training=is_training)

	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-5


	
def MSAColumnGlobalAttentionTest(args, config, global_config, is_training:bool=False):
	feat, params, res = load_data(args, 'MSAColumnGlobalAttention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_column_attention
	conf.dropout_rate = 0.0
	attn_batch = MSAColumnGlobalAttentionOptB(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_batch.load_weights_from_af2(params, rel_path='msa_column_global_attention')
	attn_single = MSAColumnGlobalAttentionOpt(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_single.load_weights_from_af2(params, rel_path='msa_column_global_attention')
	
	attn_single.cuda()
	attn_batch.cuda()
	batch_size = 8
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)
	batch_msa_act = feat['msa_act'][None, ...].repeat(batch_size, 1, 1, 1)
	batch_msa_mask = feat['msa_mask'][None, ...].repeat(batch_size, 1, 1)
	print(feat['msa_act'].size(), batch_msa_act.size())
	print(feat['msa_mask'].size(), batch_msa_mask.size())

	res_single = attn_single(feat['msa_act'], feat['msa_mask'].to(dtype=torch.float32), is_training=is_training)	
	res_batch = attn_batch(batch_msa_act, batch_msa_mask.to(dtype=torch.float32), is_training=is_training)
	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-5
	

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config

	MSARowAttentionWithPairBiasTest(args, config, global_config, is_training=True)
	MSAColumnAttentionTest(args, config, global_config)
	MSAColumnGlobalAttentionTest(args, config, global_config, is_training=True)