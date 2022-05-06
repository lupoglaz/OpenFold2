import argparse
from pathlib import Path
import pickle
import torch

from alphafold.Tests.utils import check_recursive, load_data, get_total_alloc, mem_to_str
from alphafold.Model.alphafold import AlphaFold, EmbeddingsAndEvoformer, EvoformerIterationFF
from alphafold.Model.Opt.batch_evoformer import EvoformerIterationFFB
from alphafold.Model import model_config


def EvoformerIterationTest(args, config, global_config, is_training:bool=False):
	feat, params, res = load_data(args, 'EvoformerIteration1')
	conf = config.model.embeddings_and_evoformer.evoformer
	conf.msa_row_attention_with_pair_bias.dropout_rate = 0.0
	conf.msa_column_attention.dropout_rate = 0.0
	conf.triangle_attention_starting_node.dropout_rate = 0.0
	conf.triangle_attention_ending_node.dropout_rate = 0.0
	conf.triangle_multiplication_outgoing.dropout_rate = 0.0
	conf.triangle_multiplication_incoming.dropout_rate = 0.0
	conf.outer_product_mean.dropout_rate = 0.0
	conf.pair_transition.dropout_rate = 0.0
	
	
	attn_batch = EvoformerIterationFFB(conf, global_config, msa_dim=feat['msa_act'].shape[-1], pair_dim=feat['pair_act'].shape[-1], is_extra_msa=False)
	attn_batch.load_weights_from_af2(params, rel_path='evoformer_iteration')
	
	attn_single = EvoformerIterationFF(conf, global_config, msa_dim=feat['msa_act'].shape[-1], pair_dim=feat['pair_act'].shape[-1], is_extra_msa=False)
	attn_single.load_weights_from_af2(params, rel_path='evoformer_iteration')
	
	batch_size = 8
	feat['msa_act'] = feat['msa_act'].to(device='cuda',dtype=torch.float32)
	feat['pair_act'] = feat['pair_act'].to(device='cuda',dtype=torch.float32)
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda',dtype=torch.float32)
	feat['pair_mask'] = feat['pair_mask'].to(device='cuda',dtype=torch.float32)
	batch_msa_mask = feat['msa_mask'][None, ...].repeat(batch_size, 1, 1)
	batch_msa_act = feat['msa_act'][None, ...].repeat(batch_size, 1, 1, 1)
	batch_pair_mask = feat['pair_mask'][None, ...].repeat(batch_size, 1, 1)
	batch_pair_act = feat['pair_act'][None, ...].repeat(batch_size, 1, 1, 1)

	attn_single.cuda()
	attn_batch.cuda()
	res_single_msa, res_single_pair = attn_single(msa_act=feat['msa_act'], pair_act=feat['pair_act'], 
												msa_mask=feat['msa_mask'], pair_mask=feat['pair_mask'], is_training=is_training)
		
	res_batch_msa, res_batch_pair = attn_batch(msa_act=batch_msa_act, pair_act=batch_pair_act, 
											msa_mask=batch_msa_mask, pair_mask=batch_pair_mask, is_training=is_training)
	
	for i in range(batch_size):
		err_msa = torch.sum(torch.abs(res_batch_msa[i, ...] - res_single_msa))
		err_pair = torch.sum(torch.abs(res_batch_pair[i, ...] - res_single_pair))
		print(i, err_msa.item(), err_pair.item())
		assert (err_msa < 1e-5) and (err_pair < 1e-5)


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/AlphaFold2Output', type=str)
		
	args = parser.parse_args()

	config = model_config(args.model_name)
	global_config = config.model.global_config

	EvoformerIterationTest(args, config, global_config, is_training=True)
	