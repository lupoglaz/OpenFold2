import torch
import argparse
from pathlib import Path
from alphafold.Model import model_config
import numpy as np
from alphafold.Tests.utils import check_recursive, load_data, get_total_alloc, mem_to_str
from alphafold.Model.spatial import TriangleAttention, TriangleMultiplication, OuterProductMean, Transition
from alphafold.Model.Opt.spatial import TriangleAttentionOpt, TriangleMultiplicationOpt, OuterProductMeanOpt, TransitionOpt
from alphafold.Model.Opt.fastfold_spatial import TriangleAttentionFF, TriangleMultiplicationFF, OuterProductMeanFF, TransitionFF
from alphafold.Model.Opt.batch_spatial import TriangleAttentionFFB, TriangleMultiplicationFFB, OuterProductMeanFFB, TransitionFFB, InvariantPointAttentionB

from alphafold.Model.Heads.structure import InvariantPointAttention
from alphafold.Model.affine import QuatAffine

def TriangleAttentionTest(args, config, global_config, is_training:bool=False):
	feat, params, res = load_data(args, 'TriangleAttention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node
	# conf = config.model.embeddings_and_evoformer.evoformer.triangle_attention_ending_node
	conf.dropout_rate = 0.0
	attn_batch = TriangleAttentionFFB(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_batch.load_weights_from_af2(params, 'triangle_attention')
	attn_single = TriangleAttentionFF(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_single.load_weights_from_af2(params, 'triangle_attention')
	
	attn_single.cuda()
	attn_batch.cuda()
	batch_size = 8
	feat['pair_act'] = feat['pair_act'].to(device='cuda', dtype=torch.float32)
	feat['pair_mask'] = feat['pair_mask'].to(device='cuda', dtype=torch.float32)
	batch_pair_mask = feat['pair_mask'][None, ...].repeat(batch_size, 1, 1)
	batch_pair_act = feat['pair_act'][None, ...].repeat(batch_size, 1, 1, 1)
		
	res_single = attn_single(feat['pair_act'], feat['pair_mask'], is_training=is_training)
	res_batch = attn_batch(batch_pair_act, batch_pair_mask, is_training=is_training)
	
	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-5


def TriangleMultiplicationTest(args, config, global_config, is_training:bool=False):
	feat, params, res = load_data(args, 'TriangleMultiplication')
		
	conf = config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing
	# conf = config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming
	conf.dropout_rate = 0.0
	attn_batch = TriangleMultiplicationFFB(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_batch.load_weights_from_af2(params, 'triangle_multiplication')
	attn_single = TriangleMultiplicationFF(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_single.load_weights_from_af2(params, 'triangle_multiplication')
	

	attn_single.cuda()
	attn_batch.cuda()
	batch_size = 8
	feat['pair_act'] = feat['pair_act'].to(device='cuda', dtype=torch.float32)
	feat['pair_mask'] = feat['pair_mask'].to(device='cuda', dtype=torch.float32)
	batch_pair_mask = feat['pair_mask'][None, ...].repeat(batch_size, 1, 1)
	batch_pair_act = feat['pair_act'][None, ...].repeat(batch_size, 1, 1, 1)
		
	res_single = attn_single(feat['pair_act'], feat['pair_mask'], is_training=is_training)
	res_batch = attn_batch(batch_pair_act, batch_pair_mask, is_training=is_training)
	
	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-5

def OuterProductMeanTest(args, config, global_config, is_training:bool=False):
	feat, params, res = load_data(args, 'OuterProductMean')
		
	conf = config.model.embeddings_and_evoformer.evoformer.outer_product_mean
	conf.dropout_rate = 0.0
	attn_batch = OuterProductMeanFFB(conf, global_config, msa_dim=feat['msa_act'].shape[-1], num_output_channel=256)
	attn_batch.load_weights_from_af2(params, 'outer_product_mean')
	attn_single = OuterProductMeanFF(conf, global_config, msa_dim=feat['msa_act'].shape[-1], num_output_channel=256)
	attn_single.load_weights_from_af2(params, 'outer_product_mean')
	
	attn_single.cuda()
	attn_batch.cuda()
	batch_size = 8
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)[:,:127,:]
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)[:,:127]
	batch_msa_mask = feat['msa_mask'][None, ...].repeat(batch_size, 1, 1)
	batch_msa_act = feat['msa_act'][None, ...].repeat(batch_size, 1, 1, 1)

	res_single = attn_single(feat['msa_act'], feat['msa_mask'], is_training=is_training)
	res_batch = attn_batch(batch_msa_act, batch_msa_mask, is_training=is_training)
	
	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-5


def TransitionTest(args, config, global_config,  is_training:bool=False):
	feat, params, res = load_data(args, 'Transition')

	conf = config.model.embeddings_and_evoformer.evoformer.pair_transition
	conf.dropout_rate = 0.0
	global_config.subbatch_size = 2
	attn_batch = TransitionFFB(conf, global_config, num_channel=feat['seq_act'].shape[-1])
	attn_batch.load_weights_from_af2(params, 'transition_block')
	attn_single = TransitionFF(conf, global_config, num_channel=feat['seq_act'].shape[-1])
	attn_single.load_weights_from_af2(params, 'transition_block')
	

	attn_single.cuda()
	attn_batch.cuda()
	batch_size = 8
	feat['seq_act'] = feat['seq_act'].to(device='cuda', dtype=torch.float32)
	feat['seq_mask'] = feat['seq_mask'].to(device='cuda', dtype=torch.float32)
		
	batch_seq_mask = feat['seq_mask'][None, ...].repeat(batch_size, 1, 1)
	batch_seq_act = feat['seq_act'][None, ...].repeat(batch_size, 1, 1, 1)

	res_single = attn_single(feat['seq_act'], feat['seq_mask'], is_training=is_training)
	res_batch = attn_batch(batch_seq_act, batch_seq_mask, is_training=is_training)
	
	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-5
	

def InvariantPointAttentionTest(args, config, global_config):
	print('InvariantPointAttentionTest')
	feat, params, res = load_data(args, 'InvariantPointAttention')
	conf = config.model.heads.structure_module
	
	attn_single = InvariantPointAttention(	conf, global_config, 
											num_feat_1d=feat['inputs_1d'].shape[-1],
											num_feat_2d=feat['inputs_2d'].shape[-1])
	attn_single.load_weights_from_af2(params, rel_path='invariant_point_attention')
	attn_batch = InvariantPointAttentionB(	conf, global_config, 
											num_feat_1d=feat['inputs_1d'].shape[-1],
											num_feat_2d=feat['inputs_2d'].shape[-1])
	attn_batch.load_weights_from_af2(params, rel_path='invariant_point_attention')
	
	
	
	print('inputs1d:', feat['inputs_1d'].size())
	print('inputs2d:', feat['inputs_2d'].size())
	print('activations:', feat['activations'].size())
	print('mask:', feat['mask'].size())
	batch_size = 8
	inputs_1d_batch = feat['inputs_1d'][None, ...].repeat(batch_size, 1, 1)
	inputs_2d_batch = feat['inputs_2d'][None, ...].repeat(batch_size, 1, 1, 1)
	activations_batch = feat['activations'][None, ...].repeat(batch_size, 1, 1)
	mask_batch = feat['mask'][None, ...].repeat(batch_size, 1, 1)
	
	qa_single = QuatAffine.from_tensor(feat['activations'].to(dtype=torch.float32))
	qa_batch = QuatAffine.from_tensor(activations_batch.to(dtype=torch.float32))
	
	res_single = attn_single(inputs_1d = feat['inputs_1d'], inputs_2d = feat['inputs_2d'], mask=feat['mask'], affine=qa_single)
	res_batch = attn_batch(inputs_1d = inputs_1d_batch, inputs_2d = inputs_2d_batch, mask=mask_batch, affine=qa_batch)
	print(check_recursive(res_single, res))
	print(check_recursive(res_batch[0,...], res))

	for i in range(batch_size):
		err = torch.sum(torch.abs(res_batch[i, ...] - res_single))
		print(i, err.item())
		assert err < 1e-2



if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config

	# TriangleAttentionTest(args, config, global_config, is_training=True)
	# TriangleMultiplicationTest(args, config, global_config, is_training=True)
	# OuterProductMeanTest(args, config, global_config, is_training=True)
	# TransitionTest(args, config, global_config, is_training=True)
	InvariantPointAttentionTest(args, config, global_config)