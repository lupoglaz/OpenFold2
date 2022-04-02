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


from alphafold.Tests.utils import check_recursive, load_data, get_total_alloc, mem_to_str

from pytorch_memlab import LineProfiler
from pytorch_memlab import MemReporter

def AttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'Attention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	attn_opt = AttentionFF(conf, global_config, output_dim=256, key_dim=feat['q_data'].shape[-1], value_dim=feat['m_data'].shape[-1])
	attn_opt.load_weights_from_af2(params['attention'], None)
	attn_vanilla = AttentionOpt(conf, global_config, output_dim=256, key_dim=feat['q_data'].shape[-1], value_dim=feat['m_data'].shape[-1])
	attn_vanilla.load_weights_from_af2(params['attention'], None)
	
	attn_vanilla.cuda()
	feat['q_data'] = feat['q_data'].to(device='cuda', dtype=torch.float32)
	feat['m_data'] = feat['m_data'].to(device='cuda', dtype=torch.float32)
	feat['bias'] = feat['bias'].to(device='cuda', dtype=torch.float32)
	feat['nonbatched_bias'] = feat['nonbatched_bias'].to(device='cuda')
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('Attention'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(q_data=feat['q_data'], m_data=feat['q_data'], bias=feat['bias'].zero_(), nonbatched_bias=feat['nonbatched_bias'])
		profiler.step()
	
	attn_opt.cuda()
	reporter = MemReporter()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('AttentionOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(in_data=feat['q_data'], mask=feat['bias'].fill_(1.0), nonbatched_bias=feat['nonbatched_bias'][None,:,:,None])
		profiler.step()
	# reporter.report()
		
	check_recursive(res_opt, res_vanilla)
	

def GlobalAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'GlobalAttention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	attn_opt = GlobalAttentionOpt(conf, global_config, output_dim=256, key_dim=feat['q_data'].shape[-1], value_dim=feat['m_data'].shape[-1])
	attn_opt.load_weights_from_af2(params['attention'], None)
	attn_vanilla = GlobalAttention(conf, global_config, output_dim=256, key_dim=feat['q_data'].shape[-1], value_dim=feat['m_data'].shape[-1])
	attn_vanilla.load_weights_from_af2(params['attention'], None)
	
	attn_vanilla.cuda()
	feat['q_data'] = feat['q_data'].to(device='cuda', dtype=torch.float32)
	feat['m_data'] = feat['m_data'].to(device='cuda', dtype=torch.float32)
	feat['q_mask'] = feat['q_mask'].to(device='cuda', dtype=torch.float32)
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('GlobalAttention'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(q_data=feat['q_data'], m_data=feat['m_data'], q_mask=feat['q_mask'].to(dtype=torch.float32), bias=feat['bias'])	
		profiler.step()
	
	attn_opt.cuda()
	reporter = MemReporter()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('GlobalAttentionOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(q_data=feat['q_data'], m_data=feat['m_data'], q_mask=feat['q_mask'].to(dtype=torch.float32), bias=feat['bias'])
		profiler.step()
	reporter.report()
		
	check_recursive(res_opt, res_vanilla)


def MSARowAttentionWithPairBiasTest(args, config, global_config):
	feat, params, res = load_data(args, 'MSARowAttentionWithPairBias')
	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param)
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	attn_opt = MSARowAttentionWithPairBiasFF(conf, global_config, pair_dim=feat['pair_act'].shape[-1], msa_dim=feat['msa_act'].shape[-1])
	attn_opt.load_weights_from_af2(params, rel_path='msa_row_attention_with_pair_bias')
	attn_vanilla = MSARowAttentionWithPairBiasOpt(conf, global_config, pair_dim=feat['pair_act'].shape[-1], msa_dim=feat['msa_act'].shape[-1])
	attn_vanilla.load_weights_from_af2(params, rel_path='msa_row_attention_with_pair_bias')
	
	attn_vanilla.cuda()
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)#[:63,:,:]
	feat['pair_act'] = feat['pair_act'].to(device='cuda', dtype=torch.float32)#[:63,:,:]
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)#[:63,:]
	# print(feat['pair_act'].size())
	# print(feat['msa_act'].size())
	# print(feat['msa_mask'].size())
	with torch.no_grad():
		alloc_start_vanilla = get_total_alloc()
		handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('MSARowAttentionWithPairBias'))
		with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
			res_vanilla = attn_vanilla(feat['msa_act'], feat['msa_mask'], feat['pair_act'], is_training=True)	
			profiler.step()
		alloc_end_vanilla = get_total_alloc()
		
		attn_opt.cuda()
		alloc_start_opt = get_total_alloc()
		handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('MSARowAttentionWithPairBiasOpt'))
		with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
			res_opt = attn_opt(feat['msa_act'], feat['msa_mask'], feat['pair_act'], is_training=True)
			profiler.step()
		alloc_end_opt = get_total_alloc()
	
	check_recursive(res_opt, res_vanilla + feat['msa_act'])
	print(f'Mem vanilla: {mem_to_str(alloc_end_vanilla-alloc_start_vanilla)} \t opt: {mem_to_str(alloc_end_opt-alloc_start_opt)}')

def MSAColumnAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'MSAColumnAttention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_column_attention
	attn_opt = MSAColumnAttentionFF(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_opt.load_weights_from_af2(params, rel_path='msa_column_attention')
	attn_vanilla = MSAColumnAttentionOpt(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_vanilla.load_weights_from_af2(params, rel_path='msa_column_attention')
	
	attn_vanilla.cuda()
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)
	reporter = MemReporter()
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('MSAColumnAttention'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(feat['msa_act'], feat['msa_mask'], is_training=True)	
		profiler.step()
	# reporter.report()
	
	attn_opt.cuda()
	reporter = MemReporter()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('MSAColumnAttentionOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(feat['msa_act'], feat['msa_mask'], is_training=True)
		profiler.step()
	# reporter.report()
	# mask = torch.isnan(res_opt)
	# res_vanilla = res_vanilla[~mask]
	# res_opt = res_opt[~mask]
	# torch.where(, 0, res_opt)
		
	check_recursive(res_opt, res_vanilla)

def MSAColumnGlobalAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'MSAColumnGlobalAttention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.msa_column_attention
	attn_opt = MSAColumnGlobalAttentionOpt(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_opt.load_weights_from_af2(params, rel_path='msa_column_global_attention')
	attn_vanilla = MSAColumnGlobalAttention(conf, global_config, msa_dim=feat['msa_act'].shape[-1])
	attn_vanilla.load_weights_from_af2(params, rel_path='msa_column_global_attention')
	
	attn_vanilla.cuda()
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)

	reporter = MemReporter()
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('MSAColumnGlobalAttention'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(feat['msa_act'], feat['msa_mask'].to(dtype=torch.float32))	
		profiler.step()
	reporter.report()
	
	attn_opt.cuda()
	reporter = MemReporter()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('MSAColumnGlobalAttentionOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(feat['msa_act'], feat['msa_mask'].to(dtype=torch.float32))
		profiler.step()
	reporter.report()
		
	check_recursive(res_opt, res_vanilla)
	


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config

	# AttentionTest(args, config, global_config)
	# GlobalAttentionTest(args, config, global_config)
	# MSARowAttentionWithPairBiasTest(args, config, global_config)
	MSAColumnAttentionTest(args, config, global_config)
	# MSAColumnGlobalAttentionTest(args, config, global_config)