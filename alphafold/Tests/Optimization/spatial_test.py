import torch
import argparse
from pathlib import Path
from alphafold.Model import model_config
import numpy as np
from alphafold.Tests.utils import check_recursive, load_data, get_total_alloc, mem_to_str
from alphafold.Model.spatial import TriangleAttention, TriangleMultiplication, OuterProductMean, Transition
from alphafold.Model.Opt.spatial import TriangleAttentionOpt, TriangleMultiplicationOpt, OuterProductMeanOpt, TransitionOpt

def TriangleAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'TriangleAttention')
		
	conf = config.model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node
	attn_opt = TriangleAttentionOpt(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_opt.load_weights_from_af2(params, 'triangle_attention')
	attn_vanilla = TriangleAttention(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_vanilla.load_weights_from_af2(params, 'triangle_attention')
	

	attn_vanilla.cuda()
	feat['pair_act'] = feat['pair_act'].to(device='cuda', dtype=torch.float32)
	feat['pair_mask'] = feat['pair_mask'].to(device='cuda', dtype=torch.float32)
	
	alloc_start_vanilla = get_total_alloc()
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('TriangleAttention'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(feat['pair_act'], feat['pair_mask'])	
		profiler.step()
	alloc_end_vanilla = get_total_alloc()

	attn_opt.cuda()
	alloc_start_opt = get_total_alloc()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('TriangleAttentionOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(feat['pair_act'], feat['pair_mask'])
		profiler.step()
	alloc_end_opt = get_total_alloc()
		
	check_recursive(res_opt, res_vanilla)
	print(f'Mem vanilla: {mem_to_str(alloc_end_vanilla-alloc_start_vanilla)} \t opt: {mem_to_str(alloc_end_opt-alloc_start_opt)}')


def TriangleMultiplicationTest(args, config, global_config):
	feat, params, res = load_data(args, 'TriangleMultiplication')
		
	conf = config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing
	# conf = config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming
	attn_opt = TriangleMultiplicationOpt(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_opt.load_weights_from_af2(params, 'triangle_multiplication')
	attn_vanilla = TriangleMultiplication(conf, global_config, pair_dim=feat['pair_act'].shape[-1])
	attn_vanilla.load_weights_from_af2(params, 'triangle_multiplication')
	

	attn_vanilla.cuda()
	feat['pair_act'] = feat['pair_act'].to(device='cuda', dtype=torch.float32)
	feat['pair_mask'] = feat['pair_mask'].to(device='cuda', dtype=torch.float32)
	
	alloc_start_vanilla = get_total_alloc()
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('TriangleMultiplication'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(feat['pair_act'], feat['pair_mask'])	
		profiler.step()
	alloc_end_vanilla = get_total_alloc()

	attn_opt.cuda()
	alloc_start_opt = get_total_alloc()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('TriangleMultiplicationOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(feat['pair_act'], feat['pair_mask'])
		profiler.step()
	alloc_end_opt = get_total_alloc()
		
	check_recursive(res_opt, res_vanilla)
	print(f'Mem vanilla: {mem_to_str(alloc_end_vanilla-alloc_start_vanilla)} \t opt: {mem_to_str(alloc_end_opt-alloc_start_opt)}')

def OuterProductMeanTest(args, config, global_config):
	feat, params, res = load_data(args, 'OuterProductMean')
		
	conf = config.model.embeddings_and_evoformer.evoformer.outer_product_mean
	attn_opt = OuterProductMeanOpt(conf, global_config, msa_dim=feat['msa_act'].shape[-1], num_output_channel=256)
	attn_opt.load_weights_from_af2(params, 'outer_product_mean')
	attn_vanilla = OuterProductMean(conf, global_config, msa_dim=feat['msa_act'].shape[-1], num_output_channel=256)
	attn_vanilla.load_weights_from_af2(params, 'outer_product_mean')
	

	attn_vanilla.cuda()
	feat['msa_act'] = feat['msa_act'].to(device='cuda', dtype=torch.float32)
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda', dtype=torch.float32)
	
	alloc_start_vanilla = get_total_alloc()
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('OuterProductMean'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(feat['msa_act'], feat['msa_mask'])	
		profiler.step()
	alloc_end_vanilla = get_total_alloc()

	attn_opt.cuda()
	alloc_start_opt = get_total_alloc()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('OuterProductMeanOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(feat['msa_act'], feat['msa_mask'])
		profiler.step()
	alloc_end_opt = get_total_alloc()
		
	check_recursive(res_opt, res_vanilla)
	print(f'Mem vanilla: {mem_to_str(alloc_end_vanilla-alloc_start_vanilla)} \t opt: {mem_to_str(alloc_end_opt-alloc_start_opt)}')


def TransitionTest(args, config, global_config):
	feat, params, res = load_data(args, 'Transition')

	conf = config.model.embeddings_and_evoformer.evoformer.pair_transition
	attn_opt = TransitionOpt(conf, global_config, num_channel=feat['seq_act'].shape[-1])
	attn_opt.load_weights_from_af2(params, 'transition_block')
	attn_vanilla = Transition(conf, global_config, num_channel=feat['seq_act'].shape[-1])
	attn_vanilla.load_weights_from_af2(params, 'transition_block')
	

	attn_vanilla.cuda()
	feat['seq_act'] = feat['seq_act'].to(device='cuda', dtype=torch.float32)
	feat['seq_mask'] = feat['seq_mask'].to(device='cuda', dtype=torch.float32)
	
	alloc_start_vanilla = get_total_alloc()
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('Transition'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(feat['seq_act'], feat['seq_mask'])	
		profiler.step()
	alloc_end_vanilla = get_total_alloc()

	attn_opt.cuda()
	alloc_start_opt = get_total_alloc()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('TransitionOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(feat['seq_act'], feat['seq_mask'])
		profiler.step()
	alloc_end_opt = get_total_alloc()
		
	check_recursive(res_opt, res_vanilla)
	print(f'Mem vanilla: {mem_to_str(alloc_end_vanilla-alloc_start_vanilla)} \t opt: {mem_to_str(alloc_end_opt-alloc_start_opt)}')
	

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config

	# TriangleAttentionTest(args, config, global_config)
	# TriangleMultiplicationTest(args, config, global_config)
	# OuterProductMeanTest(args, config, global_config)
	TransitionTest(args, config, global_config)
	