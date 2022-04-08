import argparse
from pathlib import Path
import pickle
import torch

from alphafold.Tests.utils import check_recursive, load_data, get_total_alloc, mem_to_str
from alphafold.Model.alphafold import AlphaFold, EmbeddingsAndEvoformer, EvoformerIterationOpt, EvoformerIterationFF
from alphafold.Model import model_config


# def load_data(args, filename):
# 	with open(Path(args.debug_dir)/Path(f'{filename}.pkl'), 'rb') as f:
# 		fnargs, params, res = pickle.load(f)
# 	return convert(fnargs), params, convert(res)

def EvoformerIterationTest(args, config, global_config):
	feat, params, res = load_data(args, 'EvoformerIteration1')
	conf = config.model.embeddings_and_evoformer.evoformer
	
	attn_vanilla = EvoformerIterationOpt(conf, global_config, msa_dim=feat['msa_act'].shape[-1], pair_dim=feat['pair_act'].shape[-1], is_extra_msa=False)
	attn_vanilla.load_weights_from_af2(params, rel_path='evoformer_iteration')
	
	attn_opt = EvoformerIterationFF(conf, global_config, msa_dim=feat['msa_act'].shape[-1], pair_dim=feat['pair_act'].shape[-1], is_extra_msa=False)
	attn_opt.load_weights_from_af2(params, rel_path='evoformer_iteration')
		
	feat['msa_act'] = feat['msa_act'].to(device='cuda',dtype=torch.float32)
	feat['pair_act'] = feat['pair_act'].to(device='cuda',dtype=torch.float32)
	feat['msa_mask'] = feat['msa_mask'].to(device='cuda',dtype=torch.float32)
	feat['pair_mask'] = feat['pair_mask'].to(device='cuda',dtype=torch.float32)

	attn_vanilla.cuda()
	alloc_start_vanilla = get_total_alloc()
	handler_vanilla = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('EvoformerIteration'))
	with torch.profiler.profile(on_trace_ready=handler_vanilla, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_vanilla = attn_vanilla(msa_act=feat['msa_act'], pair_act=feat['pair_act'], 
								msa_mask=feat['msa_mask'], pair_mask=feat['pair_mask'], is_training=False)
	profiler.step()
	alloc_end_vanilla = get_total_alloc()
	
	attn_opt.cuda()
	alloc_start_opt = get_total_alloc()
	handler_opt = torch.profiler.tensorboard_trace_handler(Path('Log')/Path('EvoformerIterationOpt'))
	with torch.profiler.profile(on_trace_ready=handler_opt, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
		res_opt = attn_opt(msa_act=feat['msa_act'], pair_act=feat['pair_act'], 
						msa_mask=feat['msa_mask'], pair_mask=feat['pair_mask'], is_training=False)
		profiler.step()
	alloc_end_opt = get_total_alloc()

	check_recursive(res_opt, res_vanilla)
	print(f'Mem vanilla: {mem_to_str(alloc_end_vanilla-alloc_start_vanilla)} \t opt: {mem_to_str(alloc_end_opt-alloc_start_opt)}')
	
def EmbeddingsAndEvoformerTest(args, config, global_config):
	feat, params, res = load_data(args, 'EmbeddingsAndEvoformer')
	conf = config.model.embeddings_and_evoformer
	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param + '  ' + str(params[key][param].shape))
	for key in feat.keys():
		print(key, feat[key].shape)

	conf.template.enabled = False
	conf.recycle_pos = False
	conf.recycle_features = False
	conf.evoformer_num_block = 1
	conf.extra_msa_stack_num_block = 1
	global_config.deterministic = True
	attn = EmbeddingsAndEvoformer(conf, global_config, 
								target_dim=feat['target_feat'].shape[-1], 
								msa_dim=feat['msa_feat'].shape[-1],
								extra_msa_dim=25)
	attn.load_weights_from_af2(params, rel_path='evoformer')
	
	this_res = attn(feat, is_training=False)
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
	conf.global_config.deterministic = True

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

	config = model_config(args.model_name)
	global_config = config.model.global_config

	EvoformerIterationTest(args, config, global_config)
		
	# handler = torch.profiler.tensorboard_trace_handler(Path('Log'))
	# with torch.profiler.profile(on_trace_ready=handler, with_stack=True, with_modules=True, profile_memory=True, record_shapes=True) as profiler:
	# 	AlphaFoldTest(args, config)
	# 	profiler.step()