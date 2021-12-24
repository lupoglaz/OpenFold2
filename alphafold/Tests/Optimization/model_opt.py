import argparse
from pathlib import Path
import pickle
import torch

from alphafold.Tests.Model.quaternion_test import convert, check_recursive
from alphafold.Model.alphafold import AlphaFold
from alphafold.Model import model_config


def load_data(args, filename):
	with open(Path(args.debug_dir)/Path(f'{filename}.pkl'), 'rb') as f:
		fnargs, params, res = pickle.load(f)
	return convert(fnargs), params, convert(res)


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
	
	
	handler = torch.profiler.tensorboard_trace_handler(Path('Log'))
	with torch.profiler.profile(on_trace_ready=handler, with_stack=True) as profiler:
		AlphaFoldTest(args, config)
		profiler.step()