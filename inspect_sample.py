import os
import sys
import argparse
import pickle
import torch
from pathlib import Path
from typing import Any, Dict

from alphafold.Model.features import AlphaFoldFeatures
from alphafold.Model.alphafold import AlphaFold
from alphafold.Common import protein
from custom_config import model_config

import deepspeed


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')
	parser.add_argument('-dataset_dir', default='/gpfs/gpfs0/g.derevyanko/OpenFold2Dataset/Features', type=str)
	parser.add_argument('-sample_name', default='4mxn_1_a_features.pkl', type=str)
	# parser.add_argument('-sample_name', default='2xon_1_a_features.pkl', type=str)
	parser.add_argument('-log_dir', default='LogTrain', type=str)
	# parser.add_argument('-log_dir', default=None, type=str)
	parser.add_argument('-model_name', default='model_tiny', type=str)
	# parser.add_argument('-model_name', default='model_small', type=str)
	# parser.add_argument('-model_name', default='model_small', type=str)
	# parser.add_argument('-precision', default='bf16')
	parser.add_argument('-precision', default='fp16')
	parser.add_argument('-deepspeed_config_path', default='deepspeed_config.json', type=str)
	args = parser.parse_args()
	
	args.dataset_dir = Path(args.dataset_dir)
	args.sample_name = Path(args.sample_name)

	config = model_config(args.model_name)
	af2features = AlphaFoldFeatures(config=config, device=None, is_training=True)
	af2features.device = 'cuda'
	
	if args.precision == 'bf16':
		af2features.dtype = torch.bfloat16
	elif args.precision == 'fp16':
		af2features.dtype = torch.float16

	af2 = AlphaFold(config=config.model, target_dim=22, msa_dim=49, extra_msa_dim=25).to(device='cuda')
		
	os.environ['RANK'] = '0'
	os.environ['LOCAL_RANK'] = '0'
	os.environ['WORLD_SIZE'] = '1'
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '6000'
	deepspeed.init_distributed(auto_mpi_discovery=False)
	af2, optimizer, _, _ = deepspeed.initialize(
		model=af2, 
		model_parameters=af2.parameters(),
		config = args.deepspeed_config_path,
		dist_init_required=True
	)
	
	with open(args.dataset_dir/args.sample_name, 'rb') as f:
		raw_features = pickle.load(f)

	batch = af2features(raw_features)
	if args.precision == 'fp16':
		batch = af2features.convert(batch, dtypes={torch.float32: torch.float16,
													torch.float64: torch.float32})
	elif args.precision == 'bf16':
		batch = af2features.convert(batch, dtypes={torch.float32: torch.bfloat16,
														torch.float64: torch.float32})
	else:
		batch = af2features.convert(batch, dtypes={torch.float32: torch.float32,
													torch.float64: torch.float32})
	for key in batch.keys():
		print(key, torch.any(torch.isnan(batch[key])))
	
	output, loss = af2(batch)
	print('Loss:', loss)

	for key in output.keys():
		print(key, torch.any(torch.isnan(output[key])))
	

	
