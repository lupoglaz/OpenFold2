import argparse
import subprocess
from pathlib import Path
import pickle
import torch
from alphafold.Data.dataset import get_fasta_stream, get_pdb_stream, get_data_stream
from alphafold.Data.pipeline import DataPipeline
from alphafold.Model.features import AlphaFoldFeatures
from alphafold.Model import model_config

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-dataset_dir', default='/media/HDD/AlphaFold2Dataset/Features', type=str)
	parser.add_argument('-data_dir', default='/media/HDD/AlphaFold2', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)

	args = parser.parse_args()
	args.data_dir = Path(args.data_dir)
	args.dataset_dir = Path(args.dataset_dir)

	config = model_config(args.model_name)
	model_config = config.model
	model_config.embeddings_and_evoformer.template.enabled = False
	model_config.resample_msa_in_recycling = False
	data_config = config.data
	data_config.eval.num_ensemble = 1
	data_config.common.use_templates = False
	af2features = AlphaFoldFeatures(config=config, device='cuda:0', is_training=True)

	data_stream = get_data_stream(args.dataset_dir)
	for feature_dict in data_stream:
		print(feature_dict.keys())
		batch = af2features(feature_dict, random_seed=42)
		for key in batch.keys():
			print(key, batch[key].shape)
			batch[key] = batch[key][0].unsqueeze(dim=0)
