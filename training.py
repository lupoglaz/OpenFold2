import argparse
import subprocess
from pathlib import Path
import pickle
import torch
from alphafold.Data.dataset import GeneralFileData, get_stream
from alphafold.Data.pipeline import DataPipeline
from alphafold.Model.features import AlphaFoldFeatures
from alphafold.Model.alphafold import AlphaFold
from alphafold.Model import model_config

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	# parser.add_argument('-dataset_dir', default='/media/HDD/AlphaFold2Dataset/Features', type=str)
	# parser.add_argument('-data_dir', default='/media/HDD/AlphaFold2', type=str)
	parser.add_argument('-dataset_dir', default='/media/lupoglaz/AlphaFold2Dataset/Features', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
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
	af2features = AlphaFoldFeatures(config=config, device='cuda:0', is_training=False)

	af2 = AlphaFold(config=model_config,
					num_res=256,
					target_dim=batch['target_feat'].shape[-1], 
					msa_dim=batch['msa_feat'].shape[-1],
					extra_msa_dim=25)
	
	def load_pkl(file_path):
		with open(file_path[0], 'rb') as f:
			return pickle.load(f)
	data = GeneralFileData(args.dataset_dir, allowed_suffixes=['.pkl'])
	data_stream = get_stream(data, batch_size=1)
	for [data_path] in data_stream:
		feature_dict = load_pkl(data_path)
		batch = af2features(feature_dict, random_seed=42)
		for key in batch.keys():
			print(key, batch[key].shape)
			batch[key] = batch[key][0].unsqueeze(dim=0)
