import argparse
import subprocess
from pathlib import Path
import pickle
import numpy as np
from ...Data import pipeline
from alphafold.Model import AlphaFold, AlphaFoldFeatures, model_config


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_path', default='T1024.fas', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/AlphaFold2Output', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
		
	args = parser.parse_args()
	
	model_config = model_config(args.model_name)
	model_config.data.eval.num_ensemble = 1
	af2features = AlphaFoldFeatures(config=model_config)

	features_path = Path(args.output_dir)/Path('T1024')/Path('features.pkl')
	proc_features_path = Path(args.output_dir)/Path('T1024')/Path('proc_features.pkl')
	with open(features_path, 'rb') as f:
		feature_dict = pickle.load(f)
	with open(proc_features_path, 'rb') as f:
		af2_proc_feature_dict = pickle.load(f)
	
	for k, v in feature_dict.items():
		print(k, v.shape)
	print('\n\n\n')
	for k, v in af2_proc_feature_dict.items():
		print(k, v.shape, v.dtype)

	this_proc_features = af2features(feature_dict, random_seed=42)