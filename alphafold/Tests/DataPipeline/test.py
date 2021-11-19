import argparse
import subprocess
from pathlib import Path
import pickle
import numpy as np
from ...Data import pipeline
from alphafold.Model import AlphaFold, AlphaFoldFeatures, model_config
import torch
import numpy as np
import matplotlib.pylab as plt

def string_plot(af2, thist, field):
	af2t = torch.from_numpy(af2_proc_features[field][0, :])
	thist = this_proc_features[field][0, :]
	N = af2t.shape[0]
	M = int(np.sqrt(N))
	af2t = af2t[:M*M].view(M,M)
	thist = thist[:M*M].view(M,M)
	
	plt.subplot(1,2,1)
	plt.title(f'AF2: {field}')
	plt.imshow(af2t)
	
	plt.subplot(1,2,2)
	plt.title(f'This: {field}')
	plt.imshow(thist)
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_path', default='T1024.fas', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/AlphaFold2Output', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
		
	args = parser.parse_args()
	
	model_config = model_config(args.model_name)
	model_config.data.eval.num_ensemble = 1
	model_config.data.common.use_templates = False
	af2features = AlphaFoldFeatures(config=model_config)

	features_path = Path(args.output_dir)/Path('T1024')/Path('features.pkl')
	proc_features_path = Path(args.output_dir)/Path('T1024')/Path('proc_features.pkl')
	with open(features_path, 'rb') as f:
		raw_feature_dict = pickle.load(f)
	with open(proc_features_path, 'rb') as f:
		af2_proc_features = pickle.load(f)
	
	# for k, v in feature_dict.items():
	# 	print(k, v.shape)
	# print('\n\n\n')
	# for k, v in af2_proc_feature_dict.items():
	# 	print(k, v.shape, v.dtype)

	this_proc_features = af2features(raw_feature_dict, random_seed=42)
	
	common_keys = set(af2_proc_features.keys()) & set(this_proc_features.keys())
	missing_keys = set(af2_proc_features.keys()) - common_keys
	print(missing_keys)
	for k in common_keys:
		if k.startswith('template_'):
			continue
		print(k, af2_proc_features[k].shape, this_proc_features[k].shape)
	
	
	string_plot(af2_proc_features, this_proc_features, 'extra_msa_row_mask')
	string_plot(af2_proc_features, this_proc_features, 'msa_row_mask')
	string_plot(af2_proc_features, this_proc_features, 'seq_mask')
	string_plot(af2_proc_features, this_proc_features, 'residue_index')

	