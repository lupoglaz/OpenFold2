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

def image_plot(af2, thist, field):
	if field == 'msa_feat':
		raise NotImplementedError()
	else:
		batch_size = this_proc_features[field].size(0)
		size_x = this_proc_features[field].size(1)
		size_y = this_proc_features[field].size(2)
		af2t = torch.from_numpy(af2_proc_features[field])
		thist = this_proc_features[field]

	fig = plt.figure(figsize=(2*6*size_x/float(size_x+size_y), batch_size*6*size_y/float(size_x+size_y)))

	for i in range(batch_size):
		plt.subplot(batch_size,2,2*i+1)
		if i==0:
			plt.title(f'AF2: {field}')
		if size_x < size_y:
			plt.imshow(af2t[i,:,:])
		else:
			plt.imshow(af2t[i,:,:].transpose(0,1))

	
	for i in range(batch_size):
		plt.subplot(batch_size,2,2*i+2)
		if i==0:
			plt.title(f'This: {field}')
		if size_x < size_y:
			plt.imshow(thist[i,:,:])
		else:
			plt.imshow(thist[i,:,:].transpose(0,1))
	plt.tight_layout()
	plt.show()

def msa_feat_plot(af2, thist):
	field = 'msa_feat'
	
	batch_size = this_proc_features[field].size(3)
	size_x = this_proc_features[field].size(1)
	size_y = this_proc_features[field].size(2)
	af2t = torch.from_numpy(af2_proc_features[field][0,:,:,:]).transpose(1,2).transpose(0,1)
	thist = this_proc_features[field][0,:,:,:].transpose(1,2).transpose(0,1)

	N = int(np.sqrt(batch_size)) + 1
	image_this = torch.zeros(size_x*N, size_y*N)
	image_af2 = torch.zeros(size_x*N, size_y*N)
	for i in range(N):
		for j in range(N):
			idx = i*N + j
			if idx >= batch_size:
				max_i = i
				max_j = j
				break
			image_this[size_x*i:size_x*(i+1), size_y*j:size_y*(j+1)] = thist[idx,:,:]
			image_af2[size_x*i:size_x*(i+1), size_y*j:size_y*(j+1)] = af2t[idx,:,:]

	fig = plt.figure(figsize=(12,6))
	
	plt.subplot(1,2,1)
	plt.title(f'AF2: {field}')
	plt.imshow(image_af2[:size_x*(max_i-1), :])
	
	plt.subplot(1,2,2)
	plt.title(f'This: {field}')
	plt.imshow(image_this[:size_x*(max_i-1), :])
	
	plt.tight_layout()
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
	
	#Correct:
	# print(f'AF2: {af2_proc_features["seq_length"]}\nThis: {this_proc_features["seq_length"]}')
	# string_plot(af2_proc_features, this_proc_features, 'residue_index')
	# string_plot(af2_proc_features, this_proc_features, 'aatype')
	# image_plot(af2_proc_features, this_proc_features, 'bert_mask')
	# image_plot(af2_proc_features, this_proc_features, 'extra_msa_mask')
	# image_plot(af2_proc_features, this_proc_features, 'true_msa')
	# image_plot(af2_proc_features, this_proc_features, 'extra_msa')
	# msa_feat_plot(af2_proc_features, this_proc_features)
	# image_plot(af2_proc_features, this_proc_features, 'target_feat')

	# image_plot(af2_proc_features, this_proc_features, 'msa_mask')
	# string_plot(af2_proc_features, this_proc_features, 'extra_msa_row_mask')
	# string_plot(af2_proc_features, this_proc_features, 'msa_row_mask')
	# string_plot(af2_proc_features, this_proc_features, 'seq_mask')
	# image_plot(af2_proc_features, this_proc_features, 'extra_has_deletion')
	# image_plot(af2_proc_features, this_proc_features, 'extra_deletion_value')

	