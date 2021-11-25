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

from alphafold.Model.embedders import InputEmbeddings, RecycleEmbedding, ExtraMSAEmbedding

def InputEmbeddingTest(input, output, params, config):
	target_feat_dim = input['target_feat'].shape[-1]
	msa_feat_dim = input['msa_feat'].shape[-1]
	pair_emb_dim = config.model.embeddings_and_evoformer.pair_channel
	msa_emb_dim = config.model.embeddings_and_evoformer.msa_channel
	relpos_wind = config.model.embeddings_and_evoformer.max_relative_feature
	
	ie = InputEmbeddings(target_feat_dim=target_feat_dim, 
						msa_feat_dim=msa_feat_dim, 
						pair_emb_dim=pair_emb_dim, 
						msa_emb_dim=msa_emb_dim, 
						relpos_wind=relpos_wind)

	ie.load_weights_from_af2(params)

	msa_act, pair_act = ie( torch.from_numpy(input['target_feat']), 
							torch.from_numpy(input['residue_index']), 
							torch.from_numpy(input['msa_feat']))
	
	print(msa_act.size(), pair_act.size())
	return msa_act, pair_act
	

def RecycleEmbeddingTest(input, output, params, config):
	msa_emb_dim = config.model.embeddings_and_evoformer.msa_channel
	pair_emb_dim = config.model.embeddings_and_evoformer.pair_channel
	min_bin = config.model.embeddings_and_evoformer.prev_pos.min_bin
	max_bin = config.model.embeddings_and_evoformer.prev_pos.max_bin
	num_bins = config.model.embeddings_and_evoformer.prev_pos.num_bins
	
	re = RecycleEmbedding(msa_emb_dim, pair_emb_dim, min_bin, max_bin, num_bins)
	re.load_weights_from_af2(params)

def ExtraMSAEmbeddingTest(input, output, params, config):
	print(input['extra_msa'].shape)
	msa_feat_dim = input['extra_msa'].shape[-1]
	msa_emb_dim = config.model.embeddings_and_evoformer.msa_channel

	eme = ExtraMSAEmbedding(msa_dim=msa_feat_dim, msa_emb_dim=msa_emb_dim)
	eme.load_weights_from_af2(params)

	extra_msa_act = eme(input['extra_msa'])
	print(extra_msa_act.size())
	return extra_msa_act

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/AlphaFold2Output', type=str)
		
	args = parser.parse_args()
	
	params = np.load(Path(args.data_dir)/Path('params')/Path(f'params_{args.model_name}.npz'))
	for k in params.keys():
		print(k)
	proc_features_path = Path(args.output_dir)/Path('T1024')/Path('proc_features.pkl')
	with open(proc_features_path, 'rb') as f:
		af2_proc_features = pickle.load(f)
	
	config = model_config(args.model_name)

	# msa_act, pair_act = InputEmbeddingTest(af2_proc_features, None, params, config)
	# RecycleEmbeddingTest( None, None, params, config)
	ExtraMSAEmbeddingTest(af2_proc_features, None, params, config)
	
	