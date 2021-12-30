import os
import sys
import pathlib
from typing import Dict
import argparse
import random
import time
import subprocess


import json

import io
import numpy as np
import pickle
import torch

from alphafold.Model.alphafold import AlphaFold
from alphafold.Model import model_config
from alphafold.Model.features import AlphaFoldFeatures
from alphafold.Model.Utils.weights_loading import params_to_torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_path', default='T1024.fas', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/AlphaFold2Output', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	
	parser.add_argument('-jackhmmer_binary_path', default='jackhmmer', type=str)
	parser.add_argument('-hhblits_binary_path', default='hhblits', type=str)
	parser.add_argument('-hhsearch_binary_path', default='hhsearch', type=str)
	parser.add_argument('-kalign_binary_path', default='kalign', type=str)

	parser.add_argument('-uniref90_database_path', default='uniref90/uniref90.fasta', type=str)
	parser.add_argument('-mgnify_database_path', default='mgnify/mgy_clusters.fa', type=str)
	parser.add_argument('-bfd_database_path', default='bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt', type=str)
	parser.add_argument('-small_bfd_database_path', default='small_bfd/bfd-first_non_consensus_sequences.fasta', type=str)
	parser.add_argument('-uniclust30_database_path', default='uniclust30/uniclust30_2018_08/uniclust30_2018_08', type=str)
	parser.add_argument('-pdb70_database_path', default='pdb70/pdb70', type=str)
	parser.add_argument('-template_mmcif_dir', default='pdb_mmcif/mmcif_files', type=str)
	parser.add_argument('-obsolete_pdbs_path', default='pdb_mmcif/obsolete.dat', type=str)
	
	parser.add_argument('-max_template_date', default='2020-05-14', type=str)
	parser.add_argument('-preset', default='reduced_dbs', type=str)
	parser.add_argument('-benchmark', default=False, type=int)
	parser.add_argument('-random_seed', default=None, type=int)
	
	args = parser.parse_args()
	args.uniref90_database_path = os.path.join(args.data_dir, args.uniref90_database_path)
	args.mgnify_database_path = os.path.join(args.data_dir, args.mgnify_database_path)
	args.bfd_database_path = os.path.join(args.data_dir, args.bfd_database_path)
	args.small_bfd_database_path = os.path.join(args.data_dir, args.small_bfd_database_path)
	args.uniclust30_database_path = os.path.join(args.data_dir, args.uniclust30_database_path)
	args.pdb70_database_path = os.path.join(args.data_dir, args.pdb70_database_path)
	args.template_mmcif_dir = os.path.join(args.data_dir, args.template_mmcif_dir)
	args.obsolete_pdbs_path = os.path.join(args.data_dir, args.obsolete_pdbs_path)

	args.jackhmmer_binary_path = subprocess.run(["which", "jackhmmer"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	args.hhblits_binary_path = subprocess.run(["which", "hhblits"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	args.hhsearch_binary_path = subprocess.run(["which", "hhsearch"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	args.kalign_binary_path = subprocess.run(["which", "kalign"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	
	
	path = os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz')
	with open(path, 'rb') as f:
		params = np.load(io.BytesIO(f.read()), allow_pickle=False)
	params = params_to_torch(params)
	

	# af2features = AlphaFoldFeatures(config=model_config)

	# path = os.path.join(args.output_dir, 'T1024', f'features.pkl')
	# with open(path, 'rb') as f:
	# 	feature_dict = pickle.load(f)
	
	# processed_features = af2features(feature_dict, random_seed=42)
	
	
	path = os.path.join(args.output_dir, 'T1024', f'proc_features.pkl')
	with open(path, 'rb') as f:
		batch = pickle.load(f)
	
	for key in batch.keys():
		print(key, batch[key].shape)
		batch[key] = torch.from_numpy(batch[key]).to(device='cuda')
		batch[key] = batch[key][0].unsqueeze(dim=0)

	model_config = model_config(args.model_name).model
	model_config.embeddings_and_evoformer.template.enabled = False
	model_config.resample_msa_in_recycling = False
	af2 = AlphaFold(config=model_config,
					num_res=batch['target_feat'].shape[-2],
					target_dim=batch['target_feat'].shape[-1], 
					msa_dim=batch['msa_feat'].shape[-1],
					extra_msa_dim=25)
	af2.load_weights_from_af2(params, rel_path='alphafold')
	af2 = af2.cuda()
	with torch.no_grad():
		prediction_result, _ = af2(batch, is_training=False)