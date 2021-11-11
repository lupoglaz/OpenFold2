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

from alphafold.Model import AlphaFold, AlphaFoldFeatures, model_config

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_path', default='T1024.fas', type=str)
	parser.add_argument('-output_dir', default='/media/HDD/AlphaFold2Output', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/HDD/AlphaFold2', type=str)
	
	parser.add_argument('-jackhmmer_binary_path', default='/usr/bin/jackhmmer', type=str)
	parser.add_argument('-hhblits_binary_path', default='/usr/bin/hhblits', type=str)
	parser.add_argument('-hhsearch_binary_path', default='/usr/bin/hhsearch', type=str)
	parser.add_argument('-kalign_binary_path', default='/usr/bin/kalign', type=str)

	parser.add_argument('-uniref90_database_path', default='uniref90/uniref90.fasta', type=str)
	parser.add_argument('-mgnify_database_path', default='mgnify/mgy_clusters_2018_12.fa', type=str)
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

	torch_params = {}
	for path, array in params.items():
		scope, name = path.split('//')
		if scope not in torch_params:
			torch_params[scope] = {}
		torch_params[scope][name] = torch.from_numpy(array)
		print(scope, torch_params[scope][name].size(), name)

	model_config = model_config(args.model_name)
	model_config.data.eval.num_ensemble = 1
	af2 = AlphaFold(config=model_config)
	af2features = AlphaFoldFeatures(config=model_config)

	path = os.path.join(args.output_dir, 'T1024', f'features.pkl')
	with open(path, 'rb') as f:
		feature_dict = pickle.load(f)
	
	processed_features = af2features(feature_dict, random_seed=42)
	
	path = os.path.join(args.output_dir, 'T1024', f'proc_features.pkl')
	with open(path, 'rb') as f:
		processed_feature_dict = pickle.load(f)
	
	for k, v in processed_feature_dict.items():
		print(k, v.shape)

	prediction_result, _ = af2(processed_feature_dict, is_training=False)