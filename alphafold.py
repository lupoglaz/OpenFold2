from genericpath import exists
import os
import sys
import pathlib
from typing import Dict
import argparse
import random
import time
import subprocess
import json
from pathlib import Path
import io
import numpy as np
import pickle
import torch

from alphafold.Model.alphafold import AlphaFold
from alphafold.Model import model_config
from alphafold.Model.features import AlphaFoldFeatures
from alphafold.Model.Utils.weights_loading import params_to_torch
from alphafold.Common import protein, residue_constants
from alphafold.Data import pipeline



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_path', default='T1024.fas', type=str)
	parser.add_argument('-feature_path', default='/media/lupoglaz/OpenFold2Output/T1024/features.pkl', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/OpenFold2Output', type=str)
	# parser.add_argument('-output_dir', default='/media/HDD/AlphaFold2Output', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	# parser.add_argument('-data_dir', default='/media/HDD/AlphaFold2', type=str)
	
	parser.add_argument('-jackhmmer_binary_path', default='jackhmmer', type=str)
	parser.add_argument('-hhblits_binary_path', default='hhblits', type=str)
	parser.add_argument('-hhsearch_binary_path', default='hhsearch', type=str)
	parser.add_argument('-kalign_binary_path', default='kalign', type=str)

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
	args.fasta_path = Path(args.fasta_path)
	args.output_dir = Path(args.output_dir)
	args.feature_path = Path(args.feature_path)
	args.uniref90_database_path = Path(args.data_dir)/Path(args.uniref90_database_path)
	args.mgnify_database_path = Path(args.data_dir)/Path(args.mgnify_database_path)
	args.bfd_database_path = Path(args.data_dir)/Path(args.bfd_database_path)
	args.small_bfd_database_path = Path(args.data_dir)/Path(args.small_bfd_database_path)
	args.uniclust30_database_path = Path(args.data_dir)/Path(args.uniclust30_database_path)
	args.pdb70_database_path = Path(args.data_dir)/Path(args.pdb70_database_path)
	args.template_mmcif_dir = Path(args.data_dir)/Path(args.template_mmcif_dir)

	args.jackhmmer_binary_path = Path(subprocess.run(["which", "jackhmmer"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	args.hhblits_binary_path = Path(subprocess.run(["which", "hhblits"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	args.hhsearch_binary_path = Path(subprocess.run(["which", "hhsearch"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	args.kalign_binary_path = Path(subprocess.run(["which", "kalign"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
		
	data_pipeline = pipeline.DataPipeline(
		jackhammer_binary_path=args.jackhmmer_binary_path,
		hhblits_binary_path=args.hhblits_binary_path,
		hhsearch_binary_path=args.hhsearch_binary_path,
		uniref90_database_path=args.uniref90_database_path,
		mgnify_database_path=args.mgnify_database_path,
		bfd_database_path=args.bfd_database_path,
		uniclust30_database_path=args.uniclust30_database_path,
		small_bfd_database_path=args.small_bfd_database_path,
		pdb70_database_path=args.pdb70_database_path,
		template_featurizer=None,
		use_small_bfd=True)

	output_dir = args.output_dir / Path(args.fasta_path.stem)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_msa_dir = output_dir / Path('MSA')
	output_msa_dir.mkdir(parents=True, exist_ok=True)

	if not args.feature_path.exists():
		feature_dict = data_pipeline.process(input_fasta_path=args.fasta_path, 
											msa_output_dir=output_msa_dir)
		with open(output_dir / Path('features.pkl'), 'wb') as f:
			pickle.dump(feature_dict, f, protocol=4)
	else:
		with open(args.feature_path, 'rb') as f:
			feature_dict = pickle.load(f)

	
	config = model_config(args.model_name)
	model_config = config.model
	model_config.embeddings_and_evoformer.template.enabled = False
	model_config.resample_msa_in_recycling = False
	data_config = config.data
	data_config.eval.num_ensemble = 1
	data_config.common.use_templates = False
		
	af2features = AlphaFoldFeatures(config=config)
	batch = af2features(feature_dict, random_seed=42)
	for key in batch.keys():
		print(key, batch[key].shape)
		batch[key] = batch[key][0].unsqueeze(dim=0).to(device='cuda')
	
	af2 = AlphaFold(config=model_config, target_dim=22, msa_dim=49, extra_msa_dim=25)
	
	path = os.path.join(args.data_dir, 'params', f'params_{args.model_name}.npz')
	with open(path, 'rb') as f:
		params = np.load(io.BytesIO(f.read()), allow_pickle=False)
	params = params_to_torch(params)
	af2.load_weights_from_af2(params, rel_path='alphafold')
	af2 = af2.cuda()
	
	with torch.no_grad():
		prediction_result, _ = af2(batch, is_training=False, compute_loss=False)
		with open(output_dir / Path('result.pkl'), 'wb') as f:
			pickle.dump(prediction_result, f, protocol=4)
		# with open('result.pkl', 'rb') as f:
		# 	prediction_result = pickle.load(f)
				
		protein_pdb = protein.from_prediction(features=batch, result=prediction_result)
		with open(output_dir / Path('test.pdb'), 'w') as f:
			f.write(protein.to_pdb(protein_pdb))