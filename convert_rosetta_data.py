import argparse
import subprocess
from pathlib import Path
import pickle
import torch
from alphafold.Data.dataset import get_stream, GeneralFileData
from alphafold.Data.pipeline import DataPipeline
import numpy as np

from alphafold.Data.parsers import parse_stockholm

import shutil

def process_msa(msa_data, msa_alphabet):
	msa = []
	deletion_matrix = []
	query = ''
	keep_columns = []
	for seq_index in range(msa_data.shape[0]):
		sequence = ''.join([msa_alphabet[msa_data[seq_index, ch_index]] for ch_index in range(msa_data.shape[1])])
		if seq_index == 0:
			query = sequence
			keep_columns = [i for i, res in enumerate(query) if res != '-']
		
		aligned_seq = ''.join([sequence[c] for c in keep_columns])
		msa.append(aligned_seq)
		deletion_vec = []
		deletion_count = 0
		for seq_res, query_res in zip(sequence, query):
			if seq_res != '-' or query_res != '-':
				if query_res == '-':
					deletion_count += 1
				else:
					deletion_vec.append(deletion_count)
					deletion_count = 0
		deletion_matrix.append(deletion_vec)
	return msa, deletion_matrix

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-rosetta_data_dir', default='/media/lupoglaz/tRosettaDataset/training_set', type=str)
	parser.add_argument('-fasta_dir', default='/media/lupoglaz/AlphaFold2Dataset/Sequences', type=str)
	parser.add_argument('-pdb_dir', default='/media/lupoglaz/AlphaFold2Dataset/Structures', type=str)
	parser.add_argument('-output_msa_dir', default='/media/lupoglaz/AlphaFold2Dataset/Alignments', type=str)
	parser.add_argument('-output_feat_dir', default='/media/lupoglaz/AlphaFold2Dataset/Features', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	
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
	parser.add_argument('-preset', default='full_dbs', type=str)
	
	args = parser.parse_args()
	args.rosetta_data_dir = Path(args.rosetta_data_dir)
	assert args.rosetta_data_dir.exists()

	args.data_dir = Path(args.data_dir)
	args.fasta_dir = Path(args.fasta_dir)
	args.pdb_dir = Path(args.pdb_dir)
	args.output_msa_dir = Path(args.output_msa_dir)
	args.output_feat_dir = Path(args.output_feat_dir)

	args.uniref90_database_path = Path(args.data_dir)/Path(args.uniref90_database_path)
	args.mgnify_database_path = Path(args.data_dir)/Path(args.mgnify_database_path)
	args.bfd_database_path = Path(args.data_dir)/Path(args.bfd_database_path)
	args.small_bfd_database_path = Path(args.data_dir)/Path(args.small_bfd_database_path)
	args.uniclust30_database_path = Path(args.data_dir)/Path(args.uniclust30_database_path)
	args.pdb70_database_path = Path(args.data_dir)/Path(args.pdb70_database_path)
	args.template_mmcif_dir = Path(args.data_dir)/Path(args.template_mmcif_dir)

	jackhmmer_binary_path = Path(subprocess.run(["which", "jackhmmer"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	if not(jackhmmer_binary_path.exists()):
		jackhmmer_binary_path = Path(args.jackhammer_binary_path)
	hhblits_binary_path = Path(subprocess.run(["which", "hhblits"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	if not(hhblits_binary_path.exists()):
		hhblits_binary_path = Path(args.hhblits_binary_path)
	hhsearch_binary_path = Path(subprocess.run(["which", "hhsearch"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	if not(hhsearch_binary_path.exists()):
		hhsearch_binary_path = Path(args.hhsearch_binary_path)
	kalign_binary_path = Path(subprocess.run(["which", "kalign"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	if not(kalign_binary_path.exists()):
		kalign_binary_path = Path(args.kalign_binary_path)
	
	data_pipeline = DataPipeline(
		jackhammer_binary_path=jackhmmer_binary_path,
		hhblits_binary_path=hhblits_binary_path,
		hhsearch_binary_path=hhsearch_binary_path,
		uniref90_database_path=args.uniref90_database_path,
		mgnify_database_path=args.mgnify_database_path,
		bfd_database_path=args.bfd_database_path,
		uniclust30_database_path=args.uniclust30_database_path,
		small_bfd_database_path=args.small_bfd_database_path,
		pdb70_database_path=args.pdb70_database_path,
		template_featurizer=None,
		use_small_bfd=False)

	args.output_msa_dir.mkdir(parents=True, exist_ok=True)
	args.output_feat_dir.mkdir(parents=True, exist_ok=True)
	args.fasta_dir.mkdir(parents=True, exist_ok=True)
	args.pdb_dir.mkdir(parents=True, exist_ok=True)

	rosetta_pdb_dir = args.rosetta_data_dir/Path('pdb')
	rosetta_a3m_dir = args.rosetta_data_dir/Path('a3m')
	rosetta_msa_dir = args.rosetta_data_dir/Path('npz')
	pdb_data = GeneralFileData(rosetta_pdb_dir, allowed_suffixes=['.pdb'])
	a3m_data = GeneralFileData(rosetta_a3m_dir, allowed_suffixes=['.a3m'])
	msa_data = GeneralFileData(rosetta_msa_dir, allowed_suffixes=['.npz'])
	all_data = get_stream(pdb_data + a3m_data + msa_data, batch_size=1)

	msa_alphabet = {i: ch for i, ch in enumerate(list('ARNDCQEGHILKMFPSTWYV-'))}

	
	for [pdb_path], [a3m_path], [msa_path] in all_data:
		pdb_path, a3m_path = Path(pdb_path), Path(a3m_path)
		try:
			shutil.copy(pdb_path, args.pdb_dir / Path(pdb_path.stem + pdb_path.suffix))
			pdb_features, pdb_sequence = data_pipeline.process_pdb(pdb_path)
			
			msa_data = np.load(msa_path)['msa']
			msa, deletion_matrix = process_msa(msa_data, msa_alphabet)
			
			assert all([pdb_res == msa_res for pdb_res, msa_res in zip(pdb_sequence, msa[0])])
			
			num_res = len(pdb_sequence)
			seq_description = pdb_path.stem.upper()
			sequence_features = data_pipeline.make_sequence_features(sequence=pdb_sequence, description=seq_description, num_res=num_res)
			msa_features = data_pipeline.make_msa_features(msas=(msa,),	deletion_matrices=(deletion_matrix, ))
			
			feature_dict = {**sequence_features, **msa_features, **pdb_features}
			with open(args.output_feat_dir / Path(f'{pdb_path.stem.lower()}_features.pkl'), 'wb') as f:
				pickle.dump(feature_dict, f, protocol=4)
		except:
			continue
		