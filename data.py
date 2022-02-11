import argparse
import subprocess
from pathlib import Path
import pickle
import torch
from alphafold.Data.dataset import GeneralFileData, get_stream
from alphafold.Data.pipeline import DataPipeline

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_dir', default='/media/HDD/AlphaFold2Dataset/Sequences', type=str)
	parser.add_argument('-pdb_dir', default='/media/HDD/AlphaFold2Dataset/Structures', type=str)
	parser.add_argument('-output_msa_dir', default='/media/HDD/AlphaFold2Dataset/Alignments', type=str)
	parser.add_argument('-output_feat_dir', default='/media/HDD/AlphaFold2Dataset/Features', type=str)
	parser.add_argument('-data_dir', default='/media/HDD/AlphaFold2', type=str)
	# parser.add_argument('-fasta_dir', default='/media/lupoglaz/AlphaFold2Dataset/Sequences', type=str)
	# parser.add_argument('-pdb_dir', default='/media/lupoglaz/AlphaFold2Dataset/Structures', type=str)
	# parser.add_argument('-output_msa_dir', default='/media/lupoglaz/AlphaFold2Dataset/Alignments', type=str)
	# parser.add_argument('-output_feat_dir', default='/media/lupoglaz/AlphaFold2Dataset/Features', type=str)
	# parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	
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
	
	args = parser.parse_args()
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

	args.jackhmmer_binary_path = Path(subprocess.run(["which", "jackhmmer"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	args.hhblits_binary_path = Path(subprocess.run(["which", "hhblits"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	args.hhsearch_binary_path = Path(subprocess.run(["which", "hhsearch"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	args.kalign_binary_path = Path(subprocess.run(["which", "kalign"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
	
	data_pipeline = DataPipeline(
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

	args.output_msa_dir.mkdir(parents=True, exist_ok=True)
	args.output_feat_dir.mkdir(parents=True, exist_ok=True)
	args.fasta_dir.mkdir(parents=True, exist_ok=True)
	
	# pdb_stream = get_pdb_stream(args.pdb_dir)
	data = GeneralFileData(args.pdb_dir, allowed_suffixes=['.pdb'])
	pdb_stream = get_stream(data, batch_size=1, process_fn=None)

	for pdb_path in pdb_stream:
		pdb_path = Path(pdb_path[0][0]) #one worker
		output_path = args.output_feat_dir / Path(f'{pdb_path.stem.lower()}_features.pkl')
		if output_path.exists():
			continue

		pdb_feature_dict, sequence, fasta_path = data_pipeline.process_pdb(pdb_path, fasta_output_dir=args.fasta_dir)
		msa_feature_dict = data_pipeline.process(input_fasta_path=fasta_path,
											msa_output_dir=args.output_msa_dir,
											feat_output_dir=None
											)
		feature_dict = {**msa_feature_dict, **pdb_feature_dict}
		with open(output_path, 'wb') as f:
			pickle.dump(feature_dict, f, protocol=4)
		