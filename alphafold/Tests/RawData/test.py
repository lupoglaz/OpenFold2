import argparse
import subprocess
from pathlib import Path
import pickle
import numpy as np
from ...Data import pipeline

def run(input_fasta_path: Path, msa_output_dir: Path, output_path: Path, data_pipeline: pipeline.DataPipeline):
	msa_output_dir.mkdir(exist_ok=True)
	feature_dict = data_pipeline.process(input_fasta_path=input_fasta_path, msa_output_dir=msa_output_dir)
	with open(output_path, 'wb') as f:
		pickle.dump(feature_dict, f)

def check(af2_output_path: Path, this_output_path: Path):
	with open(af2_output_path, 'rb') as f:
		af2_feature_dict = pickle.load(f)
	with open(this_output_path, 'rb') as f:
		this_feature_dict = pickle.load(f)
	common_keys = set(af2_feature_dict.keys()) & set(this_feature_dict.keys())
	missing_keys = set(af2_feature_dict.keys()) - common_keys
	print(f'Common keys: {common_keys}')
	print(f'Missing keys: {missing_keys}')
	for key in common_keys:
		print(f'{key}\t{this_feature_dict[key].shape}\t{af2_feature_dict[key].shape}')
		af2_feature = af2_feature_dict[key]
		this_feature = this_feature_dict[key]
		if af2_feature.dtype == np.object_:
			print(f'Objects:\n{af2_feature}\n{this_feature}')
			for af2ch, thisch in zip(af2_feature[0], this_feature[0]):
				assert af2ch == thisch
		else:
			diff_norm = np.linalg.norm(af2_feature - this_feature)
			print(f'Diff norm = {diff_norm}')
			assert diff_norm < 1e-5

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_path', default='T1024.fas', type=str)
	parser.add_argument('-output_dir', default='/media/lupoglaz/AlphaFold2Output', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
	
	parser.add_argument('-uniref90_database_path', default='uniref90/uniref90.fasta', type=str)
	parser.add_argument('-mgnify_database_path', default='mgnify/mgy_clusters.fa', type=str)
	parser.add_argument('-bfd_database_path', default='bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt', type=str)
	parser.add_argument('-small_bfd_database_path', default='small_bfd/bfd-first_non_consensus_sequences.fasta', type=str)
	parser.add_argument('-uniclust30_database_path', default='uniclust30/uniclust30_2018_08/uniclust30_2018_08', type=str)
	parser.add_argument('-pdb70_database_path', default='pdb70/pdb70', type=str)
	parser.add_argument('-template_mmcif_dir', default='pdb_mmcif/mmcif_files', type=str)
	
	args = parser.parse_args()
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

	# run(input_fasta_path=Path('T1024.fas'), msa_output_dir=Path('Tmp'),	output_path=Path('Tmp/this_output.pkl'), data_pipeline=data_pipeline)
	check(af2_output_path=Path(args.output_dir)/Path('T1024')/Path('features.pkl'), this_output_path=Path('Tmp/this_output.pkl'))