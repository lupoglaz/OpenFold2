from pathlib import Path
from typing import Mapping, Sequence, Any, Optional
from alphafold.Data.Tools import HHSearch, HHBlits, Jackhammer
from alphafold.Data import parsers
import numpy as np

FeatureDict = Mapping[str, np.ndarray]

class DataPipeline:
	def __init__(self, 
				jackhammer_binary_path: Path,
				hhblits_binary_path: Path,
				hhsearch_binary_path: Path,
				uniref90_database_path: Path,
				mgnify_database_path: Path,
				bfd_database_path: Optional[Path],
				uniclust30_database_path: Optional[Path],
				small_bfd_database_path: Optional[Path],
				pdb70_database_path: Path,
				template_featurizer,
				use_small_bfd: bool,
				mgnify_max_hits: int=501,
				uniref_max_hits: int=10000):
		self._use_small_bfd = use_small_bfd
		self.jackhmmer_uniref90_runner = Jackhammer(
			binary_path = jackhammer_binary_path,
			database_path = uniref90_database_path
		)
		if use_small_bfd:
			self.jackhmmer_small_bfd_runner = Jackhammer(
				binary_path = jackhammer_binary_path,
				database_path = small_bfd_database_path
			)
		else:
			self.hhblits_bfd_uniclust_runner = HHBlits(
				binary_path = hhblits_binary_path,
				databases=[bfd_database_path, uniclust30_database_path]
			)
		self.jackhmmer_mgnify_runner = Jackhammer(
			binary_path = jackhammer_binary_path,
			database_path = mgnify_database_path
		)
		self.jackhmmer_pdb70_runner = Jackhammer(
			binary_path = jackhammer_binary_path,
			database_path = pdb70_database_path
		)
		self.hhsearch_pdb70_runner = HHSearch(
			binary_path = hhsearch_binary_path,
			databases = [pdb70_database_path]
		)
		self.template_featurizer = template_featurizer
		self.mgnify_max_hits = mgnify_max_hits
		self.uniref_max_hits = uniref_max_hits

	def make_sequence_features(self, sequence: str, description: str, num_res: int) ->FeatureDict:
		pass

	def make_msa_features(self, msas: Sequence[Sequence[str]], deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
		pass

	def process(self, input_fasta_path: Path, msa_ouput_dir: Path) -> FeatureDict:
		with open(input_fasta_path) as f:
			input_fasta_str = f.read()
		input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
		if len(input_seqs) != 1:
			raise ValueError(f'DataPipeline: more than one sequence found {input_fasta_path}')
		input_sequence = input_seqs[0]
		input_description = input_descs[0]
		num_res = len(input_sequence)

		jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(input_fasta_path)[0]
		jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(input_fasta_path)[0]

		uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(jackhmmer_uniref90_result['sto'], max_sequences=self.uniref_max_hits)
		hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)

		uniref90_out_path = msa_ouput_dir / Path('uniref90_hits.sto')
		with open(uniref90_out_path, 'w') as f:
			f.write(jackhmmer_uniref90_result['sto'])

		mgnify_out_path = msa_ouput_dir / Path('mgnify_hits.sto')
		with open(mgnify_out_path, 'w') as f:
			f.write(jackhmmer_mgnify_result['sto'])

		pdb70_out_path = msa_ouput_dir / Path('pdb70_hits.sto')
		with open(pdb70_out_path, 'w') as f:
			f.write(hhsearch_result)

		uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
		mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
		hhsearch_hits = parsers.parse_hhr(hhsearch_result)
		mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
		mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]

		if self._use_small_bfd:
			jackhmmer_small_bfd_results = self.jackhmmer_small_bfd_runner.query(input_fasta_path)[0]
		
			bfd_out_path = msa_ouput_dir / Path('small_bfd_hits.sto')
			with open(bfd_out_path, 'w') as f:
				f.write(jackhmmer_small_bfd_results['sto'])
			
			bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(jackhmmer_small_bfd_results['sto'])
		else:
			raise NotImplementedError()

		sequence_features = self.make_sequence_features(sequence=input_sequence, description=input_description, num_res=num_res)
		msa_features = self.make_msa_features(msas=(uniref90_msa, bfd_msa, mgnify_msa),
											deletion_matrices=(uniref90_deletion_matrix, bfd_deletion_matrix, mgnify_deletion_matrix))
		return {**sequence_features, **msa_features}

