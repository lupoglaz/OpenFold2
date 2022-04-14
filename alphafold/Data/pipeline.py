from pathlib import Path
from typing import Mapping, Sequence, Any, Optional
from alphafold.Data.Tools import HHSearch, HHBlits, Jackhammer
from alphafold.Data import parsers
from alphafold.Common import residue_constants, protein
import numpy as np
import pickle
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

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
				uniref_max_hits: int=10000,
				use_precomputed_msas:bool=True):

		self.use_precomputed_msas = use_precomputed_msas
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
		self.hhsearch_pdb70_runner = HHSearch(
			binary_path = hhsearch_binary_path,
			databases = [pdb70_database_path]
		)
		self.template_featurizer = template_featurizer
		self.mgnify_max_hits = mgnify_max_hits
		self.uniref_max_hits = uniref_max_hits

	def make_sequence_features(self, sequence: str, description: str, num_res: int) ->FeatureDict:
		return {
			'aatype': residue_constants.sequence_to_onehot(sequence=sequence, mapping=residue_constants.restype_order_with_x, map_unknown_to_x=True),
			'between_segment_residues': np.zeros((num_res, ), dtype=np.int32),
			'domain_name': np.array([description.encode('utf-8')], dtype=np.object_),
			'residue_index': np.array(range(num_res), dtype=np.int32),
			'seq_length': np.array([num_res]*num_res, dtype=np.int32),
			'sequence': np.array([sequence.encode('utf-8')], dtype=np.object_)
		}
		

	def make_msa_features(self, msas: Sequence[parsers.Msa]) -> FeatureDict:
		if not msas:
			raise ValueError('DataPipeline: At least one msa is required')
		int_msa = []
		deletion_matrix = []
		species_ids = []
		seen_sequences = set()
		for msa_index, msa in enumerate(msas):
			if not msa:
				raise ValueError(f'DataPipeline: msas index {msa_index} is empty')
			for sequence_index, sequence in enumerate(msa.sequences):
				if sequence in seen_sequences:
					continue
				seen_sequences.add(sequence)
				int_msa.append([residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
				deletion_matrix.append(msa.deletion_matrix[sequence_index])
				
				identifiers = parsers.get_identifiers(msa.descriptions[sequence_index])
				species_ids.append(identifiers.species_id.encode('utf-8'))

		num_res = len(msas[0].sequences[0])
		num_alignments = len(int_msa)
		return {
			'deletion_matrix_int': np.array(deletion_matrix, dtype=np.int32),
			'msa': np.array(int_msa, dtype=np.int32),
			'num_alignments': np.array([num_alignments]*num_res, dtype=np.int32),
			'msa_species_identifiers': np.array(species_ids, dtype=np.object_)
		}
	
	def run_tool(self, msa_runner, input_fasta_path:Path, msa_out_path:Path, msa_format:str='sto', max_sto_seq:int=None):
		if not self.use_precomputed_msas or not msa_out_path.exists():
			if msa_format == 'sto' and (not(max_sto_seq is None)):
				result = msa_runner.query(input_fasta_path, max_sto_seq)[0]
			else:
				result = msa_runner.query(input_fasta_path)[0]
			with open(msa_out_path, 'w') as f:
				f.write(result[msa_format])
		else:
			print(f'Reading MSA from file {msa_out_path}')
			if msa_format == 'sto' and (not(max_sto_seq is None)):
				precomputed_msa = parsers.truncate_stockholm_msa(msa_out_path, max_sto_seq)
				result = {'sto': precomputed_msa}
			else:
				with open(msa_out_path, 'r') as f:
					result = {msa_format: f.read()}

		return result

	def process(self, input_fasta_path: Path, msa_output_dir: Path) -> FeatureDict:
		file_name = input_fasta_path.stem
		with open(input_fasta_path) as f:
			input_fasta_str = f.read()
		input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
		if len(input_seqs) != 1:
			raise ValueError(f'DataPipeline: more than one sequence found {input_fasta_path}')
		input_sequence = input_seqs[0]
		input_description = input_descs[0]
		num_res = len(input_sequence)

		file_name = input_fasta_path.stem
		with open(input_fasta_path) as f:
			input_fasta_str = f.read()
		input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
		if len(input_seqs) != 1:
			raise ValueError(f'DataPipeline: more than one sequence found {input_fasta_path}')

		input_sequence = input_seqs[0]
		input_description = input_descs[0]
		num_res = len(input_sequence)

		jackhmmer_uniref90_result = self.run_tool(	self.jackhmmer_uniref90_runner, 
													input_fasta_path=input_fasta_path, 
													msa_out_path = msa_output_dir / Path(f'uniref90_hits.sto'),
													max_sto_seq=self.uniref_max_hits)
		uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])

		jackhmmer_mgnify_result = self.run_tool(self.jackhmmer_mgnify_runner, 
												input_fasta_path=input_fasta_path, 
												msa_out_path = msa_output_dir / Path(f'mgnify_hits.sto'),
												max_sto_seq=self.mgnify_max_hits)
		mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
		
		# uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(jackhmmer_uniref90_result['sto'], max_sequences=self.uniref_max_hits)
		# hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
		# hhsearch_hits = parsers.parse_hhr(hhsearch_result)
		# mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
		# mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]

		if self._use_small_bfd:
			jackhmmer_small_bfd_results = self.run_tool(self.jackhmmer_small_bfd_runner, 
														input_fasta_path=input_fasta_path, 
														msa_out_path = msa_output_dir / Path(f'small_bfd_hits.sto'))
			bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_results['sto'])
		else:
			raise NotImplementedError()

		sequence_features = self.make_sequence_features(sequence=input_sequence, description=input_description, num_res=num_res)
		msa_features = self.make_msa_features(msas=(uniref90_msa, bfd_msa, mgnify_msa))
		
		feature_dict = {**sequence_features, **msa_features}
		return feature_dict

	def process_pdb(self, pdb_path:Path, fasta_output_dir:Path=None):
		assert pdb_path.exists()

		with open(pdb_path, 'r') as f:
			pdb_string = f.read()
		prot = protein.from_pdb_string(pdb_string)
		feature_dict = {
				"all_atom_positions": prot.atom_positions,
				"all_atom_mask": prot.atom_mask
			}
		
		sequence = ''.join([residue_constants.restypes_with_x[aatype] for aatype in prot.aatype])

		if not(fasta_output_dir is None):
			seq_name = pdb_path.stem.upper()
			fasta_path = fasta_output_dir / Path(f"{pdb_path.stem.lower()}.fasta")
			with open(fasta_path, "w") as f:
				record = SeqRecord(Seq(sequence), id=seq_name)
				SeqIO.write(record, f, "fasta")
			return feature_dict, sequence, fasta_path
		else:
			return feature_dict, sequence

		


