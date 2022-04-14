import dataclasses
import json
import copy
import tempfile
import contextlib
from pathlib import Path

from typing import Mapping, Sequence, Any, Optional
from alphafold.Data.Tools import HHSearch, HHBlits, Jackhammer
from alphafold.Data import parsers, pipeline, msa_pairing
from alphafold.Common import residue_constants, protein
import numpy as np

@dataclasses.dataclass(frozen=True)
class _FastaChain:
	sequence: str
	description: str


def _make_chain_id_map(sequences: Sequence[str], descriptions: Sequence[str]) -> Mapping[str, _FastaChain]:
	"""Makes a mapping from PDB-format chain ID to sequence and description."""
	if len(sequences) != len(descriptions):
		raise ValueError(f'Sequences and descriptions must have equal length. Got {len(sequences)} != {len(descriptions)}.')
	if len(sequences) > protein.MAX_CHAINS:
		raise ValueError(f'Cannot process more chains than the PDB format supports. Got {len(sequences)} chains.')
	chain_id_map = {}
	for chain_id, sequence, description in zip(protein.CHAIN_IDS, sequences, descriptions):
		chain_id_map[chain_id] = _FastaChain(sequence=sequence, description=description)
	return chain_id_map

@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
	with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
		fasta_file.write(fasta_str)
		fasta_file.seek(0)
		yield Path(fasta_file.name)

class DataPipeline:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/pipeline_multimer.py#L170"""
	def __init__(self, 	monomer_data_pipeline: pipeline.DataPipeline,
						jackhammer_binary_path: Path,
						uniprot_database_path: Path,
						uniprot_max_hits: int=50000,
						use_precomputed_msas:bool=True):

		self.monomer_data_pipeline = monomer_data_pipeline
		self.use_precomputed_msas = use_precomputed_msas
		self.jackhmmer_uniprot_runner = Jackhammer(
			binary_path = jackhammer_binary_path,
			database_path = uniprot_database_path
		)
		self.uniprot_max_hits = uniprot_max_hits

	def all_seq_msa_features(self, input_fasta_path:Path, msa_output_dir:Path):
		out_path = msa_output_dir/Path('uniprot_hits.sto')
		result = self.monomer_data_pipeline.run_tool(msa_runner=self.jackhmmer_uniprot_runner,
													input_fasta_path=input_fasta_path,
													msa_out_path=out_path)
		msa = parsers.parse_stockholm(result['sto'])
		msa = msa.truncate(max_seqs=self.uniprot_max_hits)
		all_seq_features = self.monomer_data_pipeline.make_msa_features([msa])
		valid_feats = msa_pairing.MSA_FEATURES + ('msa_species_identifiers',)
		feats = {f'{k}_all_seq': v for k, v in all_seq_features.items() if k in valid_feats}
		return feats

	def process_single_chain(self, chain_id:str, sequence:str, description:str, msa_output_dir:Path, is_homomer:bool):
		chain_fasta_str = f'>chain_{chain_id}\n{sequence}\n'
		chain_msa_output_dir = msa_output_dir/Path(chain_id)
		if not chain_msa_output_dir.exists():
			chain_msa_output_dir.mkdir()
		
		with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
			print(f'Running monomer pipeline on chain {chain_id}: {description}')
			chain_features = self.monomer_data_pipeline.process(input_fasta_path=chain_fasta_path, msa_output_dir=chain_msa_output_dir)
			# We only construct the pairing features if there are 2 or more unique sequences.
			if not is_homomer:
				all_seq_msa_features = self.all_seq_msa_features(chain_fasta_path, chain_msa_output_dir)
			chain_features.update(all_seq_msa_features)
		return chain_features

	def convert_monomer_features(self, monomer_features: pipeline.FeatureDict, chain_id: str) -> pipeline.FeatureDict:
		converted = {}
		converted['auth_chain_id'] = np.asarray(chain_id, dtype=np.object_)
		unnecessary_leading_dim_feats = {'sequence', 'domain_name', 'num_alignments', 'seq_length'}
		for feature_name, feature in monomer_features.items():
			if feature_name in unnecessary_leading_dim_feats:
				# asarray ensures it's a np.ndarray.
				feature = np.asarray(feature[0], dtype=feature.dtype)
			elif feature_name == 'aatype':
				# The multimer model performs the one-hot operation itself.
				feature = np.argmax(feature, axis=-1).astype(np.int32)
			elif feature_name == 'template_aatype':
				raise NotImplemented("No templates allowed")
			elif feature_name == 'template_all_atom_masks':
				raise NotImplemented("No templates allowed")
			converted[feature_name] = feature
		return converted

	def process(self, input_fasta_path: Path, msa_output_dir: Path):
		with open(input_fasta_path) as f:
			input_fasta_str = f.read()

		input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
		
		chain_id_map = _make_chain_id_map(sequences=input_seqs, descriptions=input_descs)
		chain_id_map_path = msa_output_dir/Path('chain_id_map.json')
		with open(chain_id_map_path, 'w') as f:
			chain_id_map_dict = {chain_id: dataclasses.asdict(fasta_chain) for chain_id, fasta_chain in chain_id_map.items()}
			json.dump(chain_id_map_dict, f, indent=4, sort_keys=True)
		
		all_chain_features = {}
		sequence_features = {}
		is_homomer = len(set(input_seqs)) == 1
		for chain_id, fasta_chain in chain_id_map.items():
			if fasta_chain.sequence in sequence_features:
				all_chain_features[chain_id] = copy.deepcopy(sequence_features[fasta_chain.sequence])
				continue
			chain_features = self.process_single_chain(chain_id=chain_id,
														sequence=fasta_chain.sequence,
														description=fasta_chain.description,
														msa_output_dir=msa_output_dir,
														is_homomer=is_homomer)
			print(chain_features.keys())
			chain_features = self.convert_monomer_features(chain_features, chain_id=chain_id)
			print(chain_features.keys())
			all_chain_features[chain_id] = chain_features
			sequence_features[fasta_chain.sequence] = chain_features