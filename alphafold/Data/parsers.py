import collections
import dataclasses
import re
from typing import Tuple, Sequence, List, Optional, Set
import string

DeletionMatrix = Sequence[Sequence[int]]

@dataclasses.dataclass(frozen=True)
class Msa:
	"""Class representing a parsed MSA file."""
	sequences: Sequence[str]
	deletion_matrix: DeletionMatrix
	descriptions: Sequence[str]

	def __post_init__(self):
		if not (len(self.sequences) == len(self.deletion_matrix) == len(self.descriptions)):
			raise ValueError('All fields for an MSA must have the same length. '
							f'Got {len(self.sequences)} sequences, '
							f'{len(self.deletion_matrix)} rows in the deletion matrix and '
							f'{len(self.descriptions)} descriptions.')
	def __len__(self):
		return len(self.sequences)

	def truncate(self, max_seqs: int):
		return Msa(sequences=self.sequences[:max_seqs], deletion_matrix=self.deletion_matrix[:max_seqs], descriptions=self.descriptions[:max_seqs])

@dataclasses.dataclass(frozen=True)
class TemplateHit:
	index: int
	name: str
	aligned_cols: int
	sum_probs: float
	query: str
	hit_sequence: str
	indices_query: List[int]
	indices_hit: List[int]

# Sequences coming from UniProtKB database come in the
# `db|UniqueIdentifier|EntryName` format, e.g. `tr|A0A146SKV9|A0A146SKV9_FUNHE`
# or `sp|P0C2L1|A3X1_LOXLA` (for TREMBL/Swiss-Prot respectively).
_UNIPROT_PATTERN = re.compile(
	r"""
	^
	# UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
	(?:tr|sp)
	\|
	# A primary accession number of the UniProtKB entry.
	(?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
	# Occasionally there is a _0 or _1 isoform suffix, which we ignore.
	(?:_\d)?
	\|
	# TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
	# protein ID code.
	(?:[A-Za-z0-9]+)
	_
	# A mnemonic species identification code.
	(?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
	# Small BFD uses a final value after an underscore, which we ignore.
	(?:_\d+)?
	$
	""",
	re.VERBOSE)


@dataclasses.dataclass(frozen=True)
class Identifiers:
	species_id: str = ''
	

def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/parsers.py#L68"""
	sequences = []
	descriptions = []
	index = -1
	for line in fasta_string.splitlines():
		line = line.strip()
		if line.startswith('>'):
			index += 1
			descriptions.append(line[1:])
			sequences.append('')
			continue
		elif not line:
			continue
		sequences[index] += line

	return sequences, descriptions

def parse_stockholm(stockholm_str: str) -> Msa:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/parsers.py#L97"""
	name_to_sequence = collections.OrderedDict()
	for line in stockholm_str.splitlines():
		line = line.strip()
		if (not line) or line.startswith(('#', '//')):
			continue
		name, sequence = line.split()
		if not(name in name_to_sequence):
			name_to_sequence[name] = ''
		name_to_sequence[name] += sequence

	msa = []
	deletion_matrix = []
	query = ''
	keep_columns = []
	for seq_index, sequence in enumerate(name_to_sequence.values()):
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
	
	return Msa(sequences=msa, deletion_matrix=deletion_matrix, descriptions=list(name_to_sequence.keys()))

def parse_hhr(hhr_string:str) -> Sequence[TemplateHit]:

	def _parse_hhr_hit(detailed_lines: Sequence[str]) -> TemplateHit:
		
		def _get_hhr_regex_groups(regex_pattern: str, line: str) -> Sequence[Optional[str]]:
			match = re.match(regex_pattern, line)
			if match is None:
				raise RuntimeError(f'Cant parse line {line}')
			return match.groups()
		
		def _update_hhr_residue_indices_list(sequence: str, start_index: int, indices_list: List[int]):
			counter = start_index
			for symbol in sequence:
				if symbol == '-':
					indices_list.append(-1)
				else:
					indices_list.append(counter)
					counter += 1

		number_of_hit = int(detailed_lines[0].split()[-1])
		name_hit = detailed_lines[1][1:]

		pattern = (
			'Probab=(.*)[\t ]*E-value=(.*)[\t ]*Score=(.*)[\t ]*Aligned_cols=(.*)[\t'
      		' ]*Identities=(.*)%[\t ]*Similarity=(.*)[\t ]*Sum_probs=(.*)[\t '
      		']*Template_Neff=(.*)'
		)
		match = re.match(pattern, detailed_lines[2])
		if match is None:
			raise RuntimeError(f'Cant parse section {detailed_lines}.Unexpected: {detailed_lines[2]}')
		(prob_true, e_value, _, aligned_cols, _, _, sum_probs, neff) = [float(x) for x in match.groups()]

		query = ''
		hit_sequence = ''
		indices_query = []
		indices_hit = []
		length_block = None
		for line in detailed_lines[3:]:
			if (line.startswith('Q ') and not line.startswith('Q ss_dssp')) and (not line.startswith('Q ss_pred')) and (not line.startswith('Q consensus')):
				patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)'
				groups = _get_hhr_regex_groups(patt, line[17:])
				start = int(groups[0]) - 1 
				delta_query = groups[1]
				end = int(groups[2])
				num_insertions = len([x for x in delta_query if x == '-'])
				length_block = end - start + num_insertions
				assert length_block == len(delta_query)

				query += delta_query
				_update_hhr_residue_indices_list(delta_query, start, indices_query)
			elif line.startswith('T '):
				if (not line.startswith('T ss_dssp')) and (not line.startswith('T ss_pred')) and (not line.startswith('T consensus')):
					patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*[0-9]* \([0-9]*\)'
					groups = _get_hhr_regex_groups(patt, line[17:])
					start = int(groups[0]) - 1 
					delta_hit_sequence = groups[1]
					assert length_block == len(delta_hit_sequence)
					hit_sequence += delta_hit_sequence
					_update_hhr_residue_indices_list(delta_hit_sequence, start, indices_hit)
		return TemplateHit(index = number_of_hit,
							name = name_hit,
							aligned_cols = int(aligned_cols),
							sum_probs = sum_probs,
							query = query,
							hit_sequence = hit_sequence,
							indices_query = indices_query,
							indices_hit = indices_hit)

	lines = hhr_string.splitlines()
	block_starts = [i for i, line in enumerate(lines) if line.startswith('No ')]
	hits = []
	if block_starts:
		block_starts.append(len(lines))
		for i in range(len(block_starts)-1):
			hits.append(_parse_hhr_hit(lines[block_starts[i]:block_starts[i+1]]))
	return hits


def convert_stockholm_to_a3m(stockholm_format: str, max_sequences: Optional[int]=None) -> str:
	
	def _convert_sto_sequence_to_a3m(query_non_gaps: Sequence[bool], sto_seq: str):
		for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
			if is_query_res_non_gap:
				yield sequence_res
			elif sequence_res != '-':
				yield sequence_res.lower()

	descriptions = collections.OrderedDict()
	sequences = collections.OrderedDict()
	reached_max_sequences = False

	for line in stockholm_format.splitlines():
		reached_max_sequences = max_sequences and len(sequences)>=max_sequences
		if line.strip() and not line.startswith(('#', '//')):
			seqname, aligned_seq = line.split(maxsplit=1)
			if not(seqname in sequences):
				if reached_max_sequences:
					continue
				sequences[seqname] = ''
			sequences[seqname] += aligned_seq

	for line in stockholm_format.splitlines():
		if line[:4]=='#=GS':
			columns = line.split(maxsplit=3)
			seqname, feature = columns[1:3]
			value = columns[3] if len(columns) == 4 else ''
			if feature != 'DE':
				continue
			if reached_max_sequences and (not (seqname in sequences)):
				continue
			descriptions[seqname] = value
			if len(descriptions) == len(sequences):
				break

	a3m_sequences = {}
	query_sequence = next(iter(sequences.values()))
	query_non_gaps = [res!='-' for res in query_sequence]
	for seqname, sto_sequence in sequences.items():
		a3m_sequences[seqname] = ''.join(_convert_sto_sequence_to_a3m(query_non_gaps, sto_sequence))

	fasta_chunks = (f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}" for k in a3m_sequences)
	return '\n'.join(fasta_chunks) + '\n'


def parse_a3m(a3m_string: str) -> Msa:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/parsers.py#L157"""

	sequences, descriptions = parse_fasta(a3m_string)
	deletion_matrix = []
	for msa_sequence in sequences:
		deletion_vec = []
		deletion_count = 0
		for j in msa_sequence:
			if j.islower():
				deletion_count += 1
			else:
				deletion_vec.append(deletion_count)
				deletion_count = 0
		deletion_matrix.append(deletion_vec)

	# Make the MSA matrix out of aligned (deletion-free) sequences.
	deletion_table = str.maketrans('', '', string.ascii_lowercase)
	aligned_sequences = [s.translate(deletion_table) for s in sequences]
	return Msa(sequences=aligned_sequences,
				deletion_matrix=deletion_matrix,
				descriptions=descriptions)


def _keep_line(line: str, seqnames: Set[str]) -> bool:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/parsers.py#L257"""
	if not line.strip():
		return True
	if line.strip() == '//':  # End tag
		return True
	if line.startswith('# STOCKHOLM'):  # Start tag
		return True
	if line.startswith('#=GC RF'):  # Reference Annotation Line
		return True
	if line[:4] == '#=GS':  # Description lines - keep if sequence in list.
		_, seqname, _ = line.split(maxsplit=2)
		return seqname in seqnames
	elif line.startswith('#'):  # Other markup - filter out
		return False
	else:  # Alignment data - keep if sequence in list.
		seqname = line.partition(' ')[0]
		return seqname in seqnames


def truncate_stockholm_msa(stockholm_msa_path: str, max_sequences: int) -> str:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/parsers.py#L277"""
	seqnames = set()
	filtered_lines = []

	with open(stockholm_msa_path) as f:
		for line in f:
			if line.strip() and not line.startswith(('#', '//')):
				# Ignore blank lines, markup and end symbols - remainder are alignment
				# sequence parts.
				seqname = line.partition(' ')[0]
				seqnames.add(seqname)
				if len(seqnames) >= max_sequences:
					break
		f.seek(0)
		for line in f:
			if _keep_line(line, seqnames):
				filtered_lines.append(line)

	return ''.join(filtered_lines)

def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/msa_identifiers.py#L54
	"""
	matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
	if matches:
		return Identifiers(species_id=matches.group('SpeciesIdentifier'))
	return Identifiers()


def _extract_sequence_identifier(description: str) -> Optional[str]:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/msa_identifiers.py#L75"""
	split_description = description.split()
	if split_description:
		return split_description[0].partition('/')[0]
	else:
		return None


def get_identifiers(description: str) -> Identifiers:
	"""https://github.com/deepmind/alphafold/blob/b85ffe10799ca08cc62146f1dabb4e4ee6c0a580/alphafold/data/msa_identifiers.py#L84"""
	sequence_identifier = _extract_sequence_identifier(description)
	if sequence_identifier is None:
		return Identifiers()
	else:
		return _parse_sequence_identifier(sequence_identifier)