from typing import Mapping, Sequence, Any, Optional
import numpy as np

	
restypes = [
	'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
	'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


HHBLITS_AA_TO_ID = {
	'A': 0,
	'B': 2,
	'C': 1,
	'D': 2,
	'E': 3,
	'F': 4,
	'G': 5,
	'H': 6,
	'I': 7,
	'J': 20,
	'K': 8,
	'L': 9,
	'M': 10,
	'N': 11,
	'O': 20,
	'P': 12,
	'Q': 13,
	'R': 14,
	'S': 15,
	'T': 16,
	'U': 1,
	'V': 17,
	'W': 18,
	'X': 20,
	'Y': 19,
	'Z': 3,
	'-': 21,
}
# Partial inversion of HHBLITS_AA_TO_ID.
ID_TO_HHBLITS_AA = {
    0: 'A',
    1: 'C',  # Also U.
    2: 'D',  # Also B.
    3: 'E',  # Also Z.
    4: 'F',
    5: 'G',
    6: 'H',
    7: 'I',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'V',
    18: 'W',
    19: 'Y',
    20: 'X',  # Includes J and O.
    21: '-',
}

restypes_with_x_and_gap = restypes + ['X', '-']
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i])
    for i in range(len(restypes_with_x_and_gap)))

def sequence_to_onehot(sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool=False) -> np.ndarray:
	num_entries = max(mapping.values()) + 1
	if sorted(set(mapping.values())) != list(range(num_entries)):
		raise ValueError(f'ResiduesConstants: Mapping must contain contiguous sequence of indexes, instead: {mapping.values()}')
	one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)
	for aa_index, aa_type in enumerate(sequence):
		if map_unknown_to_x:
			if aa_type.isalpha() and aa_type.isupper():
				aa_id = mapping.get(aa_type, mapping['X'])
			else:
				raise ValueError(f'ResiduesConstants: Unknown symbol in sequence {aa_type}')
		else:
			aa_id = mapping[aa_type]
		one_hot_arr[aa_index, aa_id] = 1
	return one_hot_arr