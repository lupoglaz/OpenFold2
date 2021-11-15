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
		one_hot_arr[aa_index, aa_type] = 1
	return one_hot_arr