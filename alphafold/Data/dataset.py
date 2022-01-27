import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from ml_collections import ConfigDict
from typing import Sequence, Callable
from collections import OrderedDict
import pickle

def path_collate(batch):
	batch = zip(*batch)
	return tuple(list(arg) for arg in batch)

class GeneralFileData(Dataset):
	def __init__(self, dir:Path, allowed_suffixes:Sequence[str]=None, data_proc_func:Callable=None):
		if dir is None:
			return
		self.data_proc_func = data_proc_func
		self.dir = dir
		self.data = {}
		self.keys = []
		for path in self.dir.iterdir():
			if not path.is_file():
				continue
			if not(allowed_suffixes is None):
				if not(path.suffix in allowed_suffixes):
					continue
			self.keys.append(path.stem.lower())
			self.data[path.stem.lower()] = (path.as_posix(),)

		print(f'Data dir {self.dir}, num_entries {len(self.keys)}')
		
	def __getitem__(self, index) -> Path:
		key = self.keys[index]
		data = self.data[key]
		return data

	def __len__(self):
		return len(self.keys)

	def __add__(self, other):
		new = GeneralFileData(None)
		new.keys = list(set(self.keys) & set(other.keys))
		new.data = {key: self.data[key] + other.data[key] for key in new.keys}
		return new

def get_stream(data, batch_size:int=1, process_fn:Callable=None):
	if process_fn is None:
		return DataLoader(data, shuffle=False, pin_memory=False, batch_size=batch_size, num_workers=0, collate_fn=path_collate)
	else:
		return DataLoader(	data, shuffle=False, pin_memory=False, batch_size=batch_size, num_workers=0, 
							collate_fn=lambda *a, **kw: process_fn(path_collate(*a, **kw)))

def get_data_stream(data_dir:Path, batch_size:int=1):
	assert data_dir.exists()
	def load_pkl(file_path):
		with open(file_path[0], 'rb') as f:
			return pickle.load(f)
	def dict_collate(data):
		return data[0]
	data = GeneralFileData(data_dir, allowed_suffixes=['.pkl'], data_proc_func=load_pkl)
	return DataLoader(data, shuffle=False, pin_memory=False, batch_size=batch_size, num_workers=0, collate_fn=dict_collate)


class OpenFold2Dataset(Dataset):
	def __init__(self, structures_dir:Path, alignment_dir:Path, config:ConfigDict):
		self.structures_dir = structures_dir
		self.alignment_dir = alignment_dir
		self.config = config
	
	def __getitem__(self, idx):
		pass

	def __len__(self):
		pass

if __name__=='__main__':
	a = GeneralFileData(Path('/media/HDD/AlphaFold2Dataset/Sequences'))
	stream = DataLoader(a, shuffle=False, pin_memory=False, batch_size=1, num_workers=0)
	for fasta_path in stream:
		print(fasta_path)

	b = GeneralFileData(Path('/media/HDD/AlphaFold2Dataset/Structures'))
	stream = DataLoader(a+b, shuffle=False, pin_memory=False, batch_size=1, num_workers=0)
	for fasta_path, pdb_path in stream:
		print(fasta_path, pdb_path)