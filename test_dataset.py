
import _pickle as pkl

import torch
from src.Dataset import AtomDataset, collate
from torch.utils.data import DataLoader


if __name__=='__main__':
	with open('dataset/data.pkl', 'rb') as fin:
		data = pkl.load(fin)
	
	dataset = AtomDataset(data, block_size=100)
	train_stream = DataLoader(  dataset, shuffle=False, pin_memory=True, 
								batch_size=1, 
								num_workers=0, collate_fn=collate)
	for batch in train_stream:
		print(batch)
		break