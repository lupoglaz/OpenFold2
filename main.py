import sys
import math
import logging 
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
logger = logging.getLogger(__name__)

from src.MSADataset import MSADataset, collate, DataLoader

if __name__=='__main__':
	import _pickle as pkl
	
	dataset = MSADataset(Path('dataset/test/list.dat'))

	stream = DataLoader(dataset, shuffle=True, pin_memory=True, 
						batch_size=4, num_workers=0, collate_fn=collate)

	for MSA_inp, G_inp, G_tgt, secondary, num_secondary in stream:
		print(MSA_inp)
		print(G_inp)
		print(G_tgt)
		print(secondary)
		print(num_secondary)
		break

	batch_idx = 1
	for sec_idx in range(num_secondary[batch_idx].item()):
		sec = secondary[batch_idx, sec_idx, :]
		sec = sec.view(int(sec.size(0)/3), 3)
		str = ''
		for k in range(sec.size(0)):
			if sec[k,0]:
				str += '1'
			else:
				str += '0'
		print(str)
