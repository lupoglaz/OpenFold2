import argparse
import subprocess
from pathlib import Path
import pickle
import numpy as np
from ...Data import pipeline
from alphafold.Model import AlphaFold, AlphaFoldFeatures, model_config
import torch
import numpy as np
import matplotlib.pylab as plt

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-model_name', default='model_1', type=str)
	parser.add_argument('-data_dir', default='/media/lupoglaz/AlphaFold2Data', type=str)
		
	args = parser.parse_args()
	
	params = np.load(Path(args.data_dir)/Path('params')/Path(f'params_{args.model_name}.npz'))
	for k in params.keys():
		print(k)