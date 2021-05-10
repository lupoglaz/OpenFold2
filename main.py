import sys
import math
import logging 
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader

from src.Model import SE3TConfig, SE3TransformerIt, MSAConfig
from src.Trainer import Trainer, TrainerConfig
from src.MSADataset import MSADataset, collate

import _pickle as pkl

import matplotlib 
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, CoordsRotate, Coords2Center
from TorchProteinLibrary.RMSD import Coords2RMSD

def plot_coords(coords, num_atoms, axis, **args):
	coords = coords.view(num_atoms.item(), 3)
	x, y, z = coords[:,0], coords[:,1], coords[:,2]
	axis.plot(x, y, z, **args)
	
	ax_min_x, ax_max_x = axis.get_xlim()
	ax_min_y, ax_max_y = axis.get_ylim()
	ax_min_z, ax_max_z = axis.get_zlim()

	#Preserving aspect ratio
	min_x = min(torch.min(x).item(), ax_min_x)
	max_x = max(torch.max(x).item(), ax_max_x)
	min_y = min(torch.min(y).item(), ax_min_y)
	max_y = max(torch.max(y).item(), ax_max_y)
	min_z = min(torch.min(z).item(), ax_min_z)
	max_z = max(torch.max(z).item(), ax_max_z)
	max_L = max([max_x - min_x, max_y - min_y, max_z - min_z])
	axis.set_xlim(min_x, min_x+max_L)
	axis.set_ylim(min_y, min_y+max_L)
	axis.set_zlim(min_z, min_z+max_L)

def align(src, dst, num_atoms):
	translate = CoordsTranslate()
	rotate = CoordsRotate()
	rmsd = Coords2RMSD()
	center = Coords2Center()
	with torch.no_grad():
		rmsd(src, dst, num_atoms)
		center_src = center(src, num_atoms)
		center_dst = center(dst, num_atoms)
		c_src = translate(src, -center_src, num_atoms)
		c_dst = translate(dst, -center_dst, num_atoms)
		rc_src = rotate(c_src, rmsd.UT.transpose(1,2).contiguous(), num_atoms)
	return rc_src, c_dst

def train(model_config, train_config, train_dataset, log_name=Path('log_train.txt')):
	devices = []
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			if torch.cuda.get_device_name(i) != 'GeForce GTX 980':
				print(f'Using device: {i}:{torch.cuda.get_device_name(i)}')
				devices.append(i)
			else:
				print(f'Excluding device: {i}:{torch.cuda.get_device_name(i)}')
	
	torch.cuda.set_device(1)

	msa_config = MSAConfig()
	model = SE3TransformerIt(model_config, msa_config)
	trainer = Trainer(model, train_config, device_ids=None)

	train_stream = DataLoader(  train_dataset, shuffle=True, pin_memory=True, 
								batch_size=train_config.batch_size, 
								num_workers=train_config.num_workers,
								collate_fn=collate)
	with open(log_name, 'w') as fout:
		fout.write('Epoch\tLoss\tStd\n')

	for epoch in range(train_config.max_epochs):
		losses = []
		for msa, x, y, sec, num_sec in tqdm(train_stream):
			loss = trainer.step(msa, x, y, sec, num_sec)
			losses.append(loss)
			
		print(f"Epoch {epoch}, train loss = {np.mean(losses)}")
		with open(log_name, 'a') as fout:
			fout.write(f'{epoch}\t{np.mean(losses)}\t{np.std(losses)}\n')

		trainer.save_checkpoint()

def test(model_config, train_config, test_dataset):
	torch.cuda.set_device(1)
	msa_config = MSAConfig()
	model = SE3TransformerIt(model_config, msa_config)
	trainer = Trainer(model, train_config)
	trainer.load_checkpoint()

	test_stream = DataLoader(  test_dataset, shuffle=False, pin_memory=True, 
								batch_size=train_config.batch_size, 
								num_workers=train_config.num_workers,
								collate_fn=collate)
	losses = []
	for msa, x, y, sec, num_sec in tqdm(test_stream):
		with torch.no_grad():
			loss, prot, num_atoms = trainer.test(msa, x, y)
		losses.append(loss.item())
		
		# fig = plt.figure(figsize=(12,8))
		
		# num_cols = int(math.sqrt(train_config.batch_size))
		# num_rows = int(train_config.batch_size / num_cols) + 1
		# for i in range(num_atoms.size(0)):
		# 	axis = fig.add_subplot(num_cols, num_rows, i+1, projection='3d')
		# 	pred, target = align(prot[i,:num_atoms[i]*3].unsqueeze(dim=0).detach().cpu(), y[i,:num_atoms[i]*3].unsqueeze(dim=0).cpu(), num_atoms[i].unsqueeze(dim=0).cpu())
		# 	print(pred.size(), target.size(), num_atoms[i])
		# 	plot_coords(pred[0,:], num_atoms[i].cpu(), axis, color='blue')
		# 	plot_coords(target[0,:], num_atoms[i].cpu(), axis, color='red')
		
		# plt.tight_layout()	
		# plt.show()
		# sys.exit()
			
		
	print(f"Test loss = {np.mean(losses)}")
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')
		
	args = parser.parse_args()
	
	model_config = SE3TConfig(num_iter = 4)
	
	# torch.autograd.set_detect_anomaly(True)
		
	if args.cmd is None:
		parser.print_help()
		sys.exit()

	elif args.cmd() == 'test':
		
		test_dataset = MSADataset(Path('dataset/test_pre/list.dat'))

		test_config = TrainerConfig(batch_size=6, num_workers=4, ckpt_path = 'checkpoint.th')

		test(model_config, test_config, test_dataset)

	elif args.cmd() == 'train':
		
		train_dataset = MSADataset(Path('dataset/train_pre/list.dat'))

		train_config = TrainerConfig(max_epochs=300, batch_size=5, learning_rate=6e-3, 
									lr_decay=False, warmup_tokens=32*20, 
									final_tokens=2*len(train_dataset)*10, 
									num_workers=4, ckpt_path = 'checkpoint.th')

		train(model_config, train_config, train_dataset, log_name=Path('log_train.txt'))

	else:
		print('wtf')