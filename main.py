import sys
import math
import logging 
from tqdm import tqdm
import numpy as np
import argparse
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader

from src.Model import SE3Transformer, SE3TConfig, SE3TransformerIt
from src.Trainer import Trainer, TrainerConfig
from src.Dataset import AtomDataset, collate

import _pickle as pkl

def train(model_config, train_config, train_dataset):
	devices = []
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			if torch.cuda.get_device_name(i) != 'GeForce GTX 980':
				print(f'Using device: {i}:{torch.cuda.get_device_name(i)}')
				devices.append(i)
			else:
				print(f'Excluding device: {i}:{torch.cuda.get_device_name(i)}')
	
	# model = SE3Transformer(model_config)
	model = SE3TransformerIt(model_config)
	trainer = Trainer(model, train_config, device_ids=None)

	train_stream = DataLoader(  train_dataset, shuffle=True, pin_memory=True, 
								batch_size=train_config.batch_size, 
								num_workers=train_config.num_workers,
								collate_fn=collate)

	for epoch in range(train_config.max_epochs):
		losses = []
		for x,y in tqdm(train_stream):
			loss = trainer.step(x,y)
			losses.append(loss)
			
		print(f"Epoch {epoch}, train loss = {np.mean(losses)}")
		trainer.save_checkpoint()

def test(model_config, train_config, test_dataset):
	# model = SE3Transformer(model_config)
	model = SE3TransformerIt(model_config)
	trainer = Trainer(model, train_config)
	trainer.load_checkpoint()

	test_stream = DataLoader(  test_dataset, shuffle=False, pin_memory=True, 
								batch_size=train_config.batch_size, 
								num_workers=train_config.num_workers,
								collate_fn=collate)
	losses = []
	for x,y in tqdm(test_stream):
		loss = trainer.step(x,y)
		losses.append(loss)
		
	print(f"Test loss = {np.mean(losses)}")
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')
		
	args = parser.parse_args()
	
	model_config = SE3TConfig(num_layers = 1, num_degrees = 3, edge_dim = 2, div = 1, n_heads = 4, num_iter = 4)
	# torch.autograd.set_detect_anomaly(True)
		
	if args.cmd is None:
		parser.print_help()
		sys.exit()

	elif args.cmd() == 'test':
		block_size = 10
		with open('dataset/data_test.pkl', 'rb') as fin:
			data = pkl.load(fin)
		test_dataset = AtomDataset(data, block_size, tgt='dist')

		test_config = TrainerConfig(batch_size=32, num_workers=4, ckpt_path = 'checkpoint.th')

		test(model_config, test_config, test_dataset)

	elif args.cmd() == 'train':
		block_size = 10
		with open('dataset/data_train.pkl', 'rb') as fin:
			data = pkl.load(fin)
		train_dataset = AtomDataset(data, block_size, tgt='dist')

		train_config = TrainerConfig(max_epochs=100, batch_size=32, learning_rate=6e-3, 
									lr_decay=False, warmup_tokens=64*20, 
									final_tokens=2*len(train_dataset)*block_size, 
									num_workers=4, ckpt_path = 'checkpoint.th')

		train(model_config, train_config, train_dataset)

	else:
		print('wtf')