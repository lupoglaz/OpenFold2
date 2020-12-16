import sys
import math
import logging 
from tqdm import tqdm
import numpy as np
import argparse
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader

from src.Model import SE3Transformer, SE3TConfig
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
	
	model = SE3Transformer(model_config)
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

def test(context, model_config, train_config, train_dataset):
	model = GPT(model_config)
	trainer = Trainer(model, train_config)
	trainer.load_checkpoint()

	x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
	y = trainer.sample(x, 2000, temperature=1.0, sample=True)[0]
	return ''.join([train_dataset.itos[int(i)] for i in y])
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')
		
	args = parser.parse_args()

	block_size = 128
	with open('dataset/data.pkl', 'rb') as fin:
		data = pkl.load(fin)
	train_dataset = AtomDataset(data, block_size)
	
	model_config = SE3TConfig(	num_layers=1)

	train_config = TrainerConfig(max_epochs=100, batch_size=64, learning_rate=6e-4, 
							lr_decay=True, warmup_tokens=64*20, 
							final_tokens=2*len(train_dataset)*block_size, 
							num_workers=4, ckpt_path = 'checkpoint.th')
	
	if args.cmd is None:
		parser.print_help()
		sys.exit()
	elif args.cmd() == 'test':
		output = test("Ever more, was there pain!", model_config, train_config, train_dataset)
		print(output)
	elif args.cmd() == 'train':
		train(model_config, train_config, train_dataset)
	else:
		print('wtf')