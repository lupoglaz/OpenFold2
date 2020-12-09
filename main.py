import sys
import math
import logging 
from tqdm import tqdm
import numpy as np
import argparse
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader

from src.Model import GPT, GPTConfig
from src.Trainer import Trainer, TrainerConfig
from src.Dataset import CharDataset


def train(model_config, train_config, train_dataset):
	devices = []
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			if torch.cuda.get_device_name(i) != 'GeForce GTX 980':
				print(f'Using device: {i}:{torch.cuda.get_device_name(i)}')
				devices.append(i)
			else:
				print(f'Excluding device: {i}:{torch.cuda.get_device_name(i)}')
	
	model = GPT(model_config)
	trainer = Trainer(model, train_config, device_ids=devices)

	train_stream = DataLoader(  train_dataset, shuffle=True, pin_memory=True, 
								batch_size=train_config.batch_size, 
								num_workers=train_config.num_workers)

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
	text = open('input.txt').read()
	train_dataset = CharDataset(text, block_size)

	model_config = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
							n_layer=8, n_head=8, n_embd=512)

	train_config = TrainerConfig(max_epochs=2, batch_size=256, learning_rate=6e-4, 
							lr_decay=True, warmup_tokens=512*20, 
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