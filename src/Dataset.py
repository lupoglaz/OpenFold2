import math
import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
	def __init__(self, data, block_size):
		chars = sorted(list(set(data)))
		data_size, vocab_size = len(data), len(chars)
		print(f'Data: {data_size} char, {vocab_size} dict')
		self.stoi = {ch:i for i, ch in enumerate(chars)}
		self.itos = {i:ch for i, ch in enumerate(chars)}
		self.block_size = block_size
		self.vocab_size = vocab_size
		self.data = data

	def __getitem__(self, idx):
		chunk = self.data[idx:idx+self.block_size+1]
		dix = [self.stoi[ch] for ch in chunk]
		x = torch.tensor(dix[:-1], dtype=torch.long)
		y = torch.tensor(dix[1:], dtype=torch.long)
		return x,y

	def __len__(self):
		return len(self.data) - self.block_size

if __name__ == '__main__':
	block_size = 128
	text = open('input.txt').read()
	train_dataset = CharDataset(text, block_size)
	for i in range(len(train_dataset)):
		x, y = train_dataset[i]
		str_x = ''.join([train_dataset.itos[k] for k in x.numpy()])
		str_y = ''.join([train_dataset.itos[k] for k in y.numpy()])
		print(str_x)
		print('_______________________________')
		print(str_y)
		break