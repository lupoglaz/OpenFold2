import math
import logging 

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class TrainerConfig:
	max_epochs = 10
	batch_size = 128
	learning_rate = 3e-4
	betas = (0.9, 0.95)
	grad_norm_clip = 1.0
	weight_decay = 0.1
	lr_decay = False
	warmup_tokens = 375e6
	final_tokens = 260e9
	ckpt_path = None
	num_workers = 0

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

class Trainer:
	def __init__(self, model, config, device_ids=None):
		self.model = model
		self.config = config

		self.device = 'cpu'
		if torch.cuda.is_available():
			self.device = torch.device(torch.cuda.current_device())
			print(self.device)
			if not (device_ids is None):
				self.model = torch.nn.DataParallel(self.model, device_ids=device_ids).to(self.device)
			else:
				self.model = self.model.to(self.device)

		raw_model = self.model.module if hasattr(self.model, "module") else self.model
		self.optimizer = raw_model.configure_optimizers(config)
		self.tokens = 0

	def save_checkpoint(self):
		raw_model = self.model.module if hasattr(self.model, "module") else self.model
		logging.info(f'Saving {self.config.ckpt_path}')
		torch.save(raw_model.state_dict(), self.config.ckpt_path)

	def load_checkpoint(self):
		raw_model = self.model.module if hasattr(self.model, "module") else self.model
		logging.info(f'Loading {self.config.ckpt_path}')
		checkpoint = torch.load(self.config.ckpt_path)
		raw_model.load_state_dict(checkpoint)

	def step(self, x, y):
		self.model.train()
		x = x.to(self.device)
		y = y.to(self.device)


		logits, loss = self.model(x, y)
		loss = loss.mean()
		self.model.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
		self.optimizer.step()

		if self.config.lr_decay:
			self.tokens += (y>=0).sum()
			if self.tokens < self.config.warmup_tokens:
				lr_mult = float(self.tokens)/float(max(1, self.config.warmup_tokens))
			else:
				progress = float(self.tokens - self.config.warmup_tokens)/float(max(1, self.config.final_tokens - self.config.warmup_tokens))
				lr_mult = max(0.1, 0.5*(1.0 + math.cos(math.pi*progress)))
			lr = self.config.learning_rate * lr_mult
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = lr
		else:
			lr = self.config.learning_rate
		
		return loss.item()
	
	def test(self, x, y):
		self.model.eval()
		x = x.to(self.device)
		y = y.to(self.device)


		logits, loss = self.model(x, y)
		return loss.mean()
	