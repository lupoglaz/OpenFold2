import math
import logging 
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

class GPTConfig:
	embd_pdrop = 0.1
	resid_pdrop = 0.1
	attn_pdrop = 0.1
	t_n_layer = 12
	t_n_head = 12 
	t_n_embd = 768


	def __init__(self, vocab_size, block_size, **kwargs):
		self.vocab_size = vocab_size
		self.block_size = block_size
		for k, v in kwargs.items():
			setattr(self, k, v)


class SelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.t_n_embd % config.t_n_head == 0

		self.key = nn.Linear(config.n_embd, config.t_n_embd)
		self.query = nn.Linear(config.n_embd, config.t_n_embd)
		self.value = nn.Linear(config.n_embd, config.t_n_embd)

		self.attn_drop = nn.Dropout(config.attn_pdrop)
		self.resid_drop = nn.Dropout(config.resid_pdrop)

		self.proj = nn.Linear(config.t_n_embd, config.t_n_embd)

		self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size) )
		self.n_head = config.n_head

	def forward(self, x, layer_past=None):
		B, T, C = x.size()

		k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
		q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
		v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)

		att = (q @ k.transpose(-2, -1)) * (1.0 /math.sqrt(k.size(-1)))
		att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
		att = F.softmax(att, dim=-1)
		att = self.attn_drop(att)
		y = att @ v
		y = y.transpose(1,2).contiguous().view(B, T, C)

		y = self.resid_drop(self.proj(y))
		return y

class TransformerBlock(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.ln1 = nn.LayerNorm(config.n_embd)
		self.ln2 = nn.LayerNorm(config.n_embd)
		self.attn = CasualSelfAttention(config)
		self.mlp = nn.Sequential(
			nn.Linear(config.n_embd, 4*config.n_embd),
			nn.GELU(),
			nn.Linear(4*config.n_embd, config.n_embd),
			nn.Dropout(config.resid_pdrop)
		)
	def forward(self, x):
		x = x + self.attn(self.ln1(x))
		x = x + self.mlp(self.ln1(x))
		return x