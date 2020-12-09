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

	def __init__(self, vocab_size, block_size, **kwargs):
		self.vocab_size = vocab_size
		self.block_size = block_size
		for k, v in kwargs.items():
			setattr(self, k, v)

class GPT1Config(GPTConfig):
	n_layer = 12
	n_head = 12 
	n_embd = 768

class CasualSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0

		self.key = nn.Linear(config.n_embd, config.n_embd)
		self.query = nn.Linear(config.n_embd, config.n_embd)
		self.value = nn.Linear(config.n_embd, config.n_embd)

		self.attn_drop = nn.Dropout(config.attn_pdrop)
		self.resid_drop = nn.Dropout(config.resid_pdrop)

		self.proj = nn.Linear(config.n_embd, config.n_embd)

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

class Block(nn.Module):
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

class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.tok_embd = nn.Embedding(config.vocab_size, config.n_embd)
		self.pos_embd = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
		self.drop = nn.Dropout(config.embd_pdrop)

		self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

		self.ln_f = nn.LayerNorm(config.n_embd)
		self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		self.block_size = config.block_size
		self.apply(self.__init_weights)

		logger.info(f'Number of parameters: {sum(p.numel() for p in self.parameters())}')

	def get_block_size(self):
		return self.block_size

	def __init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and (not (module.bias is None)):
				module.bias.data.zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def configure_optimizers(self, train_config):
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear, )
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
		for mn, m in self.named_modules():
			for pn, p in m.named_parameters():
				fpn = f"{mn}.{pn}" if mn else pn
				if pn.endswith('bias'):
					no_decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					no_decay.add(fpn)
		no_decay.add('pos_embd')

		param_dict = {pn: p for pn, p in self.named_parameters()}
		inter_params = decay & no_decay
		union_params = decay | no_decay
		assert len(inter_params) == 0, f"Parameters {str(inter_params)} made into two dicts"
		assert len(union_params - param_dict.keys()) == 0, f"Parameters {str(param_dict.keys() - union_params)} were ommited"

		optim_groups = [
			{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
			{"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
		]
		optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer

	def forward(self, idx, targets=None):
		b, t = idx.size()
		assert t <= self.block_size, "Block size exhausted"

		token_embeddings = self.tok_embd(idx)
		position_embeddings = self.pos_embd[:, :t, :]
		x = self.drop(token_embeddings + position_embeddings)
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.head(x)

		loss = None
		if not (targets is None):
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

		return logits, loss 
