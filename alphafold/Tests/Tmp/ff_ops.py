import torch
from torch import nn
from einops import rearrange
from FastFold.Kernel import scale_mask_softmax, scale_mask_bias_softmax, bias_sigmod_ele

if __name__=='__main__':
	softmax = nn.Softmax()
	b1 = 64
	h = 8
	n = 128
	d = 4

	q = torch.zeros(b1, h, n, d)
	k = torch.zeros(b1, h, n, d)
	logits = torch.matmul(q, k.transpose(-1,-2))

	logits = torch.matmul(q, k)

	weights = scale_mask_softmax(logits, bias, self.scaling)

	logits = logits + bias
	weights = softmax(logits)