import torch

from FastFold.Kernel import scale_mask_softmax, scale_mask_bias_softmax, bias_sigmod_ele, bias_dropout_add
from einops import rearrange

def test_scale_mask_bias_softmax():
	batch_size = 16
	N = 32
	M = 46
	num_heads = 4
	softmax = torch.nn.Softmax(dim=-1)


	logits = torch.randn(batch_size, N, num_heads, M, M).cuda()
	mask = torch.bernoulli(torch.empty(batch_size, N, M).uniform_(0, 1)).cuda()
	nonbatched_bias = torch.randn(batch_size, M, M, num_heads).cuda()
	scaling = 0.1

	nonbatched_bias = rearrange(nonbatched_bias, 'b q k h -> b h q k')
	print(logits.size(), mask.size(), nonbatched_bias.size())
	exp_weights = scale_mask_bias_softmax(logits, mask, nonbatched_bias, scaling)
	
	bias = (1e9 * (mask.to(dtype=torch.float32)-1.0))[...,None,None,:]
	ref_weights = softmax(logits * scaling + bias + nonbatched_bias.unsqueeze(1))
	for i in range(batch_size):
		err = torch.sum(torch.abs(exp_weights[i] - ref_weights[i]))
		print(i, err.item())

def test_scale_mask_softmax():
	batch_size = 4
	N = 16
	M = 32
	num_heads = 4
	softmax = torch.nn.Softmax(dim=-1)

	logits = torch.randn(batch_size, N, num_heads, M, M).cuda()
	mask = torch.bernoulli(torch.empty(batch_size, N, M).uniform_(0, 1)).cuda()
	scaling = 0.1
	
	exp_weights = bias_dropout_add(logits, mask, scaling)

	bias = (1e9 * (mask.to(dtype=torch.float32)-1.0))[...,None,None,:]
	ref_weights = softmax(logits * scaling + bias)

	for i in range(batch_size):
		err = torch.sum(torch.abs(exp_weights[i] - ref_weights[i]))
		print(i, err.item())


def test_bias_dropout_add():
	batch_size = 4
	N = 16
	M = 32
	num_heads = 4
	softmax = torch.nn.Softmax(dim=-1)

	logits = torch.randn(batch_size, N, num_heads, M, M).cuda()
	mask = torch.bernoulli(torch.empty(batch_size, N, M).uniform_(0, 1)).cuda()
	scaling = 0.1
	
	exp_weights = scale_mask_softmax(logits, mask, scaling)

	bias = (1e9 * (mask.to(dtype=torch.float32)-1.0))[...,None,None,:]
	ref_weights = softmax(logits * scaling + bias)

	for i in range(batch_size):
		err = torch.sum(torch.abs(exp_weights[i] - ref_weights[i]))
		print(i, err.item())


if __name__=='__main__':
	test_scale_mask_bias_softmax()
	# test_scale_mask_softmax()