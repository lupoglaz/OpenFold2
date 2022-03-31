import torch
from torch import nn
from einops import rearrange
from FastFold.Kernel import scale_mask_softmax, scale_mask_bias_softmax, bias_sigmod_ele

if __name__=='__main__':
	softmax = nn.Softmax(dim=-1)
	b1 = 2
	h = 1
	n = 128
	d = 5
	scaling = 1.0

	logits = torch.ones(b1, d).cuda()
	logits[0,:4] = 0
	logits[1,:2] = 0
	# print(logits)
	mask = torch.ones(b1, d).cuda()
	mask[0,:3] = 0
	mask[1,:2] = 0
		
	weights = scale_mask_softmax(logits, mask, scaling)

	print(weights[0,:])


	bias = (1e9 * (mask-1.0))#[:,None,None,:]
	logits = logits + bias
	# print(logits)
	weights = softmax(logits)
	print(weights[0,:])
