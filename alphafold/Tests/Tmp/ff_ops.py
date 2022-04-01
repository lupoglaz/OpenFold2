import torch
from torch import nn
from einops import rearrange
from FastFold import Kernel#scale_mask_softmax, scale_mask_bias_softmax, bias_sigmod_ele

def scale_mask_softmax():
	softmax = nn.Softmax(dim=-1)
	b = 2
	n = 3
	h = 1
	scaling = 1.0

	logits = torch.ones(b, h, n, n).cuda()
	logits[0,:,0,0] = 0
	logits[0,:,1,1] = 0
	logits[1,:,0,0] = 0
	logits[1,:,1,1] = 0
	
	mask = torch.ones(b, n).cuda()
	mask[0,0] = 0
	# mask[0,1] = 0
	
		
	ff_weights = Kernel.scale_mask_softmax(logits.unsqueeze(2), mask, scaling).squeeze(2)
	
	bias = (1e18 * (mask-1.0))[:,None,None,:]
	logits = logits + bias#.unsqueeze(dim=-1).unsqueeze(dim=-2)
	
	# print(logits)
	n_weights = softmax(logits)
	print(torch.abs(ff_weights - n_weights).sum())

if __name__=='__main__':
	scale_mask_softmax()
