import torch
from torch import nn
import math
from scipy.stats import truncnorm

def trunc_init(weights, scale:float=1.0, fan='fan_in'):
	"""
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L53
	"""	
	fans = {'fan_in': lambda f_out, f_in: f_in, 
			'fan_out': lambda f_out, f_in: f_out, 
			'fan_avg': lambda f_out, f_in: (f_in+f_out)/2.0 }
	assert fan in (fans.keys())
	f = fans[fan](* weights.shape)
	scale = scale/max(1.0, f)
	a, b = -2.0, 2.0
	std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0.0, scale=1.0)
	samples = truncnorm.rvs(a=a, b=b, loc=0.0, scale=std, size=weights.size(0)*weights.size(1))
	return samples
	


class Linear(nn.Linear):
	"""
	https://github.com/lupoglaz/alphafold/blob/2d53ad87efedcbbda8e67ab3be96af769dbeae7d/alphafold/model/common_modules.py#L20
	and
	https://github.com/aqlaboratory/openfold/blob/e1c7c9e7cf353b068c3df7b8a23803f45dfea75d/openfold/model/primitives.py#L99
	"""
	def __init__(self, num_input:int, num_output:int, use_bias:bool=True, initializer:str='default') -> None:
		super(Linear, self).__init__(num_input, num_output, bias=use_bias)
		if initializer == 'default':
			self.weight.data.copy_(torch.from_numpy(trunc_init(weights=self.weight, scale=1.0)))
		elif initializer == 'relu':
			self.weight.data.copy_(torch.from_numpy(trunc_init(weights=self.weight, scale=2.0)))
		elif initializer == 'glorot':
			nn.init.xavier_uniform_(self.weight, gain=1.0)
		elif initializer == 'gating':
			self.weight.data.fill_(0.0)
			if use_bias:
				self.bias.data.fill_(1.0)
		elif initializer == 'normal':
			nn.init.kaiming_normal_(self.weight, nonlinearity='linear')
		elif initializer == 'final':
			self.weight.data.fill_(0.0)
		else:
			raise ValueError('Unknown init')