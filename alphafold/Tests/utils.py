import torch
import numpy as np
from pathlib import Path
import pickle

from pytorch_memlab import MemReporter
from pytorch_memlab.utils import readable_size as mem_to_str
reporter = MemReporter()


def randomize_params(params):
	for key in params.keys():
		print(key)
		for param in params[key].keys():
			print('\t' + param)
			params[key][param] = np.random.rand(*params[key][param].shape) - 0.5
	return params

def convert(arg, device:torch.device=None):
	if isinstance(arg, tuple):
		return tuple([convert(arg_i) for arg_i in arg])
	elif isinstance(arg, list):
		return [convert(arg_i) for arg_i in arg]
	elif isinstance(arg, np.ndarray):
		if device is None:
			return torch.from_numpy(arg)
		else:
			return torch.from_numpy(arg).to(device=device)
	elif isinstance(arg, dict):
		return {k: convert(v) for k, v in arg.items()}
	else:
		return arg

def check_success(this_res, res):
	err = torch.abs(this_res.detach().to(dtype=torch.float32, device='cpu') - res.detach().to(dtype=torch.float32, device='cpu'))
	max_err = torch.max(err).item()
	mean_err = torch.mean(err).item()
	return err.sum().numpy(), max_err, mean_err

def check_recursive(a, b, depth:int=0, key=None, tol_max:float=1e-3, tol_mean=1e-3):
	str_depth = ''.join(['--' for i in range(depth)])
	if isinstance(a, tuple) or isinstance(a, list):
		errs = []
		max_errs = []
		mean_errs = []
		for i, (a_i, b_i) in enumerate(zip(a, b)):
			err_i, max_err_i, mean_err_i = check_recursive(a_i, b_i, depth=depth+1, key=i)
			errs.append(err_i)
			max_errs.append(max_err_i)
			mean_errs.append(mean_err_i)
			succ = (max_err_i<tol_max) and (mean_err_i<tol_mean)
			if succ:
				print(f'{str_depth}>{i}: success = {succ}')
			else:
				print(f'{str_depth}>{i}: success = {succ}:\t{err_i}\t{max_err_i}\t{mean_err_i}')

		return np.sum(errs), max(max_errs), np.mean(mean_errs)
	
	if isinstance(a, dict):
		errs = []
		max_errs = []
		mean_errs = []
		for key in a.keys():
			err_i, max_err_i, mean_err_i = check_recursive(a[key], b[key], depth=depth+1, key=key)
			errs.append(err_i)
			max_errs.append(max_err_i)
			mean_errs.append(mean_err_i)
			succ = (max_err_i<tol_max) and (mean_err_i<tol_mean)
			if succ:
				print(f'{str_depth}>{key}: success = {succ}')
			else:
				print(f'{str_depth}>{key}: success = {succ}:\t{err_i}\t{max_err_i}\t{mean_err_i}')

		return np.sum(errs), max(max_errs), np.mean(mean_errs)
	
	if isinstance(a, np.ndarray):
		a = torch.from_numpy(a)
	
	if isinstance(b, np.ndarray):
		b = torch.from_numpy(b)

	if isinstance(a, float) or isinstance(a, int):
		a = torch.Tensor([a])
	if isinstance(b, float) or isinstance(b, int):
		b = torch.Tensor([b])
	
	err, max_err, mean_err = check_success(a, b)
	succ = (max_err<tol_max) and (mean_err<tol_mean)
	print(f'{str_depth}> success = {succ}:\t{err}\t{max_err}\t{mean_err}')
	return check_success(a, b)

def load_data(args, filename):
	with open(Path(args.debug_dir)/Path(f'{filename}.pkl'), 'rb') as f:
		data = pickle.load(f)
	if len(data) == 4:
		fnargs1, fnargs2, params, res = data
		return convert(fnargs1), convert(fnargs2), params, convert(res)
	if len(data) == 3:
		args, params, res = data
		return convert(args), params, convert(res)
	elif len(data) == 2:
		args, res = data
		return convert(args), res

def get_total_alloc():
	reporter.collect_tensor()
	reporter.get_stats()
	target_device = torch.device('cuda:0')
	total_mem = 0
	total_numel = 0
	for device, tensor_stats in reporter.device_tensor_stat.items():
		if device != target_device:
			continue
		for stat in tensor_stats:
			name, size, numel, mem = stat
			total_mem += mem
			total_numel += numel
	return total_mem