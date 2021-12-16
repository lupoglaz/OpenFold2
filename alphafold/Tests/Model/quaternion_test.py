import torch
from alphafold.Model.affine import QuatAffine
from alphafold.Tests.Model.module_test import load_data
import pickle
from pathlib import Path
import argparse
import numpy as np

def convert(arg):
	if isinstance(arg, tuple):
		return tuple([convert(arg_i) for arg_i in arg])
	elif isinstance(arg, list):
		return [convert(arg_i) for arg_i in arg]
	elif isinstance(arg, np.ndarray):
		return torch.from_numpy(arg)
	elif isinstance(arg, dict):
		return {k: convert(v) for k, v in arg.items()}
	else:
		raise(NotImplementedError())

def load_data(args, filename):
	with open(Path(args.debug_dir)/Path(f'{filename}.pkl'), 'rb') as f:
		fnargs, res = pickle.load(f)
	torch_args = convert(fnargs)
	return convert(fnargs), res

def check_success(this_res, res):
	err = torch.abs(this_res.detach().to(dtype=torch.float32) - res.to(dtype=torch.float32))
	max_err = torch.max(err).item()
	mean_err = torch.mean(err).item()
	return err.sum(), max_err, mean_err

def check_recursive(a, b, depth:int=0, key=None):
	if isinstance(a, tuple) or isinstance(a, list):
		errs = []
		max_errs = []
		mean_errs = []
		for i, (a_i, b_i) in enumerate(zip(a, b)):
			err_i, max_err_i, mean_err_i = check_recursive(a_i, b_i, depth=depth+1, key=i)
			errs.append(err_i)
			max_errs.append(max_err_i)
			mean_errs.append(mean_err_i)
		
		print(i, np.sum(errs), max(max_errs), np.mean(mean_errs))
		return np.sum(errs), max(max_errs), np.mean(mean_errs)
	
	if isinstance(a, dict):
		errs = []
		max_errs = []
		mean_errs = []
		for key in zip(a.keys()):
			err_i, max_err_i, mean_err_i = check_recursive(a[key], b[key], depth=depth+1, key=key)
			errs.append(err_i)
			max_errs.append(max_err_i)
			mean_errs.append(mean_err_i)

		print(key, np.sum(errs), max(max_errs), np.mean(mean_errs))
		return np.sum(errs), max(max_errs), np.mean(mean_errs)
	
	if isinstance(a, np.ndarray):
		a = torch.from_numpy(a)
	
	if isinstance(b, np.ndarray):
		b = torch.from_numpy(b)
	
	return check_success(a, b)

def init_test(args, name):
	args, res = load_data(args, name)
	qa = QuatAffine.from_tensor(*args)
	this_res = qa.quaternion, qa.translation, qa.rotation
	for a,b in zip(res, this_res):
		err, max_err, mean_err = check_recursive(a,b)
		print(f'Max error = {max_err}, mean error = {mean_err} total error = {err}')
		print(f'Success = {(max_err < 1e-4) and (mean_err < 1e-5)}')

def scale_translation_test(args, name):
	(tensor, scale), res = load_data(args, name)
	qa = QuatAffine.from_tensor(tensor)
	qa = qa.scale_translation(scale)
	this_res = qa.quaternion, qa.translation, qa.rotation
	for a,b in zip(res, this_res):
		err, max_err, mean_err = check_recursive(a,b)
		print(f'Max error = {max_err}, mean error = {mean_err} total error = {err}')
		print(f'Success = {(max_err < 1e-4) and (mean_err < 1e-5)}')

def apply_rot_func_test(args, name):
	(tensor, ), res = load_data(args, name)
	qa = QuatAffine.from_tensor(tensor)
	qa = qa.apply_rotation_tensor_fn(lambda t: t+1.0)
	this_res = qa.quaternion, qa.translation, qa.rotation
	for a,b in zip(res, this_res):
		err, max_err, mean_err = check_recursive(a,b)
		print(f'Max error = {max_err}, mean error = {mean_err} total error = {err}')
		print(f'Success = {(max_err < 1e-4) and (mean_err < 1e-5)}')

def to_tensor_test(args, name):
	(tensor, ), res = load_data(args, name)
	qa = QuatAffine.from_tensor(tensor)
	qa = qa.apply_rotation_tensor_fn(lambda t: t+1.0)
	this_res = qa.to_tensor()
	err, max_err, mean_err = check_recursive(res,this_res)
	print(f'Max error = {max_err}, mean error = {mean_err} total error = {err}')
	print(f'Success = {(max_err < 1e-4) and (mean_err < 1e-5)}')

def to_tensor_test(args, name):
	(tensor, ), res = load_data(args, name)
	qa = QuatAffine.from_tensor(tensor)
	qa = qa.apply_rotation_tensor_fn(lambda t: t+1.0)
	this_res = qa.to_tensor()
	err, max_err, mean_err = check_recursive(res,this_res)
	print(f'Max error = {max_err}, mean error = {mean_err} total error = {err}')
	print(f'Success = {(max_err < 1e-4) and (mean_err < 1e-5)}')

def apply_to_point_test(args, name):
	(tensor, point), res = load_data(args, name)
	qa = QuatAffine.from_tensor(tensor)
	this_res = qa.apply_to_point(point)
	err, max_err, mean_err = check_recursive(res, this_res)
	print(f'Max error = {max_err}, mean error = {mean_err} total error = {err}')
	print(f'Success = {(max_err < 1e-4) and (mean_err < 1e-5)}')

def invert_point_test(args, name):
	(tensor, point), res = load_data(args, name)
	qa = QuatAffine.from_tensor(tensor)
	this_res = qa.invert_point(point)
	err, max_err, mean_err = check_recursive(res, this_res)
	print(f'Max error = {max_err}, mean error = {mean_err} total error = {err}')
	print(f'Success = {(max_err < 1e-4) and (mean_err < 1e-5)}')


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
	args = parser.parse_args()
	
	# init_test(args, 'quat_init')
	# scale_translation_test(args, 'quat_scale_translation')
	# apply_rot_func_test(args, 'quat_apply_rotation_func')
	# to_tensor_test(args, 'quat_to_tensor')
	# apply_to_point_test(args, 'quat_apply_to_point')
	# invert_point_test(args, 'quat_invert_point')


	
	
