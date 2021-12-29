import torch
from alphafold.Model.affine import QuatAffine
from alphafold.Tests.Model.module_test import load_data
import argparse

from alphafold.Tests.utils import check_recursive, load_data

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

def pre_compose_test(args, name):
	(tensor, update), res = load_data(args, name)
	qa = QuatAffine.from_tensor(tensor)
	this_res = qa.pre_compose(update).to_tensor()
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
	pre_compose_test(args, 'quat_pre_compose')


	
	
