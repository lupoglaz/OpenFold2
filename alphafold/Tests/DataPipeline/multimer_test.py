import argparse
import subprocess
from pathlib import Path
import pickle
import numpy as np
import torch
import numpy as np
import matplotlib.pylab as plt

from alphafold.Tests.utils import check_recursive, convert
from alphafold.Model.data_transforms_multimer import make_msa_profile, sample_msa, make_masked_msa, \
	nearest_neighbor_clusters, create_msa_feat, create_extra_msa_feature
from alphafold.Model.config_multimer import CONFIG_MULTIMER
from torch.distributions.gumbel import Gumbel


def test_gumbel():
	print([Gumbel(torch.zeros(1), torch.ones(1)).sample().item() for i in range(10)])
	g = Gumbel(torch.zeros(1), torch.ones(1))
	print([g.sample().item() for i in range(10)])
	

def test_sample_msa(args):
	with open(Path(args.test_dir)/Path('sample_msa.pkl'), 'rb') as f:
		input, output = pickle.load(f)
	input = convert(input)
	for key in input.keys():
		print(key, input[key].shape)
	this_output = sample_msa(input, 128)
	for key in this_output.keys():
		if not(key in output.keys()):
			print(f'{key} not found')
		else:
			assert output[key].shape == this_output[key].shape
	# check_recursive(output, this_output)

def test_make_msa_profile(args):
	with open(Path(args.test_dir)/Path('make_msa_profile.pkl'), 'rb') as f:
		input, output = pickle.load(f)
	input = convert(input)
	for key in input.keys():
		print(key, input[key].shape)
	
	print(output.shape)
	this_output = make_msa_profile(input)

	check_recursive(output, this_output)

def test_make_msa_profile(args):
	with open(Path(args.test_dir)/Path('make_msa_profile.pkl'), 'rb') as f:
		input, output = pickle.load(f)
	input = convert(input)
	for key in input.keys():
		print(key, input[key].shape)
	
	print(output.shape)
	this_output = make_msa_profile(input)

	check_recursive(output, this_output)

def test_nearest_neighbor_clusters(args):
	with open(Path(args.test_dir)/Path('nearest_neighbor_clusters.pkl'), 'rb') as f:
		input, output = pickle.load(f)
	input = convert(input)
	for key in input.keys():
		print(key, input[key].shape)
	
	this_output = nearest_neighbor_clusters(input)
	
	check_recursive(output, this_output)

def test_create_msa_feat(args):
	with open(Path(args.test_dir)/Path('create_msa_feat.pkl'), 'rb') as f:
		input, output = pickle.load(f)
	input = convert(input)
	for key in input.keys():
		print(key, input[key].shape)
	
	this_output = create_msa_feat(input)
	
	check_recursive(output, this_output)

def test_create_extra_msa_feature(args):
	with open(Path(args.test_dir)/Path('create_extra_msa_feature.pkl'), 'rb') as f:
		input, output = pickle.load(f)
	input = convert(input)
	for key in input.keys():
		print(key, input[key].shape)
	
	this_output = create_extra_msa_feature(input, 128)
	
	check_recursive(output, this_output)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-test_dir', default='/home/lupoglaz/Projects/alphafold21/Tests', type=str)
	args = parser.parse_args()
	
	# test_gumbel()
	# test_sample_msa(args)
	# test_make_msa_profile(args)
	# test_make_masked_msa(args)
	# test_nearest_neighbor_clusters(args)
	# test_create_msa_feat(args)
	test_create_extra_msa_feature(args)