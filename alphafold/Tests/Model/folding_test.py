from typing import Dict
import torch
import argparse
from pathlib import Path
import pickle
import numpy as np
from alphafold.Model import model_config
import numpy as np

from alphafold.Model.msa import Attention, MSARowAttentionWithPairBias, MSAColumnAttention, GlobalAttention, MSAColumnGlobalAttention
from alphafold.Model.spatial import TriangleAttention, TriangleMultiplication, OuterProductMean, Transition
from alphafold.Model.alphafold import EvoformerIteration, EmbeddingsAndEvoformer
from alphafold.Model.embedders import ExtraMSAEmbedding

from alphafold.Tests.Model.module_test import load_data, check_success

def InvariantPointAttentionTest(args, config, global_config):
	feat, params, res = load_data(args, 'EvoformerIteration2')
	conf = config.model.embeddings_and_evoformer.evoformer
	
	attn = EvoformerIteration(	conf, global_config, 
								msa_dim=feat['msa_act'].shape[-1], pair_dim=feat['pair_act'].shape[-1], is_extra_msa=True)
	attn.load_weights_from_af2(params, rel_path='evoformer_iteration')

	activations = {'msa': feat['msa_act'], 'pair': feat['pair_act']}
	masks = {'msa': feat['msa_mask'], 'pair': feat['pair_mask']}
	
	this_res = attn(activations, masks, is_training=False)
	for key in activations.keys():
		print(key)
		check_success(this_res[key], res[key])

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-debug_dir', default='/home/lupoglaz/Projects/alphafold/Debug', type=str)
		
	args = parser.parse_args()
	config = model_config('model_1')
	global_config = config.model.global_config