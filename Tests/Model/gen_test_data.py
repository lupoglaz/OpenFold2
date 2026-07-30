import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
from pathlib import Path
# try:

#Using version 2.1.2
#Go to this repository and use:
#git checkout 1109480e6f38d71b3b265a4a25039e51e2343368
#Also change alphafold/common/residue_constants.py
#L775, L778 np.int -> np.int32
# old_root_folder = str(Path(os.path.dirname(sys.argv[0])).as_posix())
# new_root_folder = str(Path(os.path.dirname(sys.argv[0])).parent.parent.parent/Path('alphafold').as_posix())
# sys.path[0] = new_root_folder

# from alphafold.model.modules import (
# 	Attention, GlobalAttention, MSARowAttentionWithPairBias, MSAColumnAttention, MSAColumnGlobalAttention,
# 	TriangleAttention, TriangleMultiplication, OuterProductMean, Transition,
# 	EvoformerIteration, EmbeddingsAndEvoformer, AlphaFoldIteration, AlphaFold,
# 	pseudo_beta_fn, dgram_from_positions, create_extra_msa_feature)
# from alphafold.model import config
# IMPL = 'af2'
# print('Using original alphafold implementation')
# except:
from colabdesign.af.alphafold.model.modules import (
	Attention, GlobalAttention, MSARowAttentionWithPairBias, MSAColumnAttention, MSAColumnGlobalAttention,
	TriangleAttention, TriangleMultiplication, OuterProductMean, Transition,
	EvoformerIteration, EmbeddingsAndEvoformer, AlphaFoldIteration, AlphaFold,
	pseudo_beta_fn, dgram_from_positions, create_extra_msa_feature)
from colabdesign.af.alphafold.model import config
IMPL = 'cd'
print('Using colabdesign implementation')

def recursive_apply(d, f):
	new_d = {}
	for k,v in d.items():
		if isinstance(v, dict):
			new_d[k] = recursive_apply(v, f)
		elif isinstance(v, jnp.ndarray):
			new_d[k] = f(v)
		else:
			raise ValueError(f'Value is unknown type: {v}')
	return new_d

if __name__=='__main__':
	model_config = config.model_config('model_1')
	model_config.model.global_config.zero_init = False
	rng = jax.random.PRNGKey(42)
	
	batch_size = 4
	N_queries = 64
	N_keys = 64
	q_channels = 32
	m_channels = 32

	N_seq = 64
	N_res = 128
	c_z = 16
	c_m = 32
	
	feat = {
	# 'q_data': jnp.zeros((batch_size, N_queries, q_channels), dtype=jnp.float32),
	# 'm_data': jnp.zeros((batch_size, N_keys, m_channels), dtype=jnp.float32),
	# 'bias': jnp.zeros((batch_size, 1, N_queries, N_keys), dtype=jnp.float32),
	# 'nonbatched_bias': jnp.zeros((N_queries, N_keys), dtype=jnp.float32),
	'q_data': jax.random.normal(rng, (batch_size, N_queries, q_channels), dtype=jnp.float32),
	'm_data': jax.random.normal(rng, (batch_size, N_keys, m_channels), dtype=jnp.float32),
	'bias': jax.random.normal(rng, (batch_size, 1, N_queries, N_keys), dtype=jnp.float32),
	'nonbatched_bias': jax.random.normal(rng, (N_queries, N_keys), dtype=jnp.float32),
	'q_mask': jax.random.bernoulli(rng, 0.5, (batch_size, N_queries, q_channels)),

	'msa_act': jax.random.normal(rng, (N_seq, N_res, c_m), dtype=jnp.float32),
	'msa_mask': jax.random.bernoulli(rng, 0.5, (N_seq, N_res)),
	'pair_act': jax.random.normal(rng, (N_res, N_res, c_z), dtype=jnp.float32),
	'pair_mask': jax.random.bernoulli(rng, 0.5, (N_res, N_res)),

	'seq_act': jax.random.normal(rng, (batch_size, N_res, q_channels), dtype=jnp.float32),
	'seq_mask': jax.random.bernoulli(rng, 0.5, (batch_size, N_res)),
	}
	
	def test_wrapper(filename, _forward_fn):
		apply = hk.transform(_forward_fn).apply
		init = hk.transform(_forward_fn).init
		
		params = init(rng, feat)
		params = hk.data_structures.to_mutable_dict(params)
		
		res = apply(params, rng, feat)
		
		npparams = recursive_apply(params, lambda x: np.array(x))
		npfeat = recursive_apply(feat, lambda x: np.array(x))
		if isinstance(res, jax.Array):
			npres = np.array(res)
		else:
			npres = recursive_apply(res, lambda x: np.array(x))
		res_path = Path('Tests/Data')/Path(f'{filename}.pkl')
		with open(res_path, 'wb') as f:
			pickle.dump((npfeat, npparams, npres), f)

	global_config = model_config.model.global_config
	conf = model_config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	conf.gating = False
	test_wrapper('Attention_no_gate',
		lambda batch:
			Attention(conf, global_config, output_dim=256)
			(q_data=batch['q_data'], m_data=batch['m_data'], bias=batch['bias'], nonbatched_bias=batch['nonbatched_bias'])
		)

	conf.gating = True
	test_wrapper('Attention_gate',
		lambda batch:
			Attention(conf, global_config, output_dim=256)
			(q_data=batch['q_data'], m_data=batch['m_data'], bias=batch['bias'], nonbatched_bias=batch['nonbatched_bias'])
		)

	test_wrapper('GlobalAttention',
		lambda batch:
			GlobalAttention(conf, global_config, output_dim=256)
			(q_data=batch['q_data'], m_data=batch['m_data'], q_mask=batch['q_mask'], bias=batch['bias'])
		)
	if IMPL == 'af2':
		raise Exception('Some problems with the original repo')
		test_wrapper('MSARowAttentionWithPairBias',
			lambda batch:
				MSARowAttentionWithPairBias(conf, global_config)
				(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'], pair_act=batch['pair_act']), is_training=False
			)
		
		conf = model_config.model.embeddings_and_evoformer.evoformer.msa_column_attention
		test_wrapper('MSAColumnAttention',
			lambda batch:
				MSAColumnAttention(conf, global_config)
				(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'], is_training=False)
			)
		
		test_wrapper('MSAColumnGlobalAttention',
			lambda batch:
				MSAColumnGlobalAttention(conf, global_config)
				(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'], is_training=False)
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node
		test_wrapper('TriangleAttention',
			lambda batch:
				TriangleAttention(conf, global_config)
				(pair_act=batch['pair_act'], pair_mask=batch['pair_mask'], is_training=False)
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing
		test_wrapper('TriangleMultiplicationOutgoing',
			lambda batch:
				TriangleMultiplication(conf, global_config)
				(act=batch['pair_act'], mask=batch['pair_mask'], is_training=False)
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming
		test_wrapper('TriangleMultiplicationIncoming',
			lambda batch:
				TriangleMultiplication(conf, global_config)
				(act=batch['pair_act'], mask=batch['pair_mask'], is_training=False)
			)

		feat['msa_act'] = jax.random.normal(rng, (N_seq, N_res, 16), dtype=jnp.float32)
		feat['msa_mask'] = jax.random.bernoulli(rng, 0.5, (N_seq, N_res))
		conf = model_config.model.embeddings_and_evoformer.evoformer.outer_product_mean
		test_wrapper('OuterProductMean',
			lambda batch:
				OuterProductMean(conf, global_config, num_output_channel=256)
				(act=batch['msa_act'], mask=batch['msa_mask'], is_training=False)
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.pair_transition
		test_wrapper('Transition',
			lambda batch:
				Transition(conf, global_config)
				(act=batch['seq_act'], mask=batch['seq_mask'], is_training=False)
			)
	else:
		test_wrapper('MSARowAttentionWithPairBias',
			lambda batch:
				MSARowAttentionWithPairBias(conf, global_config)
				(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'], pair_act=batch['pair_act'])
			)
		
		conf = model_config.model.embeddings_and_evoformer.evoformer.msa_column_attention
		test_wrapper('MSAColumnAttention',
			lambda batch:
				MSAColumnAttention(conf, global_config)
				(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'])
			)
		
		test_wrapper('MSAColumnGlobalAttention',
			lambda batch:
				MSAColumnGlobalAttention(conf, global_config)
				(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'])
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node
		test_wrapper('TriangleAttentionStartingNode',
			lambda batch:
				TriangleAttention(conf, global_config)
				(pair_act=batch['pair_act'], pair_mask=batch['pair_mask'])
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_attention_ending_node
		test_wrapper('TriangleAttentionEndingNode',
			lambda batch:
				TriangleAttention(conf, global_config)
				(pair_act=batch['pair_act'], pair_mask=batch['pair_mask'])
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing
		test_wrapper('TriangleMultiplicationOutgoing',
			lambda batch:
				TriangleMultiplication(conf, global_config)
				(left_act=batch['pair_act'], left_mask=batch['pair_mask'])
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming
		test_wrapper('TriangleMultiplicationIncoming',
			lambda batch:
				TriangleMultiplication(conf, global_config)
				(left_act=batch['pair_act'], left_mask=batch['pair_mask'])
			)

		feat['msa_act'] = jax.random.normal(rng, (N_seq, N_res, 16), dtype=jnp.float32)
		feat['msa_mask'] = jax.random.bernoulli(rng, 0.5, (N_seq, N_res))
		conf = model_config.model.embeddings_and_evoformer.evoformer.outer_product_mean
		test_wrapper('OuterProductMean',
			lambda batch:
				OuterProductMean(conf, global_config, num_output_channel=256)
				(act=batch['msa_act'], mask=batch['msa_mask'])
			)

		conf = model_config.model.embeddings_and_evoformer.evoformer.pair_transition
		test_wrapper('Transition',
			lambda batch:
				Transition(conf, global_config)
				(act=batch['seq_act'], mask=batch['seq_mask'])
			)
		
	sys.exit()
	proc_features_path = Path('/media/lupoglaz/AlphaFold2Output')/Path('T1024')/Path('proc_features.pkl')
	with open(proc_features_path, 'rb') as f:
		feat = pickle.load(f)
	feat = {k: v[0] for k, v in feat.items()}

	num_msa = 300
	num_seq = 400
	num_extra_seq = 100
	
	feat['aatype'] = feat['aatype'][:num_seq]
	feat['residue_index'] = feat['residue_index'][:num_seq]
	feat['seq_length'] = np.int32(num_seq)
	feat['seq_mask'] = feat['seq_mask'][:num_seq]
	
	feat['msa_mask'] = feat['msa_mask'][:num_msa, :num_seq]
	feat['msa_row_mask'] = feat['msa_row_mask'][:num_msa]
	

	feat['template_aatype'] = feat['template_aatype'][:,:num_seq]
	feat['template_all_atom_masks'] = feat['template_all_atom_masks'][:,:num_seq,:]
	feat['template_all_atom_positions'] = feat['template_all_atom_positions'][:,:num_seq,:,:]
	feat['template_pseudo_beta'] = feat['template_pseudo_beta'][:,:num_seq,:]
	feat['template_pseudo_beta_mask'] = feat['template_pseudo_beta_mask'][:,:num_seq]
	feat['atom14_atom_exists'] = feat['atom14_atom_exists'][:num_seq,:]
	feat['residx_atom14_to_atom37'] = feat['residx_atom14_to_atom37'][:num_seq,:]
	feat['residx_atom37_to_atom14'] = feat['residx_atom37_to_atom14'][:num_seq,:]
	feat['atom37_atom_exists'] = feat['atom37_atom_exists'][:num_seq,:]
	
	
	feat['extra_msa'] = feat['extra_msa'][:num_extra_seq, :num_seq]
	feat['extra_msa_mask'] = feat['extra_msa_mask'][:num_extra_seq, :num_seq]
	feat['extra_msa_row_mask'] = feat['extra_msa_row_mask'][:num_extra_seq]
	feat['extra_has_deletion'] = feat['extra_has_deletion'][:num_extra_seq, :num_seq]
	feat['extra_deletion_value'] = feat['extra_deletion_value'][:num_extra_seq, :num_seq]

	feat['bert_mask'] = feat['bert_mask'][:num_msa, :num_seq]
	feat['true_msa'] = feat['true_msa'][:num_msa, :num_seq]
	feat['msa_feat'] = feat['msa_feat'][:num_msa, :num_seq, :]
	feat['target_feat'] = feat['target_feat'][:num_seq, :]

	# for key in feat.keys():
	# 	print(key, feat[key].shape, feat[key].dtype)
	
	conf = model_config.model.embeddings_and_evoformer
	conf.recycle_pos = False
	conf.recycle_features = False
	conf.template.enabled = False
	conf.evoformer_num_block = 1
	conf.extra_msa_stack_num_block = 1
	global_config.deterministic = True
	# test_wrapper('EmbeddingsAndEvoformer',
	# 	lambda batch:EmbeddingsAndEvoformer(conf, global_config)(batch, is_training=False)
	# 	)


	conf = model_config.model
	conf.embeddings_and_evoformer.recycle_pos = False
	conf.embeddings_and_evoformer.recycle_features = False
	conf.embeddings_and_evoformer.template.enabled = False
	conf.embeddings_and_evoformer.evoformer_num_block = 1
	conf.embeddings_and_evoformer.extra_msa_stack_num_block = 1
	conf.num_recycle = 0
	conf.resample_msa_in_recycling = False
	global_config.deterministic = True

	# non_ensembled_batch = feat
	# ensembled_batch = {k: jnp.expand_dims(v, axis=0) for k, v in feat.items()}

	# fwd = lambda x,y: AlphaFoldIteration(conf, global_config)(x, y, is_training=False)
	# apply = hk.transform(fwd).apply
	# init = hk.transform(fwd).init
		
	# params = init(rng, ensembled_batch, non_ensembled_batch)
	# params = hk.data_structures.to_mutable_dict(params)
	
	# res = apply(params, rng, ensembled_batch, non_ensembled_batch)
		
	# res_path = Path('Debug')/Path(f'AlphaFoldIteration.pkl')
	# with open(res_path, 'wb') as f:
	# 	pickle.dump((ensembled_batch, non_ensembled_batch, params, res), f)	

	# batch = {k: jnp.expand_dims(v, axis=0) for k, v in feat.items()}
	
	# fwd = lambda x: AlphaFold(conf)(x, is_training=False)
	# apply = hk.transform(fwd).apply
	# init = hk.transform(fwd).init
		
	# params = init(rng, batch)
	# params = hk.data_structures.to_mutable_dict(params)
	
	# res = apply(params, rng, batch)
		
	# res_path = Path('Debug')/Path(f'AlphaFold.pkl')
	# with open(res_path, 'wb') as f:
	# 	pickle.dump((batch, params, res), f)	