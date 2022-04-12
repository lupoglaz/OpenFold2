import ml_collections
import copy
from alphafold.Model.config import CONFIG

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'

def model_config(name: str) -> ml_collections.ConfigDict:
	"""Get the ConfigDict of a CASP14 model."""

	if name not in CONFIG_DIFFS:
		raise ValueError(f'Invalid model name {name}.')
	# cfg = copy.deepcopy(CUSTOM_CONFIG)
	cfg = copy.deepcopy(CONFIG)
	cfg.update_from_flattened_dict(CONFIG_DIFFS[name])
	return cfg

CONFIG_DIFFS = {
	'model_tiny': { #fits to 1080 with fp32 and runs with latency ~1.0
		'data.eval.crop_size': 128,
		'data.common.max_extra_msa': 256,
		'data.common.num_recycle': 0, #this and next one shoould be the same
		'model.num_recycle': 0, #this shoould be the same as the previous one
		'data.common.resample_msa_in_recycling': False,
		'data.common.use_templates': False,
		'model.embeddings_and_evoformer.evoformer_num_block': 4, #default 48
		'model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias.num_head': 4, #default 8
		'model.embeddings_and_evoformer.evoformer.msa_column_attention.num_head': 4, #default 8
		'model.embeddings_and_evoformer.evoformer.outer_product_mean.num_outer_channel': 16, #default 32
		'model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node.num_head': 2, #default 4
		'model.embeddings_and_evoformer.evoformer.triangle_attention_ending_node.num_head': 2, #default 4
		'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.num_intermediate_channel': 32, #default 128
		'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.num_intermediate_channel': 32, #default 128
		'model.embeddings_and_evoformer.extra_msa_channel': 32, #default 64
		'model.embeddings_and_evoformer.extra_msa_stack_num_block': 2, #default 4
		'model.embeddings_and_evoformer.msa_channel': 64, #default 256
		'model.embeddings_and_evoformer.pair_channel': 64, #default 128
		'model.heads.predicted_aligned_error.filter_by_resolution': False,
		'model.heads.experimentally_resolved.filter_by_resolution': False,
		'model.heads.predicted_lddt.filter_by_resolution': False,
		'model.heads.structure_module.structural_violation_loss_weight': 0.0
	},
	'model_small': { #fits to 1080 with fp16 with native amp (weight in fp32)
		'data.eval.crop_size': 128,
		'data.common.max_extra_msa': 256,
		'data.common.num_recycle': 3, #this and next one shoould be the same
		'model.num_recycle': 3, #this shoould be the same as the previous one
		'data.common.resample_msa_in_recycling': False,
		'data.common.use_templates': False,
		'model.embeddings_and_evoformer.evoformer_num_block': 32, #default 48
		'model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias.num_head': 8, #default 8
		'model.embeddings_and_evoformer.evoformer.msa_column_attention.num_head': 8, #default 8
		'model.embeddings_and_evoformer.evoformer.outer_product_mean.num_outer_channel': 32, #default 32
		'model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node.num_head': 4, #default 4
		'model.embeddings_and_evoformer.evoformer.triangle_attention_ending_node.num_head': 4, #default 4
		'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.num_intermediate_channel': 64, #default 128
		'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.num_intermediate_channel': 64, #default 128
		'model.embeddings_and_evoformer.extra_msa_channel': 32, #default 64
		'model.embeddings_and_evoformer.extra_msa_stack_num_block': 2, #default 4
		'model.embeddings_and_evoformer.msa_channel': 128, #default 256
		'model.embeddings_and_evoformer.pair_channel': 128, #default 128
		'model.heads.predicted_aligned_error.filter_by_resolution': False,
		'model.heads.experimentally_resolved.filter_by_resolution': False,
		'model.heads.predicted_lddt.filter_by_resolution': False,
		'model.heads.structure_module.structural_violation_loss_weight': 0.0
	},
	'model_big': { #fits to 1080 with fp16 with deepspeed amp (weight in fp16)
		'data.eval.crop_size': 256,
		'data.common.max_extra_msa': 1024,
		'data.common.num_recycle': 3, #this and next one shoould be the same
		'model.num_recycle': 3, #this shoould be the same as the previous one
		'data.common.resample_msa_in_recycling': False,
		'data.common.use_templates': False,
		'model.embeddings_and_evoformer.evoformer_num_block': 48, #default 48
		'model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias.num_head': 8, #default 8
		'model.embeddings_and_evoformer.evoformer.msa_column_attention.num_head': 8, #default 8
		'model.embeddings_and_evoformer.evoformer.outer_product_mean.num_outer_channel': 32, #default 32
		'model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node.num_head': 4, #default 4
		'model.embeddings_and_evoformer.evoformer.triangle_attention_ending_node.num_head': 4, #default 4
		'model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.num_intermediate_channel': 128, #default 128
		'model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.num_intermediate_channel': 128, #default 128
		'model.embeddings_and_evoformer.extra_msa_channel': 64, #default 64
		'model.embeddings_and_evoformer.extra_msa_stack_num_block': 4, #default 4
		'model.embeddings_and_evoformer.msa_channel': 128, #default 256
		'model.embeddings_and_evoformer.pair_channel': 128, #default 128
		'model.heads.predicted_aligned_error.filter_by_resolution': False,
		'model.heads.experimentally_resolved.filter_by_resolution': False,
		'model.heads.predicted_lddt.filter_by_resolution': False,
		'model.heads.structure_module.structural_violation_loss_weight': 0.0
	}
}