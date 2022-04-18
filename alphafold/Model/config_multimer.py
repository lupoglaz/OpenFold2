import copy
import ml_collections

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'


MODEL_PRESETS = {
    'monomer': (
        'model_1',
        'model_2',
        'model_3',
        'model_4',
        'model_5',
    ),
    'monomer_ptm': (
        'model_1_ptm',
        'model_2_ptm',
        'model_3_ptm',
        'model_4_ptm',
        'model_5_ptm',
    ),
    'multimer': (
        'model_1_multimer_v2',
        'model_2_multimer_v2',
        'model_3_multimer_v2',
        'model_4_multimer_v2',
        'model_5_multimer_v2',
    ),
}
MODEL_PRESETS['monomer_casp14'] = MODEL_PRESETS['monomer']

CONFIG_MULTIMER = ml_collections.ConfigDict({
    'model': {
        'embeddings_and_evoformer': {
            'evoformer_num_block': 48,
            'evoformer': {
                'msa_column_attention': {
                    'dropout_rate': 0.0,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'msa_row_attention_with_pair_bias': {
                    'dropout_rate': 0.15,
                    'gating': True,
                    'num_head': 8,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'msa_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'outer_product_mean': {
                    'chunk_size': 128,
                    'dropout_rate': 0.0,
                    'first': True,
                    'num_outer_channel': 32,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'pair_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_attention_ending_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'triangle_attention_starting_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_multiplication_incoming': {
                    'dropout_rate': 0.25,
                    'equation': 'kjc,kic->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True
                },
                'triangle_multiplication_outgoing': {
                    'dropout_rate': 0.25,
                    'equation': 'ikc,jkc->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True
                }
            },
            'extra_msa_channel': 64,
            'extra_msa_stack_num_block': 4,
            'num_msa': 252,
            'num_extra_msa': 1152,
            'masked_msa': {
                'profile_prob': 0.1,
                'replace_fraction': 0.15,
                'same_prob': 0.1,
                'uniform_prob': 0.1
            },
            'use_chain_relative': True,
            'max_relative_chain': 2,
            'max_relative_idx': 32,
            'seq_channel': 384,
            'msa_channel': 256,
            'pair_channel': 128,
            'prev_pos': {
                'max_bin': 20.75,
                'min_bin': 3.25,
                'num_bins': 15
            },
            'recycle_features': True,
            'recycle_pos': True,
            'template': {
                'attention': {
                    'gating': False,
                    'num_head': 4
                },
                'dgram_features': {
                    'max_bin': 50.75,
                    'min_bin': 3.25,
                    'num_bins': 39
                },
                'enabled': True,
                'max_templates': 4,
                'num_channels': 64,
                'subbatch_size': 128,
                'template_pair_stack': {
                    'num_block': 2,
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 2,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'triangle_attention_ending_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'num_head': 4,
                        'orientation': 'per_column',
                        'shared_dropout': True
                    },
                    'triangle_attention_starting_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'num_head': 4,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'triangle_multiplication_incoming': {
                        'dropout_rate': 0.25,
                        'equation': 'kjc,kic->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    },
                    'triangle_multiplication_outgoing': {
                        'dropout_rate': 0.25,
                        'equation': 'ikc,jkc->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True
                    }
                }
            },
        },
        'global_config': {
            'deterministic': False,
            'multimer_mode': True,
            'subbatch_size': 4,
            'use_remat': False,
            'zero_init': True
        },
        'heads': {
            'distogram': {
                'first_break': 2.3125,
                'last_break': 21.6875,
                'num_bins': 64,
                'weight': 0.3
            },
            'experimentally_resolved': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'weight': 0.01
            },
            'masked_msa': {
                'weight': 2.0
            },
            'predicted_aligned_error': {
                'filter_by_resolution': True,
                'max_error_bin': 31.0,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'num_bins': 64,
                'num_channels': 128,
                'weight': 0.1
            },
            'predicted_lddt': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'num_bins': 50,
                'num_channels': 128,
                'weight': 0.01
            },
            'structure_module': {
                'angle_norm_weight': 0.01,
                'chi_weight': 0.5,
                'clash_overlap_tolerance': 1.5,
                'dropout': 0.1,
                'interface_fape': {
                    'atom_clamp_distance': 1000.0,
                    'loss_unit_distance': 20.0
                },
                'intra_chain_fape': {
                    'atom_clamp_distance': 10.0,
                    'loss_unit_distance': 10.0
                },
                'num_channel': 384,
                'num_head': 12,
                'num_layer': 8,
                'num_layer_in_transition': 3,
                'num_point_qk': 4,
                'num_point_v': 8,
                'num_scalar_qk': 16,
                'num_scalar_v': 16,
                'position_scale': 20.0,
                'sidechain': {
                    'atom_clamp_distance': 10.0,
                    'loss_unit_distance': 10.0,
                    'num_channel': 128,
                    'num_residual_block': 2,
                    'weight_frac': 0.5
                },
                'structural_violation_loss_weight': 1.0,
                'violation_tolerance_factor': 12.0,
                'weight': 1.0
            }
        },
        'num_ensemble_eval': 1,
        'num_recycle': 3,
        'resample_msa_in_recycling': True
    }
})