import torch
from alphafold.Model import affine

def torsion_angles_to_frames(aatype:torch.Tensor, backb_to_global:affine.Rigids, torsion_angles_sin_cos:torch.Tensor) -> affine.Rigids:
	assert aatype.ndimension() == 1
	assert backb_to_global.rot.xx.ndimension() == 1
	assert torsion_angles_sin_cos.ndimension() == 3
	assert torsion_angles_sin_cos.size(1) == 7
	assert torsion_angles_sin_cos.size(2) == 2

	print(backb_to_global)
