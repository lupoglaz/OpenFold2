import torch
from torch import nn
from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import GSE3Res, GNormSE3, GConvSE3, GMaxPooling, get_basis_and_r

if __name__=='__main__':
    print("SE3 attention")
    num_degrees = 4
    num_features = 4
    fiber_in = Fiber(1, num_features)
    fiber_mid = Fiber(num_degrees, 32)
    fiber_out = Fiber(1, 128)

    model = nn.ModuleList([GSE3Res(fiber_in, fiber_mid),
                            GNormSE3(fiber_mid),
                            GConvSE3(fiber_mid, fiber_mid, self_interaction=True),
                            GMaxPooling()
                        ])
    fc_layer = nn.Linear(128, 1)

    basis, r = get_basis_and_r()