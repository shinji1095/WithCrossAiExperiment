from .edgenext_bn_hs import EdgeNeXtBNHS
from timm.models.registry import register_model

"""
-- Main Models
    XX-Small -> 1.3M
    X-Small -> 2.3M
    Small -> 5.6M
"""

def edgenext_xx_small_bn_hs(pretrained=False, **kwargs):
    # 1.33M & 259.53M @ 256 resolution
    # 70.33% Top-1 accuracy
    # For A100: FPS @ BS=1: 219.66 & @ BS=256: 10359.98
    model = EdgeNeXtBNHS(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model