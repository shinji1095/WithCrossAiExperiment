# This script were from the following GitHub repository:
# https://github.com/mmaaz60/EdgeNeXt/blob/main/models/edgenext_bn_hs.py
# Original license: MIT License (see repository for details)
# Retrieved on: September 12, 2025


import torch
from torch import nn
from timm.models.layers import trunc_normal_
from .layers import LayerNorm, PositionalEncodingFourier, BlurPoolDW, DownsampleAAMix
from .sdta_encoder import SDTAEncoder
from .conv_encoder import ConvEncoder


class EdgeNeXt(nn.Module):
    """
    Non-BNHS EdgeNeXt.

    Added args:
      - downsample_strides: [stem, s1->s2, s2->s3, s3->s4] (default [4,2,2,2])
      - stage_dilations: per stage (len=4), default all 1
      - use_blurpool_downsample / use_downsample_aa_mix / use_maxpool_downsample (mutual exclusive)
        ※ stem は常に従来の Conv のまま。後段3箇所のみ Pooling 切替を適用。
        ※ 優先順位: aa_mix > maxpool > blurpool > conv
    """
    def __init__(
        self, in_chans=3, num_classes=1000,
        depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
        global_block=[0, 0, 0, 3], global_block_type=['None', 'None', 'None', 'SDTA'],
        drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
        kernel_sizes=[7, 7, 7, 7], heads=[8, 8, 8, 8],
        use_pos_embd_xca=[False, False, False, False],
        use_pos_embd_global=False, d2_scales=[2, 3, 4, 5],
        downsample_strides=[4, 2, 2, 2],
        stage_dilations=[1, 1, 1, 1],
        # NEW:
        use_blurpool_downsample: bool = False,
        use_downsample_aa_mix: bool = False,
        use_maxpool_downsample: bool = False,
        **kwargs
    ):
        super().__init__()
        for g in global_block_type:
            assert g in ['None', 'SDTA']

        assert len(downsample_strides) == 4
        stem_s = int(downsample_strides[0]); assert stem_s in (1, 2, 4)
        for s in downsample_strides[1:]:
            assert s in (1, 2)
        assert len(stage_dilations) == 4 and all(int(d) >= 1 for d in stage_dilations)
        stage_dilations = [int(d) for d in stage_dilations]

        # mode for later downsamples (stem is always conv)
        mode = "conv"
        if use_downsample_aa_mix:
            mode = "blur_mix"
        elif use_maxpool_downsample:
            mode = "max"
        elif use_blurpool_downsample:
            mode = "blur"

        self.num_classes = num_classes
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ---- Downsample layers ----
        self.downsample_layers = nn.ModuleList()

        # stem: ★既存と同じ（Conv の stride で実装）
        stem_kernel = 4 if stem_s == 4 else (2 if stem_s == 2 else 1)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=stem_kernel, stride=stem_s),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # later 3 downsamples: Pooling(+1×1 Conv) or Conv(stride)
        def _down(in_ch: int, out_ch: int, stride: int, mode: str):
            if stride == 1:
                # ダウンサンプル無し：Poolingしない／1x1 ConvでC合わせのみ
                return nn.Sequential(
                    LayerNorm(in_ch, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                )
            # stride==2 のとき Pooling 切替
            if mode == "blur":
                return nn.Sequential(
                    LayerNorm(in_ch, eps=1e-6, data_format="channels_first"),
                    BlurPoolDW(in_ch, stride=2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                )
            if mode == "max":
                return nn.Sequential(
                    LayerNorm(in_ch, eps=1e-6, data_format="channels_first"),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                )
            if mode == "blur_mix":
                return nn.Sequential(
                    LayerNorm(in_ch, eps=1e-6, data_format="channels_first"),
                    DownsampleAAMix(in_ch, stride=2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                )
            # fallback: 旧実装（Conv の stride）
            return nn.Sequential(
                LayerNorm(in_ch, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
            )

        for i in range(3):
            s = int(downsample_strides[i + 1])
            self.downsample_layers.append(_down(dims[i], dims[i + 1], s, mode))

        # ---- Stages ----
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    assert global_block_type[i] == 'SDTA'
                    blocks.append(SDTAEncoder(
                        dim=dims[i],
                        drop_path=dpr[cur + j],
                        expan_ratio=expan_ratio,
                        scales=d2_scales[i],
                        use_pos_emb=use_pos_embd_xca[i],
                        num_heads=heads[i],
                        conv_dilation=stage_dilations[i],
                    ))
                else:
                    blocks.append(ConvEncoder(
                        dim=dims[i],
                        drop_path=dpr[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        expan_ratio=expan_ratio,
                        kernel_size=kernel_sizes[i],
                        dw_dilation=stage_dilations[i],
                    ))
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head_dropout = nn.Dropout(kwargs.get("classifier_dropout", 0.0))

        self.apply(self._init_weights)
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.pos_embd = PositionalEncodingFourier(dim=dims[0]) if use_pos_embd_global else None

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layers[0](x)          # stem (conv)
        x = self.stages[0](x)
        if self.pos_embd:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)      # later downsamples (pooling or conv)
            x = self.stages[i](x)
        return self.norm(x).mean(dim=[-2, -1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(self.head_dropout(x))
        return x
