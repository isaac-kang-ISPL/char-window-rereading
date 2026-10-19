import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import Block, PatchEmbed
from openrec.modeling.encoders.svtrv2_lnconv import Feat2D, LastStage


class ViT(nn.Module):

    def __init__(
        self,
        img_size=[32, 128],
        patch_size=[4, 8],
        in_channels=3,
        out_channels=256,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        last_stage=False,
        feat2d=False,
        use_cls_token=False,
        pretrained=None,
        pretrained_strict=False,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = embed_dim
        self.use_cls_token = use_cls_token
        self.feat_sz = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        if use_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros([1, 1, embed_dim], dtype=torch.float32),
                requires_grad=True,
            )
            trunc_normal_(self.cls_token, mean=0, std=0.02)
            self.pos_embed = nn.Parameter(
                torch.zeros([1, num_patches + 1, embed_dim],
                            dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros([1, num_patches, embed_dim], dtype=torch.float32),
                requires_grad=True,
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=act_layer,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.last_stage = last_stage
        self.feat2d = feat2d
        if last_stage:
            self.out_channels = out_channels
            self.stages = LastStage(embed_dim, out_channels, last_drop=0.1)
        if feat2d:
            self.stages = Feat2D()
        trunc_normal_(self.pos_embed, mean=0, std=0.02)
        self.apply(self._init_weights)

        # Optional ImageNet-21k pretraining (CSD paper §4.1 / §5.2). The repo's
        # ViT uses timm-identical module names (patch_embed.proj, blocks.i.{norm1,
        # attn.qkv, attn.proj, norm2, mlp.fc1, mlp.fc2}, norm), so a timm
        # `vit_*_patch16_*.augreg_in21k` state_dict loads almost key-for-key.
        # `pretrained` is a path to such a local checkpoint; shape-mismatched keys
        # (e.g. a non-16x16 patch stem) are skipped, and pos_embed is interpolated
        # to this model's grid. Leave it unset to train the encoder from scratch.
        if pretrained:
            self.load_pretrained(pretrained, strict=pretrained_strict)

    def _resize_pos_embed(self, src_pos):
        """Adapt a checkpoint positional embedding [1, S, D] to this model's
        grid, dropping/inserting the cls-token row and bicubic-interpolating the
        patch grid as needed. Returns None if the source cannot be reconciled."""
        D = src_pos.shape[-1]
        if D != self.embed_dim:
            return None
        S = src_pos.shape[1]
        tgt_total = self.patch_embed.num_patches + (1 if self.use_cls_token else 0)
        if S == tgt_total:
            return src_pos
        gh_t, gw_t = self.feat_sz
        # timm ViTs ship a single cls token (n_cls=1); fall back to n_cls=0.
        for n_cls in (1, 0):
            grid_len = S - n_cls
            gh_s = int(round(grid_len ** 0.5))
            if gh_s * gh_s != grid_len:
                continue
            grid = src_pos[:, n_cls:].reshape(1, gh_s, gh_s, D).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(gh_t, gw_t), mode='bicubic',
                                 align_corners=False)
            grid = grid.permute(0, 2, 3, 1).reshape(1, gh_t * gw_t, D)
            if not self.use_cls_token:
                return grid
            cls_pos = src_pos[:, :1] if n_cls == 1 else src_pos.new_zeros(1, 1, D)
            return torch.cat([cls_pos, grid], dim=1)
        return None

    def load_pretrained(self, ckpt, strict=False):
        """Load a (timm-style) ViT checkpoint into this encoder. Only keys whose
        shapes match are loaded; everything else is reported and skipped, so a
        single recipe works across encoder sizes / patch configs."""
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f'ViT(pretrained=...): checkpoint not found: {ckpt}. '
                'Provide a local timm ViT state_dict '
                '(e.g. vit_base_patch16_224.augreg_in21k) or unset `pretrained`.')
        sd = torch.load(ckpt, map_location='cpu')
        for key in ('model', 'state_dict', 'model_ema', 'teacher', 'module'):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        # Strip common wrapper prefixes.
        clean = {}
        for k, v in sd.items():
            for p in ('module.', 'model.', 'encoder.', 'backbone.'):
                if k.startswith(p):
                    k = k[len(p):]
            clean[k] = v
        sd = clean
        # Drop classifier / pre-logits heads not present in this encoder.
        sd = {k: v for k, v in sd.items()
              if not k.startswith(('head.', 'pre_logits.', 'fc_norm.',
                                   'norm_pre.'))}
        if not self.use_cls_token:
            sd.pop('cls_token', None)
        if 'pos_embed' in sd:
            resized = self._resize_pos_embed(sd['pos_embed'])
            if resized is not None and resized.shape == self.pos_embed.shape:
                sd['pos_embed'] = resized
            else:
                sd.pop('pos_embed', None)
        # Keep only matching-shape keys (avoids size-mismatch errors).
        model_sd = self.state_dict()
        filtered, bad_shape = {}, []
        for k, v in sd.items():
            if k in model_sd:
                if tuple(model_sd[k].shape) == tuple(v.shape):
                    filtered[k] = v
                else:
                    bad_shape.append(k)
        missing = [k for k in model_sd if k not in filtered]
        self.load_state_dict(filtered, strict=strict)
        print(f'[ViT] loaded {len(filtered)}/{len(model_sd)} pretrained tensors '
              f'from {ckpt} | skipped(shape): {len(bad_shape)} | '
              f'missing: {len(missing)}')
        if bad_shape:
            print(f'[ViT]   shape-skipped (random-init): {bad_shape[:6]}'
                  f'{" ..." if len(bad_shape) > 6 else ""}')
        if missing:
            print(f'[ViT]   missing (random-init): {missing[:6]}'
                  f'{" ..." if len(missing) > 6 else ""}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.use_cls_token:
            x = torch.concat([self.cls_token.tile([x.shape[0], 1, 1]), x], 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.use_cls_token:
            x = x[:, 1:, :]
        if self.last_stage:
            x, sz = self.stages(x, self.feat_sz)
        if self.feat2d:
            x, sz = self.stages(x, self.feat_sz)
        return x
