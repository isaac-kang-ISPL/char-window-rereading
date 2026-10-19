import torch
from torch import nn

from openrec.modeling.decoders import build_decoder
from openrec.modeling.encoders import build_encoder
from openrec.modeling.transforms import build_transform

__all__ = ['BaseRecognizer']


class BaseRecognizer(nn.Module):

    def __init__(self, config):
        """the module for OCR.

        args:
            config (dict): the super parameters for module.
        """
        super(BaseRecognizer, self).__init__()
        in_channels = config.get('in_channels', 3)
        self.use_wd = config.get('use_wd', True)
        # build transfrom,
        # for rec, transfrom can be TPS,None
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone
        if 'Encoder' not in config or config['Encoder'] is None:
            self.use_encoder = False
        else:
            self.use_encoder = True
            config['Encoder']['in_channels'] = in_channels
            self.encoder = build_encoder(config['Encoder'])
            in_channels = self.encoder.out_channels

        # build decoder
        if 'Decoder' not in config or config['Decoder'] is None:
            self.use_decoder = False
        else:
            self.use_decoder = True
            config['Decoder']['in_channels'] = in_channels
            self.decoder = build_decoder(config['Decoder'])

    @torch.jit.ignore
    def no_weight_decay(self):
        if not self.use_wd:
            return set()
        # Accumulate into a set of our own: submodules return sets, and the
        # empty case used to start from a dict, which then raised on update().
        # Copying also keeps a submodule's own set from being mutated here.
        # A model can legitimately have no encoder — CLIP4STR keeps its whole
        # two-branch network in the decoder — so `self.encoder` may not exist,
        # and hasattr() on it would raise rather than return False.
        names = set()
        if self.use_encoder and hasattr(self.encoder, 'no_weight_decay'):
            names |= set(self.encoder.no_weight_decay())
        if self.use_decoder and hasattr(self.decoder, 'no_weight_decay'):
            names |= set(self.decoder.no_weight_decay())
        return names

    @torch.jit.ignore
    def param_group_lr_scale(self):
        """Per-parameter lr multipliers, keyed by the name this model exposes.

        Optional, and mirrors `no_weight_decay` — a submodule that does not
        define it simply contributes nothing, so models that train at a single
        lr are untouched. A model needs this when one part of it is pretrained
        and another starts from scratch: CLIP4STR runs its decoders at
        `coef_lr` (19x) the rate of the CLIP encoder it is built on.
        """
        scales = {}
        if self.use_decoder and hasattr(self.decoder, 'param_group_lr_scale'):
            scales.update({
                'decoder.' + name: scale
                for name, scale in self.decoder.param_group_lr_scale().items()
            })
        return scales

    def forward(self, x, data=None):
        if self.use_transform:
            x = self.transform(x)
        if self.use_encoder:
            x = self.encoder(x)
        if self.use_decoder:
            x = self.decoder(x, data=data)
        return x
