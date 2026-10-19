"""CLIP4STR (VL4STR) as a single openrec decoder.

Why a decoder and not an encoder+decoder pair
---------------------------------------------
`BaseRecognizer` is a straight chain — `transform -> encoder -> decoder` — and
CLIP4STR does not split along it. It runs two branches: a visual one that reads
the CLIP image encoder, and a cross-modal one that feeds the visual prediction
back through the CLIP *text* encoder to refine it. Splitting that across
openrec's encoder/decoder boundary would mean threading the text tower and the
intermediate logits through an interface that has no place for them.

So the whole VL4STR module sits here, behind `use_encoder: False`, and only the
model is separate — the framework around it (trainer, LMDB datasets, label
overrides, ISF masks, UB/CB eval, latest.pth + resume, lr scaling, wandb) is
shared with every other model in this repo. That was the point: CLIP4STR was
previously trained by a standalone script that had to reimplement all of it, and
did so incompletely (no latest.pth, no resume).

Returning a loss during training is an existing convention here, not a new one:
`mdiff_decoder` does the same, and `trainer.py` already distinguishes the two
cases by `preds.dim() >= 2`.

Two interfaces line up exactly, which is what makes this cheap:

  label indices   openrec's ARLabelEncode builds `[EOS] + charset + [BOS, PAD]`
                  and emits `[BOS] + text + [EOS] + [PAD]*`. strhub's Tokenizer
                  builds `(EOS,) + charset + (BOS, PAD)` and emits the same
                  layout. No remapping — the tensors are interchangeable as long
                  as the charsets match, which the config enforces.
  normalisation   the dataset now takes `norm_mean`/`norm_std`, so the loader
                  produces OpenAI-CLIP-normalised 224x224 tensors directly.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_EXT = Path(__file__).resolve().parents[3] / 'external' / 'clip4str'


def _import_vl4str():
    """Import VL4STR from the vendored strhub tree.

    Kept lazy and path-scoped: `external/clip4str` is not a package this repo
    installs, and adding it to sys.path at module import time would shadow
    `strhub` for anything else that happens to use that name.
    """
    if str(_EXT) not in sys.path:
        sys.path.insert(0, str(_EXT))
    from strhub.models.vl_str.system import VL4STR
    return VL4STR


class VL4STRDecoder(nn.Module):
    """Wraps CLIP4STR so openrec can train it.

    Args:
        in_channels: unused (there is no openrec encoder in front); accepted
            because BaseRecognizer passes it positionally.
        out_channels: vocabulary size from the charset, injected by
            BaseRecognizer. Must equal len(charset) + 3 for the [E]/[B]/[P]
            specials — asserted against the checkpoint's own charset below.
        clip_pretrained: path to the CLIP backbone (.bin).
        ckpt: optional CLIP4STR checkpoint. `hyper_parameters` are read from it
            either way; `load_weights` decides whether the STR weights come too.
        load_weights: False = CLIP backbone + fresh decoders, which is the
            paper's own recipe (CLIP4STR is single-stage). True = continue from
            a trained CLIP4STR.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 clip_pretrained,
                 ckpt=None,
                 load_weights=False,
                 max_len=25,
                 charset_train=None,
                 coef_lr=None,
                 **kwargs):
        super(VL4STRDecoder, self).__init__()
        VL4STR = _import_vl4str()

        hp = {}
        if ckpt:
            raw = torch.load(ckpt, map_location='cpu', weights_only=False)
            hp = dict(raw.get('hyper_parameters', {}))
        else:
            raw = None

        # The checkpoint records the author's own absolute path; ours is local.
        hp['clip_pretrained'] = clip_pretrained
        hp['max_label_length'] = max_len
        if charset_train is not None:
            hp['charset_train'] = charset_train
        # Read by param_group_lr_scale() below, not by VL4STR's forward pass.
        # Defaults to the checkpoint's own value (19.0 for the released
        # CLIP4STR), then to 1.0 if there is no checkpoint to read.
        if coef_lr is not None:
            hp['coef_lr'] = coef_lr
        hp.setdefault('coef_lr', 1.0)
        # Inference charset only affects VL4STR's internal decode helpers, which
        # openrec does not use — it runs ARLabelDecode on the logits instead.
        hp.setdefault('charset_test', hp.get('charset_train', ''))
        hp.update({k: v for k, v in kwargs.items() if k in hp or k.startswith('cross_')})

        # openrec sizes out_channels as len(charset) + 3 ([E] + charset + [B] +
        # [P]) because ARLabelDecode indexes that way. VL4STR emits one logit
        # less than that pair: len(charset) + 1, dropping [B] and [P], which are
        # inputs to the decoder and never prediction targets.
        #
        # The two still line up, because the shared prefix is identical:
        # index 0 = [E], 1..len(charset) = the charset. argmax over 95 logits can
        # only return indices ARLabelDecode already understands, so no remapping
        # is needed — the model simply never predicts [B] or [P], which is right.
        # What must match is the charset itself.
        n_charset = len(hp['charset_train'])
        if out_channels != n_charset + 3:
            raise ValueError(
                f'out_channels={out_channels} but the decoder charset has '
                f'{n_charset} characters (expected {n_charset + 3} = charset + '
                f'[E][B][P]). The yml character_dict_path and the checkpoint '
                f'charset must be the same set, in the same order.')
        self.n_logits = n_charset + 1

        self.model = VL4STR(**hp)
        if load_weights and raw is not None:
            missing, unexpected = self.model.load_state_dict(raw['state_dict'],
                                                             strict=False)
            print(f'[VL4STRDecoder] warm start: {len(missing)} missing / '
                  f'{len(unexpected)} unexpected')
        self.max_len = max_len
        self.out_channels = out_channels

    @torch.jit.ignore
    def no_weight_decay(self):
        """Names that must not take weight decay, beyond the 1-D ones.

        openrec already exempts every parameter with `ndim <= 1` (norms, biases),
        which covers most of CLIP4STR's own exclusion list. These three are the
        remainder: they are multi-dimensional but are still positional or
        scaling parameters, and upstream exempts them by name.
        """
        return {'logit_scale', 'pos_queries', 'text_embed'}

    @torch.jit.ignore
    def param_group_lr_scale(self):
        """Run the decoders at `coef_lr` times the CLIP encoder's lr.

        This is CLIP4STR's own recipe, and the reason it needs saying here:
        upstream applies it in `configure_optimizers`, a PyTorch Lightning hook
        that this repo's trainer never calls. Without it every parameter trains
        at the encoder rate, and the decoders — which start from random
        initialisation, unlike the pretrained CLIP backbone — are left at 1/19
        of the intended lr.

        The split matches upstream exactly: everything outside `clip_model` is
        decoder. Returns {} at coef_lr=1.0 so the optimizer keeps its plain
        two-group path.
        """
        coef_lr = getattr(self.model, 'coef_lr', 1.0)
        if coef_lr == 1.0:
            return {}
        return {
            name: coef_lr
            for name, _ in self.named_parameters() if 'clip_model' not in name
        }

    def forward(self, x, data=None):
        """x: image batch (B,3,H,W). data[0]: label indices when training."""
        # VL4STR builds helper tensors (bos, attention masks) on `self._device`,
        # which LightningModule only sets when a PL Trainer owns the model. Here
        # openrec owns it, so that attribute stays 'cpu' and every such tensor
        # lands on the wrong device. Track the real one from the input.
        self.model._device = x.device
        if self.training and data is not None:
            # VL4STR.training_step computes the permutation-averaged loss itself
            # (perm_num permutations, plus the cross-modal term), so hand the
            # scalar straight back — same contract as mdiff_decoder.
            return self.model.training_step((x, data[0]), 0)
        return self.model.forward(x)
