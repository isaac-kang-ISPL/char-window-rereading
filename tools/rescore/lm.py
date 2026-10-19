"""ByT5 pseudo-log-likelihood (PLL) of a string — the language-model term of S(y). GPU.

For each position i of y the character is replaced by one sentinel (encoder input: y with y_i → <extra_id_0>, then EOS)
and the decoder reads "<extra_id_0> y_i <extra_id_1>"; the position term is
    log P(y_i | <extra_id_0>) + log P(<extra_id_1> | <extra_id_0> y_i).
'own' keeps the character as written; 'ci' (used by the method) takes the log-sum-exp over the upper- and lower-case
forms of the masked letter; the surrounding context retains its original case. PLL(y) = Σ_i terms. Strings must use
the 94 printable ASCII characters of the recognizer dictionary (an empty string has no PLL).

Strings are grouped by length and batched (≈ 3,072 positions per batch for ByT5-XL). Identical masked encoder inputs
are evaluated once *within that batch*, then gathered into the original decoder order. No hidden-state cache survives
the batch. Decoder requests, both case branches, EOS and the score formula are unchanged. With bf16, changing encoder
batch shape can change numerical scores; this is not a bit-equivalence guarantee. PLL_CONTEXT_REUSE=0 keeps the old
path. `run_shards` puts the new path in a separate *_ctxdedup_v1 directory, preserving old score tables.

    python -m tools.rescore.lm <byt5-xl> <work folder with strings.json, chars94.json> 16 fwd cuda:0
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SENT0, SENT1, PAD, EOS, OFFSET = 258, 257, 0, 1, 3
STALE = 20 * 60
CONTEXT_REUSE_VERSION = 'batch_masked_context_v1'


def context_reuse_enabled():
    value = os.environ.get('PLL_CONTEXT_REUSE', '1')
    if value not in ('0', '1'):
        raise ValueError('PLL_CONTEXT_REUSE must be 0 or 1')
    return value == '1'


def masked_context_indices(texts):
    """Stable unique encoder-row indices and inverse for one equal-length batch.

    A row is (word, masked position). ASCII text cannot contain the sentinel, so
    (position, prefix, suffix) uniquely identifies its exact encoder token row.
    CPU bookkeeping only; no GPU sync and no global/string-to-feature cache.
    """
    if not texts:
        return [], []
    length = len(texts[0])
    if any(len(text) != length for text in texts):
        raise ValueError('masked contexts require equal-length strings')
    unique, first, inverse = {}, [], []
    for text in texts:
        for pos in range(length):
            key = (pos, text[:pos], text[pos + 1:])
            group = unique.get(key)
            if group is None:
                group = len(first)
                unique[key] = group
                first.append(len(inverse))
            inverse.append(group)
    return first, inverse


class PLL:
    """ByT5 PLL scorer — `score(texts)` → {index: (own terms, ci terms)} per position (float32)."""

    def __init__(self, model_name='byt5-xl', device='cuda:0', chars=None, fp32=False, *, dedup_contexts=None):
        self.dedup_contexts = context_reuse_enabled() if dedup_contexts is None else bool(dedup_contexts)
        import torch
        import transformers
        from transformers import T5ForConditionalGeneration
        self.torch, self.device, self.model_name = torch, device, model_name
        kw = {} if int(transformers.__version__.split('.')[0]) >= 5 else dict(low_cpu_mem_usage=True)
        self.model = T5ForConditionalGeneration.from_pretrained(
            f'google/{model_name}', dtype=torch.float32 if fp32 else torch.bfloat16, **kw).to(device).eval()
        self.per_batch = (3072 if model_name == 'byt5-xl' else 1536) // (2 if fp32 else 1)
        self.chars = chars or default_chars()
        self.cix = {c: k for k, c in enumerate(self.chars)}
        self.byte_of = torch.tensor([ord(c) + OFFSET for c in self.chars], device=device)
        self.sw_of = torch.tensor([ord(c.swapcase()) + OFFSET if c.swapcase() in self.cix else ord(c) + OFFSET for c in self.chars],
                                  device=device)
        self.dtype = 'fp32' if fp32 else 'bf16'
        self.transformers = transformers.__version__

    def _score_batch(self, texts):
        """One bounded batch; return CPU scores/counts, never hidden states.

        Keeping all intermediate tensors in this scope also releases them before
        the next encoder call, rather than retaining the previous batch's tensors.
        """
        from transformers.modeling_outputs import BaseModelOutput
        torch, model, device = self.torch, self.model, self.device
        B, L = len(texts), len(texts[0])
        first, inverse = masked_context_indices(texts) if self.dedup_contexts else ([], [])
        reuse = bool(first) and len(first) < B * L
        tok = torch.tensor([[self.cix[c] for c in text] for text in texts], device=device)
        byte = self.byte_of[tok]
        pos = torch.arange(L, device=device)
        enc_in = byte[:, None, :].expand(B, L, L).clone()
        enc_in[:, pos, pos] = SENT0
        enc_in = torch.cat([enc_in, enc_in.new_full((B, L, 1), EOS)], -1).view(B * L, L + 1)
        if reuse:
            enc_in = enc_in.index_select(0, torch.tensor(first, device=device, dtype=torch.long))
        enc = model.encoder(input_ids=enc_in).last_hidden_state
        sw = self.sw_of[tok]
        both = torch.stack([byte, sw], -1).view(B * L, 2)
        dec_in = torch.stack([torch.full_like(both, PAD), torch.full_like(both, SENT0), both], -1).view(B * L * 2, 3)
        if reuse:
            # One gather into the original two-branch decoder order. Do not first
            # expand to B*L and then repeat, which would keep a third feature copy.
            decoder_rows = torch.tensor(inverse, device=device, dtype=torch.long).repeat_interleave(2)
            decoder_memory = enc.index_select(0, decoder_rows)
        else:
            decoder_memory = enc.repeat_interleave(2, 0)
        logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=decoder_memory),
                       decoder_input_ids=dec_in).logits.float()
        lp1 = torch.log_softmax(logits[:, 1], -1).gather(-1, dec_in[:, 2:3])[:, 0]
        lp2 = torch.log_softmax(logits[:, 2], -1)[:, SENT1]
        s = (lp1 + lp2).view(B, L, 2)
        lo = s[..., 0]
        lc = torch.where(sw == byte, lo, torch.logsumexp(s, -1))
        return (lo.cpu().numpy().astype(np.float32), lc.cpu().numpy().astype(np.float32),
                len(first) if reuse else B * L)

    def score(self, texts, idx=None, heartbeat=None):
        """Position terms of texts[i] for i in idx (default: all) — {i: (own, ci)}."""
        torch = self.torch
        idx = range(len(texts)) if idx is None else idx
        by_len = defaultdict(list)
        for i in idx:
            by_len[len(texts[i])].append(i)
        own, ci = {}, {}
        stats = dict(context_reuse=self.dedup_contexts, batches=0, masked_rows=0, encoder_rows=0)
        with torch.inference_mode():
            for L, ids_all in sorted(by_len.items()):
                if L == 0:
                    continue
                step = max(1, self.per_batch // L)
                for b in range(0, len(ids_all), step):
                    ids = ids_all[b:b + step]
                    lo, lc, encoder_rows = self._score_batch([texts[i] for i in ids])
                    stats['batches'] += 1
                    stats['masked_rows'] += len(ids) * L
                    stats['encoder_rows'] += encoder_rows
                    for j, i in enumerate(ids):
                        own[i], ci[i] = lo[j], lc[j]
                if heartbeat:
                    heartbeat()
        self.last_stats = stats
        return {i: (own[i], ci[i]) for i in own}

    def table(self, texts, field='ci'):
        """{string: PLL} for the strings the scorer can read (94 characters, not empty)."""
        ok = [t for t in dict.fromkeys(texts) if t and set(t) <= set(self.cix)]
        terms = self.score(ok)
        k = 1 if field == 'ci' else 0
        return {ok[i]: float(v[k].astype(np.float64).sum()) for i, v in terms.items()}


def default_chars():
    """The 94 printable characters of the recognizer dictionary (strtrain/utils/EN_symbol_dict.txt)."""
    from tools.rescore import store
    return (store.REPO / 'strtrain' / 'utils' / 'EN_symbol_dict.txt').read_text().splitlines()


def run_shards(model_name, work, strings_file, n, order='fwd', device='cuda:0'):
    """Score <work>/<strings_file> in n shards (string i in shard i mod n).

    Destination: <work>/<model>[_fp32][_ctxdedup_v1]/<model>_pll_shard{k}of{n}.npz.
    The context-reuse suffix is enabled by default; PLL_CONTEXT_REUSE=0 selects the legacy folder/kernel.
    Finished shards are skipped; a lock file per shard lets several GPUs share a folder (a lock untouched for 20 minutes
    is taken over); `order` rev starts from the last shard."""
    work, n = Path(work), int(n)
    fp32 = os.environ.get('PLL_DTYPE', 'bf16') == 'fp32'
    dedup = context_reuse_enabled()
    out_dir = work / (model_name + ('_fp32' if fp32 else '') + ('_ctxdedup_v1' if dedup else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = json.loads((work / strings_file).read_text())
    chars = json.loads((work / 'chars94.json').read_text())
    shards = list(range(n)) if order == 'fwd' else list(range(n))[::-1]
    fin = lambda k: out_dir / f'{model_name}_pll_shard{k}of{n}.npz'
    lockf = lambda k: out_dir / f'{model_name}_pll_shard{k}of{n}.lock'
    scorer = None
    for k in shards:
        if fin(k).exists():
            continue
        lk = lockf(k)
        if lk.exists() and time.time() - lk.stat().st_mtime > STALE:
            lk.unlink(missing_ok=True)
        try:
            os.close(os.open(lk, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        if fin(k).exists():
            lk.unlink(missing_ok=True)
            continue
        if scorer is None:
            scorer = PLL(model_name, device, chars, fp32, dedup_contexts=dedup)
        mine = [i for i in range(len(texts)) if i % n == k]
        t0 = time.time()
        terms = scorer.score(texts, mine, heartbeat=lambda: os.utime(lk))
        tmp = out_dir / f'{model_name}_pll_shard{k}of{n}.tmp.npz'
        meta = dict(host=socket.gethostname(), gpu=scorer.torch.cuda.get_device_name(device), torch=scorer.torch.__version__,
                    dtype=scorer.dtype, transformers=scorer.transformers, seconds=round(time.time() - t0, 1),
                    encoder_policy=CONTEXT_REUSE_VERSION if dedup else 'legacy', context_counts=scorer.last_stats)
        np.savez_compressed(tmp, index=np.array(mine), lengths=np.array([len(texts[i]) for i in mine]),
                            own=np.concatenate([terms[i][0] for i in mine]), ci=np.concatenate([terms[i][1] for i in mine]),
                            meta=np.array(json.dumps(meta)))
        tmp.rename(fin(k))
        lk.unlink(missing_ok=True)
        print(f'{model_name} shard {k}/{n}: {len(mine):,} strings · {time.time() - t0:.0f}s · {meta["gpu"]}', flush=True)
    left = [k for k in range(n) if not fin(k).exists()]
    print(f'{model_name}: {len(left)} shards left (held by other processes or servers)', flush=True)


if __name__ == '__main__':
    run_shards(*sys.argv[1:])
