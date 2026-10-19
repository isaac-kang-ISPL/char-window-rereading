"""SVTRv2-AR beam search — the greedy transcript a and the hypothesis list of each image. GPU.

The image takes the evaluation path of the public weights: an image taller than 1.5 × its width is turned 90°
counter-clockwise, the width-to-height ratio is rounded into 1 … 4 and the image resized (BICUBIC) to 64×64, 96×48,
112×40 or 128×32. The encoder runs once per image; the decoder runs
  greedy   the deployed `forward_test` path, with its log P (EOS included)
  beam     width 10; every live hypothesis is extended by its 10 most probable characters and EOS; the search score is
           the raw cumulative log P (EOS included); it stops once 10 finished hypotheses outscore the best live one.
Hypotheses with the same text keep the higher score; the list is ordered by log P. When the beam pruned a, a is
appended with its greedy score (`greedy_forced`). `candidates()` turns the lists into the input of `hypotheses`.
Both decoders keep the self-attention keys / values of the prefix and the cross-attention keys / values of the encoder
output and decode one character per step (`kv`); the recognizer's float32 matrix products run in TF32 on Ampere or newer
GPUs (`tf32`, `precision`). Defaults: beam 10 · 10, cached keys / values, TF32 — the
setting of the manuscript's results. The first manuscript version (v1) used beam 20 · 20, the whole prefix re-run at every
step (as the public `forward_test` · `forward_train`) and fp32 matmuls — `PAPER` (`--paper`, or LEGIT_AR_KV=0
LEGIT_AR_TF32=0 with --width 20 --token-topk 20).
The cached path is the same arithmetic with other matrix shapes and TF32 rounds the products, so values can differ in the
last digits; on a GPU without TF32 (compute capability below 8.0) the products stay fp32, with a warning, and the run
records the precision actually used.

    python -m tools.rescore.nbest --pool <tsv: dataset, sample_idx, gt> --out <file.jsonl> --device cuda:0
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE_SHAPE = ((64, 64), (96, 48), (112, 40), (128, 32))
ROT_RATIO = 1.5
TOP = 10                                                  # hypotheses kept as candidates
WIDTH = 10                                                # beam width (manuscript v1 runs: 20)
TOKEN_TOPK = 10                                           # characters expanded per live hypothesis (manuscript v1 runs: 20)
KV_ENV = 'LEGIT_AR_KV'                                    # '1' (default) cached keys / values · '0' whole prefix per step
TF32_ENV = 'LEGIT_AR_TF32'                                # '1' (default) TF32 matmuls in the recognizer · '0' fp32
PAPER = dict(width=20, token_topk=20, kv=False, tf32=False)   # recognizer setting of the first manuscript version (v1)
_WARNED = set()


def _flag(name):
    value = os.environ.get(name, '1')
    if value not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return value == '1'


def kv_default():
    return _flag(KV_ENV)


def tf32_default():
    return _flag(TF32_ENV)


def matmul_tf32(device, tf32=None):
    """TF32 actually used for the recognizer's float32 matmuls on device: requested (None → `tf32_default()`) and a CUDA
    device of compute capability 8.0 or higher. A request on an older GPU warns once and stays fp32."""
    import torch
    want = tf32_default() if tf32 is None else bool(tf32)
    device = torch.device(device)
    if not want or device.type != 'cuda':
        return False
    if torch.cuda.get_device_capability(device) >= (8, 0):
        return True
    name = torch.cuda.get_device_name(device)
    if name not in _WARNED:
        _WARNED.add(name)
        import warnings
        warnings.warn(f'{name} has no TF32 (compute capability below 8.0): the recognizer runs in fp32, which is the '
                      f'fp32 setting and not the TF32 default; set {TF32_ENV}=0 to request fp32 explicitly', stacklevel=2)
    return False


@contextlib.contextmanager
def precision(device, tf32=None):
    """Run the recognizer under `matmul_tf32(device, tf32)` (torch.backends.cuda.matmul.allow_tf32), restored after.
    cuDNN keeps its own flag (torch default: TF32 convolutions, as in every stored run)."""
    import torch
    flags = torch.backends.cuda.matmul
    before = flags.allow_tf32
    flags.allow_tf32 = matmul_tf32(device, tf32)
    try:
        yield flags.allow_tf32
    finally:
        flags.allow_tf32 = before


def setting(width=None, token_topk=None, kv=None, tf32=None, device=None):
    """The recognizer setting of a run, defaults resolved — recorded in run protocols and beam metadata."""
    out = dict(beam_width=WIDTH if width is None else int(width), token_topk=TOKEN_TOPK if token_topk is None else int(token_topk),
               kv_cache=kv_default() if kv is None else bool(kv), tf32_requested=tf32_default() if tf32 is None else bool(tf32))
    if device is not None:
        out['matmul_tf32'] = matmul_tf32(device, out['tf32_requested'])
    return out


def _cross_kv(decoder, memory):
    """Cross-attention keys / values of the encoder output, per decoder layer — (B, heads, N, head_dim) each."""
    out = []
    for layer in decoder.decoder:
        ca = layer.cross_attn
        B, N = memory.shape[:2]
        k, v = ca.kv(memory).reshape(B, N, 2, ca.num_heads, ca.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        out.append((k, v))
    return out


def _step(decoder, tokens, pos, cache, cross):
    """One decoder step for rows of the same prefix length: tokens (R,) at position pos, cache = per layer (K, V) of the
    positions before (R, heads, pos, head_dim) or None at pos 0, cross = per layer (K, V) row-aligned.
    → (logits (R, classes), cache with this position appended, last-layer cross-attention (R, heads, N))."""
    import torch
    x = decoder.embedding(tokens[:, None])
    pe = decoder.positional_encoding
    x = pe.dropout(x + pe.pe[:, pos:pos + 1, :])
    new, attn_last = [], None
    for li, layer in enumerate(decoder.decoder):
        sa = layer.self_attn
        R = x.shape[0]
        q, k, v = sa.qkv(x).reshape(R, 1, 3, sa.num_heads, sa.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if cache is not None:
            k = torch.cat([cache[li][0], k], dim=2)
            v = torch.cat([cache[li][1], v], dim=2)
        new.append((k, v))
        a = torch.softmax(q.matmul(k.transpose(2, 3)) * sa.scale, dim=-1)
        y = sa.out_proj(a.matmul(v).transpose(1, 2).reshape(R, 1, sa.embed_dim))
        x = layer.norm1(x + layer.dropout1(y))
        ca = layer.cross_attn
        q = ca.q(x).reshape(R, 1, ca.num_heads, ca.head_dim).transpose(1, 2)
        kc, vc = cross[li]
        a = torch.softmax(q.matmul(kc.transpose(2, 3)) * ca.scale, dim=-1)
        attn_last = a[:, :, -1, :]
        y = ca.out_proj(a.matmul(vc).transpose(1, 2).reshape(R, 1, ca.embed_dim))
        x = layer.norm2(x + layer.dropout2(y))
        x = layer.norm3(x + layer.dropout3(layer.mlp(x)))
    return decoder.tgt_word_prj(x[:, -1]), new, attn_last


def _forward_test_kv(decoder, memory, capture=False):
    """`forward_test` with cached keys / values → (probabilities (B, steps, classes), per step the last-layer cross-attention
    of the newest character averaged over heads (B, N) if capture). Same stopping rule: every row has produced EOS."""
    import torch
    bs = memory.shape[0]
    with torch.inference_mode():
        cross = _cross_kv(decoder, memory)
        tok = torch.full((bs,), decoder.bos, dtype=torch.int64, device=memory.device)
        ended = torch.zeros(bs, dtype=torch.bool, device=memory.device)
        cache, probs, maps = None, [], []
        for pos in range(decoder.max_len):
            logits, cache, attn = _step(decoder, tok, pos, cache, cross)
            prob = torch.softmax(logits, dim=-1)
            probs.append(prob[:, None])
            if capture:
                maps.append(attn.mean(dim=1).detach())
            tok = prob.argmax(-1)
            ended |= tok == decoder.eos
            if bool(ended.all()):
                break
    return torch.cat(probs, dim=1), maps


def prepare(img):
    """PIL image → uint8 HWC array in the recognizer's input size (evaluation path of the public SVTRv2-AR weights)."""
    from PIL import Image
    w, h = img.size
    if h > ROT_RATIO * w:
        img = img.transpose(Image.ROTATE_90)
    w0, h0 = img.size
    r = int(np.clip(round(w0 / max(1, h0)), 1, len(BASE_SHAPE)))
    imgW, imgH = BASE_SHAPE[r - 1]
    return np.array(img.convert('RGB').resize((imgW, imgH), Image.BICUBIC), dtype=np.uint8)


def _text(tokens, vocab):
    return ''.join(vocab[t] for t in tokens if t != 0)


def greedy(decoder, memory, vocab, *, return_attention=False, kv=None, tf32=None):
    """`_greedy` under the recognizer precision (`precision` · tf32 None → `tf32_default()`)."""
    with precision(memory.device, tf32):
        return _greedy(decoder, memory, vocab, return_attention, kv)


def _greedy(decoder, memory, vocab, return_attention=False, kv=None):
    """Deployed greedy path and its exact log probability (EOS included).

    With ``return_attention=True``, also return one float32 (characters, encoder tokens) array per image.
    The final decoder layer's last-query cross-attention is averaged over heads during this same forward pass.
    EOS and later steps are discarded per image; normalization and float16 quantization match
    ``window.attention_record``. The ordinary return value remains the list of greedy rows.
    kv: cached keys / values (`_forward_test_kv`) · None → `kv_default()`.
    """
    import torch
    captured = []
    handle = None
    if kv_default() if kv is None else kv:
        prob, captured = _forward_test_kv(decoder, memory, return_attention)
        prob = prob.float()
    else:
        if return_attention:
            def capture(module, inputs, output):
                captured.append(module.attn_map[:, :, -1, :].mean(dim=1).detach())
            handle = decoder.decoder[-1].cross_attn.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                prob = decoder.forward_test(memory).float()
        finally:
            if handle is not None:
                handle.remove()
    # Transfer the batch once instead of synchronizing a CUDA scalar for each token and probability.
    ids = prob.argmax(-1).cpu().numpy()
    prob = prob.cpu().numpy()
    out = []
    for b in range(len(memory)):
        tokens, score = [], 0.0
        for step in range(prob.shape[1]):
            t = int(ids[b, step])
            score += math.log(max(float(prob[b, step, t]), 1e-30))
            if t == decoder.eos:
                break
            tokens.append(t)
        out.append(dict(tokens=tuple(tokens), text=_text(tokens, vocab), raw_logp=score, length=len(tokens) + 1))
    if not return_attention:
        return out
    if len(captured) != prob.shape[1]:
        raise RuntimeError('greedy attention capture does not match the decoded steps')
    grid = torch.stack(captured, dim=1).float().cpu().numpy()
    maps = []
    for b, row in enumerate(out):
        raw = [np.asarray(m, dtype=np.float32) for m in grid[b, :len(row['tokens'])]]
        raw = [np.round(a / s, 6) for a in raw if (s := float(np.asarray(a, dtype=np.float64).sum())) > 0]
        stored = np.asarray(raw, dtype=np.float16) if raw else np.zeros((0, grid.shape[-1]), np.float16)
        maps.append(stored.astype(np.float32))
    return out, maps


def beam(decoder, memory, vocab, width=WIDTH, token_topk=TOKEN_TOPK, max_len=25, *, kv=None, tf32=None):
    """Batched beam search over an encoded batch — per image a list of {text, raw_logp, length, mean_logp, length_score}.
    kv: cached keys / values (`_beam_kv`) · None → `kv_default()`. tf32: matmul precision (`precision`)."""
    with precision(memory.device, tf32):
        return _beam(decoder, memory, vocab, width, token_topk, max_len, kv)


def _beam(decoder, memory, vocab, width, token_topk, max_len, kv):
    import torch
    if kv_default() if kv is None else kv:
        return _beam_kv(decoder, memory, vocab, width, token_topk, max_len)
    n = int(memory.shape[0])
    active = [[(tuple(), 0.0)] for _ in range(n)]
    finished = [[] for _ in range(n)]
    for _ in range(max_len + 1):
        flat = [(i, tokens, score) for i, beams in enumerate(active) for tokens, score in beams]
        if not flat:
            break
        length = max(len(t) for _, t, _ in flat)
        if any(len(t) != length for _, t, _ in flat):
            raise RuntimeError('live beam lengths diverged')
        ids = torch.full((len(flat), length + 2), decoder.ignore_index, dtype=torch.long, device=memory.device)
        ids[:, 0] = decoder.bos
        for row, (_, tokens, _) in enumerate(flat):
            if tokens:
                ids[row, 1:1 + len(tokens)] = torch.as_tensor(tokens, dtype=torch.long, device=memory.device)
            ids[row, 1 + len(tokens)] = decoder.eos
        image_index = torch.as_tensor([r[0] for r in flat], dtype=torch.long, device=memory.device)
        with torch.inference_mode():
            logits = decoder.forward_train(memory.index_select(0, image_index), ids)
            logp = logits[:, -1].float().log_softmax(-1)
        k = min(token_topk, logp.shape[-1])
        values, indices = torch.topk(logp, k, dim=-1)
        values, indices = values.cpu().numpy(), indices.cpu().numpy()
        eos_values = logp[:, decoder.eos].cpu().numpy()
        proposals = [[] for _ in range(n)]
        for row, (i, tokens, score) in enumerate(flat):
            choices = {int(t): float(v) for t, v in zip(indices[row], values[row])}
            choices[decoder.eos] = float(eos_values[row])
            for t, lp in choices.items():
                s = score + lp
                if t == decoder.eos:
                    finished[i].append((tokens, s, len(tokens) + 1))
                elif len(tokens) < max_len:
                    proposals[i].append((tokens + (t,), s))
        for i in range(n):
            proposals[i].sort(key=lambda r: -r[1])
            active[i] = proposals[i][:width]
            finished[i].sort(key=lambda r: -r[1])
            finished[i] = finished[i][:width * 4]
            if len(finished[i]) >= width and active[i] and finished[i][width - 1][1] >= active[i][0][1]:
                active[i] = []
    out = []
    for i in range(n):
        cands = finished[i] or [(t, s, len(t) + 1) for t, s in active[i]]
        unique = {}
        for tokens, raw, length in cands:
            text = _text(tokens, vocab)
            row = dict(text=text, raw_logp=float(raw), length=int(length), mean_logp=float(raw / max(1, length)), length_score=float(raw))
            if text not in unique or row['raw_logp'] > unique[text]['raw_logp']:
                unique[text] = row
        out.append(sorted(unique.values(), key=lambda r: -r['length_score'])[:width])
    return out


def _beam_kv(decoder, memory, vocab, width, token_topk, max_len):
    """`beam` with cached keys / values — every live hypothesis carries the cache row of its parent; one step feeds only
    its newest character. Proposal, pruning, stopping and output rules are those of `beam`."""
    import torch
    n = int(memory.shape[0])
    active = [[(tuple(), 0.0, -1)] for _ in range(n)]
    finished = [[] for _ in range(n)]
    with torch.inference_mode():
        cross_all = _cross_kv(decoder, memory)
        cache = None
        for pos in range(max_len + 1):
            flat = [(i, tokens, score, parent) for i, beams in enumerate(active) for tokens, score, parent in beams]
            if not flat:
                break
            if any(len(t) != pos for _, t, _, _ in flat):
                raise RuntimeError('live beam lengths diverged')
            image_index = torch.as_tensor([r[0] for r in flat], dtype=torch.long, device=memory.device)
            if pos == 0:
                tok = torch.full((len(flat),), decoder.bos, dtype=torch.long, device=memory.device)
            else:
                parent = torch.as_tensor([r[3] for r in flat], dtype=torch.long, device=memory.device)
                cache = [(k.index_select(0, parent), v.index_select(0, parent)) for k, v in cache]
                tok = torch.as_tensor([r[1][-1] for r in flat], dtype=torch.long, device=memory.device)
            cross = [(k.index_select(0, image_index), v.index_select(0, image_index)) for k, v in cross_all]
            logits, cache, _ = _step(decoder, tok, pos, cache, cross)
            logp = logits.float().log_softmax(-1)
            k = min(token_topk, logp.shape[-1])
            values, indices = torch.topk(logp, k, dim=-1)
            values, indices = values.cpu().numpy(), indices.cpu().numpy()
            eos_values = logp[:, decoder.eos].cpu().numpy()
            proposals = [[] for _ in range(n)]
            for row, (i, tokens, score, _) in enumerate(flat):
                choices = {int(t): float(v) for t, v in zip(indices[row], values[row])}
                choices[decoder.eos] = float(eos_values[row])
                for t, lp in choices.items():
                    s = score + lp
                    if t == decoder.eos:
                        finished[i].append((tokens, s, len(tokens) + 1))
                    elif len(tokens) < max_len:
                        proposals[i].append((tokens + (t,), s, row))
            for i in range(n):
                proposals[i].sort(key=lambda r: -r[1])
                active[i] = proposals[i][:width]
                finished[i].sort(key=lambda r: -r[1])
                finished[i] = finished[i][:width * 4]
                if len(finished[i]) >= width and active[i] and finished[i][width - 1][1] >= active[i][0][1]:
                    active[i] = []
    out = []
    for i in range(n):
        cands = finished[i] or [(t, s, len(t) + 1) for t, s, _ in active[i]]
        unique = {}
        for tokens, raw, length in cands:
            text = _text(tokens, vocab)
            row = dict(text=text, raw_logp=float(raw), length=int(length), mean_logp=float(raw / max(1, length)), length_score=float(raw))
            if text not in unique or row['raw_logp'] > unique[text]['raw_logp']:
                unique[text] = row
        out.append(sorted(unique.values(), key=lambda r: -r['length_score'])[:width])
    return out


def decode(model, vocab, arrays, device, width=WIDTH, token_topk=TOKEN_TOPK, max_len=25, *, return_attention=False, kv=None,
           tf32=None):
    """Encode a batch of same-size arrays and return (greedy rows, beam lists); a is appended if pruned.

    ``return_attention=True`` adds a third result: the per-image character maps from ``greedy``, reusing
    the same encoder and greedy decoder pass. The hook is removed before beam search begins.
    kv: cached keys / values in both decoders · None → `kv_default()`. tf32: matmul precision (`precision`).
    """
    import torch
    with precision(device, tf32):
        x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
        x = (x.float() / 255 - .5) / .5
        with torch.inference_mode():
            memory = model.encoder(x)
        if return_attention:
            g, maps = greedy(model.decoder, memory, vocab, return_attention=True, kv=kv, tf32=tf32)
        else:
            g = greedy(model.decoder, memory, vocab, kv=kv, tf32=tf32)
        b = beam(model.decoder, memory, vocab, width, token_topk, max_len, kv=kv, tf32=tf32)
    attach_greedy(g, b)
    return (g, b, maps) if return_attention else (g, b)


def attach_greedy(g, b):
    """Append a to its beam list with the greedy score when the beam pruned it (`forced`), then order by log P (in place)."""
    for anchor, cands in zip(g, b):
        anchor['forced'] = anchor['text'] not in {r['text'] for r in cands}
        if anchor['forced']:
            cands.append(dict(text=anchor['text'], raw_logp=anchor['raw_logp'], length=anchor['length'],
                              mean_logp=anchor['raw_logp'] / max(1, anchor['length']), length_score=anchor['raw_logp'],
                              forced_greedy=True))
        cands.sort(key=lambda r: -r['length_score'])


def candidates(rows):
    """Beam records ({key, gt, greedy, nbest}) → {key: {'ar': a, 'ar_lp': log P of a, 'gt', 'cands': [{y, ar, nb}]}} — the
    top 10 of the list (the empty hypothesis kept); a outside the top 10 is put in front with rank −1."""
    out = {}
    for d in rows:
        nb, g = d['nbest'], d['greedy']
        x = next((x for x in nb if x['text'] == g), None)
        if x is None:
            raise RuntimeError(f"{d['key']}: no score of the greedy transcript in the beam record")
        cs = [dict(y=y['text'], ar=y['raw_logp'], nb=i) for i, y in enumerate(nb[:TOP])]
        if not any(c['y'] == g for c in cs):
            cs.insert(0, dict(y=g, ar=x['raw_logp'], nb=-1))
        out[d['key']] = dict(ar=g, ar_lp=x['raw_logp'], gt=d['gt'], cands=cs)
    return out


def read_jsonl(paths):
    for p in paths:
        with open(p) as fh:
            for line in fh:
                d = json.loads(line)
                d['key'] = d['key'].replace('evaluation/', 'test/')
                yield d


def run(pool, out, device='cuda:0', batch=48, width=WIDTH, token_topk=TOKEN_TOPK, max_len=25, workers=16, checkpoint=None, kv=None,
        tf32=None):
    """Beam lists of every image of a pool TSV (dataset, sample_idx, gt) → JSONL, one image per line; resumes by key."""
    import torch
    from tools._io import open_image, open_lmdb, read_sample, read_tsv
    from tools.rescore.window_score import load_reader
    rows = [dict(key=f"{r['dataset']}#{int(r['sample_idx'])}", gt=r.get('gt') or r.get('full') or r.get('label'))
            for r in read_tsv(pool)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['key'] for line in open(out)} if out.exists() else set()
    todo = [r for r in rows if r['key'] not in done]
    envs = {}
    for r in todo:
        ds = r['key'].rsplit('#', 1)[0]
        envs.setdefault(ds, None)
    envs = {ds: open_lmdb(ds) for ds in envs}

    def load(r):
        ds, i = r['key'].rsplit('#', 1)
        raw, _ = read_sample(envs[ds], int(i))
        return None if raw is None else (r, prepare(open_image(raw)))

    with ThreadPoolExecutor(workers) as ex:
        prepared = [v for v in ex.map(load, todo, chunksize=16) if v is not None]
    buckets = defaultdict(list)
    for r, a in prepared:
        buckets[a.shape[:2]].append((r, a))
    model, char_index = load_reader(device, checkpoint)
    vocab = sorted(char_index, key=char_index.get)
    t0, written = time.time(), 0
    with out.open('a') as fh:
        for shape, items in sorted(buckets.items()):
            for s in range(0, len(items), batch):
                part = items[s:s + batch]
                g, b = decode(model, vocab, [a for _, a in part], device, width, token_topk, max_len, kv=kv, tf32=tf32)
                for (r, _), anchor, cands in zip(part, g, b):
                    fh.write(json.dumps(dict(key=r['key'], gt=r['gt'], greedy=anchor['text'], greedy_forced=anchor['forced'],
                                             nbest=cands), ensure_ascii=False) + '\n')
                    written += 1
                fh.flush()
                print(f'{shape} · {len(done) + written:,}/{len(rows):,} · {written / max(time.time() - t0, 1e-9):.1f} img/s', flush=True)
    meta = dict(method='SVTRv2-AR beam search', pool=str(pool), images=len(rows), beam=width, token_topk=token_topk, max_len=max_len,
                search_objective='raw cumulative token log probability including EOS', torch=torch.__version__,
                kv_cache=kv_default() if kv is None else bool(kv), matmul_tf32=matmul_tf32(device, tf32),
                device=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else 'cpu',
                pool_sha256=hashlib.sha256(Path(pool).read_bytes()).hexdigest())
    out.with_suffix(out.suffix + '.meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--width', type=int, help=f'beam width (default {WIDTH})')
    ap.add_argument('--token-topk', type=int, help=f'characters expanded per hypothesis (default {TOKEN_TOPK})')
    ap.add_argument('--no-kv', action='store_true', help=f're-run the whole prefix at every step (default: cached, {KV_ENV})')
    ap.add_argument('--fp32', action='store_true', help=f'fp32 matmuls (default: TF32 on Ampere or newer, {TF32_ENV})')
    ap.add_argument('--paper', action='store_true', help='the first manuscript version (v1): beam 20 · 20, no cache, fp32')
    a = ap.parse_args()
    paper = PAPER if a.paper else {}
    run(a.pool, a.out, a.device, a.batch, a.width or paper.get('width', WIDTH), a.token_topk or paper.get('token_topk', TOKEN_TOPK),
        checkpoint=a.checkpoint, kv=False if a.no_kv else paper.get('kv'), tf32=False if a.fp32 else paper.get('tf32'))


if __name__ == '__main__':
    main()
