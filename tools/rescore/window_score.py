"""창 점수 W(y) — 가설 y 를 그 장의 글자 창마다 같은 인식기(svtrv2-ar)로 다시 읽어 모은 값.

창 w(자리 범위 lo..hi · `window.prepare` 의 slot_ids)에 대해 y 를 a 위로 편집 정렬해 그 범위에 대응하는 y 의 부분 문자열 c_w(y) 를 잡고,
창 그림을 입력으로 c_w(y) + ⟨eos⟩ 를 교사 강제한 로그우도를 창 점수 s_w(y) 로 한다. 기본은 겹침 가중 합이다:

    s_w(y) = log P_AR(c_w(y) ⟨eos⟩ | 창 w)          W(y) = Σ_w u_w s_w(y)          u_w = mean_{i ∈ slot_ids(w)} 1 / n_i

n_i = 자리 i 를 덮는 유효 창 수(한 장의 모든 가설에 같은 무게). 원고 v1(`SCORE_PAPER_V1`)은 핵심 ± 경계 한 글자씩(변형 최대 넷)의
logsumexp 를 창 점수로 하고 무게 없이 더한다: s_w(y) = logsumexp_{v ∈ variants_w(y)} log P_AR(v ⟨eos⟩ | 창 w) · W(y) = Σ_w s_w(y).

결과는 장마다 {가설: W} — greedy a 도 넣는다(Δ 는 결정 쪽이 만든다). 규칙:
빈 문자열 · 사전 밖 글자가 든 가설은 뺀다 · 유효한 창이 하나도 없는 장은 모든 가설 0.0 · 어떤 창에서 부분 문자열을 못 재면 그 창 항 −1e30
(겹침 가중은 잰 항을 창 순서로 더한 뒤 −1e30 × 못 잰 창 수 · 원고 v1 합은 창 순서대로 −1e30 을 끼워 더한다 — 저장 점수와 같음) ·
창을 못 만든 장(`window.prepare` 가 skip)은 결과에 열쇠가 없다.
인식기의 float32 행렬곱은 기본이 TF32(Ampere 이상 · `nbest.precision`)다. 원고 저장 점수는 fp32 —
`tf32=False` · `--fp32` · LEGIT_AR_TF32=0(`PAPER_V1_RUN` 은 fp32 를 포함한다).

`SCORE_PAPER_V1` gives the stored v1 scores bit for bit on the same machine and batches as the research code that
produced them; the comparison lives in the research repository.

    PYTHONPATH=. python -m tools.rescore.window_score --pool <TSV(dataset, sample_idx)> --hyps <JSON {key: [y, …]}> --out <dir> \
        --device cuda:0 [--checkpoint <UnionST-U14MFilter.pth>]
    # v1 definition: … --paper-v1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().with_name('data') / 'svtrv2-ar.yml'   # inference configuration of the recognizer
MAX_LEN = 25            # 교사 강제할 부분 문자열의 최대 길이(인식기 출력 길이 상한)
CHUNK = 5000            # 저장 묶음 — 셋마다 정렬한 장 번호를 5,000장씩
WINDOW_BATCH = 256      # 인코더에 한 번에 넣는 창 수(같은 입력 크기끼리)
CANDIDATE_BATCH = 2048  # 디코더에 한 번에 넣는 (창, 부분 문자열) 수
AGGREGATIONS = ('coverage', 'sum')
SCORE_DEFINITION = dict(boundary=False, aggregation='coverage')   # 기본 — 핵심 부분 문자열 하나 · 겹침 가중
SCORE_PAPER_V1 = dict(boundary=True, aggregation='sum')           # 원고 v1 — 경계 변형 logsumexp · 합


# ---------------------------------------------------------------- 가설 ↔ 창 대응

def edit_ops(a, g):
    """a → g 편집 정렬(레벤슈타인 DP · 뒤에서 되짚기 · 같음/바꿈 > a 에만 > g 에만 순) → [(op, i, j)] · op '=' · 'S' · 'I'(a 에만) · 'D'(g 에만)."""
    n, m = len(a), len(g); D = np.zeros((n + 1, m + 1), int); D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != g[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != g[j - 1]):
            out.append(('=' if a[i - 1] == g[j - 1] else 'S', i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            out.append(('I', i - 1, None)); i -= 1
        else:
            out.append(('D', None, j - 1)); j -= 1
    return out[::-1]


def coords(a, y):
    """y 의 글자마다 a 위 좌표 — 같음 · 바꿈은 그 자리 i, y 에만 있는 글자는 바로 앞 대응 자리 + 0.5."""
    out = []; last = -1
    for op, i, j in edit_ops(a, y):
        if op in ('=', 'S'):
            out.append((float(i), y[j])); last = i
        elif op == 'D':
            out.append((last + .5, y[j]))
    return out


def variants(lo, hi, L, units):
    """창이 덮는 a 의 자리 lo..hi 에 대응하는 y 의 부분 문자열 — 핵심 ± 경계 한 글자씩(최대 넷).
    첫 창은 y 앞에 붙은 글자를, 끝 창은 뒤에 붙은 글자를 핵심에 넣는다. 핵심이 비면(y 가 그 자리를 모두 뺐다) 경계 글자만."""
    core = [k for k, (c, _) in enumerate(units) if lo <= c <= hi or (c < 0 and lo == 0) or (c > L - 1 and hi == L - 1)]
    if core:
        k0, k1 = core[0], core[-1]
    else:
        before = [k for k, (c, _) in enumerate(units) if c < lo]
        k0 = (before[-1] + 1) if before else 0; k1 = k0 - 1
    mid = ''.join(ch for _, ch in units[k0:k1 + 1])
    left = units[k0 - 1][1] if k0 - 1 >= 0 else None
    right = units[k1 + 1][1] if k1 + 1 < len(units) else None
    return [l + mid + r for l in ([''] + ([left] if left else [])) for r in ([''] + ([right] if right else []))]


# ---------------------------------------------------------------- 교사 강제 로그우도

def encode_text(text, char_index, max_len):
    if len(text) > max_len or any(char not in char_index for char in text):
        return None
    return [char_index[char] for char in text]


def forced_scores(decoder, memory, requests, char_index, candidate_batch, max_len):
    """[(창 번호, 문자열)] → {(창 번호, 문자열): (log P(문자열 ⟨eos⟩ | 창), 토큰당 평균)} — ⟨bos⟩ 뒤 문자열 + ⟨eos⟩ 를 한 번에 교사 강제.
    비슷한 길이끼리 묶어 패딩을 줄인다. 사전 밖 글자 · max_len 보다 긴 문자열은 결과에 없다."""
    import torch
    encoded = []
    for image_index, text in requests:
        tokens = encode_text(text, char_index, max_len)
        if tokens is not None:
            encoded.append((image_index, text, tokens))
    encoded.sort(key=lambda item: len(item[2]))
    output = {}
    for start in range(0, len(encoded), candidate_batch):
        part = encoded[start:start + candidate_batch]
        width = max(len(tokens) for _image, _text, tokens in part) + 2
        ids_np = np.full((len(part), width), decoder.ignore_index, dtype=np.int64)
        ids_np[:, 0] = decoder.bos
        lengths = np.empty(len(part), dtype=np.int64)
        for row, (_image, _text, tokens) in enumerate(part):
            n = len(tokens)
            if n:
                ids_np[row, 1:1 + n] = tokens
            ids_np[row, 1 + n] = decoder.eos
            lengths[row] = n + 1
        ids = torch.from_numpy(ids_np).to(memory.device, non_blocking=True)
        image_index = torch.as_tensor([item[0] for item in part], dtype=torch.long, device=memory.device)
        with torch.inference_mode():
            logits = decoder.forward_train(memory.index_select(0, image_index), ids)
            logp = logits.float().log_softmax(-1)
            targets = ids[:, 1:]
            gathered = logp.gather(-1, targets.clamp_max(logp.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
            valid = torch.arange(gathered.shape[1], device=gathered.device)[None, :] < torch.from_numpy(lengths).to(gathered.device)[:, None]
            raw = torch.where(valid, gathered, torch.zeros_like(gathered)).sum(1).cpu().numpy()
        for row, (image_i, text, _tokens) in enumerate(part):
            value = float(raw[row])
            output[(image_i, text)] = (value, value / int(lengths[row]))
    return output


def checkpoint_path(checkpoint=None):
    """The SVTRv2-AR checkpoint — `checkpoint`, else LEGIT_AR_CHECKPOINT > `local_paths.json` (`tools._io.local_path`;
    a release has no default). Stops with the setting to use when the file is not there."""
    from tools._io import local_path
    path = checkpoint or local_path('ar_checkpoint')
    if not path or not Path(path).is_file():
        raise FileNotFoundError('SVTRv2-AR checkpoint (UnionST-U14MFilter.pth) ' + (f'not found: {path}' if path else 'not set')
                                + ' — set LEGIT_AR_CHECKPOINT or pass --checkpoint')
    return Path(path)


def __getattr__(name):
    """`CHECKPOINT` (LEGIT_AR_CHECKPOINT > `local_paths.json`) is resolved when it is read, so a release without a
    checkpoint imports cleanly and stops only where the checkpoint is needed; `load_reader` also checks that it exists."""
    if name == 'CHECKPOINT':
        from tools._io import local_path
        path = local_path('ar_checkpoint')
        return Path(path) if path else checkpoint_path()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def reader_config():
    """`CONFIG` as a dict, with the dictionary path taken from the repository root instead of the working directory."""
    from strtrain.engine.config import Config
    cfg = Config(str(CONFIG)).cfg
    for part in (cfg['Global'], cfg['PostProcess']):
        path = part.get('character_dict_path')
        if path and not os.path.isabs(os.path.expanduser(path)):
            part['character_dict_path'] = os.path.normpath(os.path.join(REPO, path))
    return cfg


def load_reader(device, checkpoint=None):
    """svtrv2-ar(공개 가중치 UnionST-U14MFilter) → (모델, 사전 {글자: 번호}) — 디코더 출력의 마지막 둘(특수 토큰)은 사전에서 뺀다."""
    import torch
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    cfg = reader_config()
    post = build_post_process(cfg['PostProcess'], cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels'] = post.get_character_num()
    model = build_model(cfg['Architecture'])
    state = torch.load(checkpoint_path(checkpoint), map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('state_dict', state.get('model', state)), strict=True)
    model = model.eval().to(device)
    vocab = list(post.character)[:model.decoder.out_channels - 2]
    return model, {c: k for k, c in enumerate(vocab)}


def coverage_weights(slots, length):
    """u_w = mean_{i ∈ slots_w} 1 / (자리 i 를 덮는 유효 창 수) — 창마다 무게 하나(가설과 무관)."""
    count = np.zeros(length, dtype=np.int64)
    for ids in slots:
        count[ids] += 1
    return [float(np.mean(1. / count[ids])) for ids in slots]


def aggregate(raw, aggregation='coverage'):
    """창별 원점수 → {key: {y: W(y)}} — `score_windows` 가 쓰는 바로 그 집계(GPU 회차와 CPU 재집계가 바이트 같다).
    raw: `score_windows(detail=True)` 의 둘째 값, 또는 `run` 이 쓴 <묶음>.raw.pkl.gz 를 읽은 객체(dict(definition, rows)).
    'sum'(원고 v1): 창 순서대로 더하며 못 잰 창은 −1e30 을 끼워 더한다 · 'coverage': 잰 항 u_w · s_w 를 창 순서로 더한 뒤 −1e30 × 못 잰 창 수."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'aggregation must be one of {AGGREGATIONS}')
    rows = raw['rows'] if 'rows' in raw and 'definition' in raw else raw
    out = {}
    for key, r in rows.items():
        weights = [w['u'] for w in r['windows']]
        res = {}
        for y, each in r['s'].items():
            tot = 0.
            if aggregation == 'sum':
                for value in each:
                    tot += value if value is not None else -1e30
            else:
                for value, u_w in zip(each, weights):
                    if value is not None:
                        tot += u_w * value
                tot = tot - 1e30 * float(sum(value is None for value in each))
            res[y] = tot
        out[key] = res
    return out


def hypotheses_of(text, hyps_y, char_index):
    """채점할 가설 — a 를 맨 앞에, 중복 · 빈 문자열 · 사전 밖 글자가 든 가설을 뺀다."""
    return [y for y in dict.fromkeys([text] + list(hyps_y)) if y and all(c in char_index for c in y)]


def plan_record(record, hyps_y, char_index, boundary=False):
    """한 장의 채점 계획 — GPU 앞의 Python 계산(편집 정렬 · coords · 부분 문자열 · 창 무게)을 장마다 끝낸 것.
    `run` 은 CPU 일꾼(`_prepare`)에서 미리 만들어 기록의 'plan' 에 붙이고, `score_windows` 는 붙은 계획이 지금 인자와 맞으면
    그대로 쓰고 없거나 안 맞으면 그 자리에서 이 함수로 만든다 — 어느 쪽이든 같은 값이다.
    → dict(text, ys, boundary, n_windows, local=[유효 창의 r['windows'] 번호], meta=[dict(slot_ids, target, u)],
           per=[(y, [(유효 창 순번, [부분 문자열…])…])], need=[유효 창마다 물을 부분 문자열(정렬 · 중복 없음)])."""
    a = record['text']; L = len(a)
    ys = hypotheses_of(a, hyps_y, char_index)
    local, spans, meta = [], [], []
    for k, w in enumerate(record['windows']):
        if not w['valid'] or not w['slot_ids']:
            continue
        local.append(k); spans.append((min(w['slot_ids']), max(w['slot_ids'])))
        meta.append(dict(slot_ids=[int(i) for i in w['slot_ids']], target=int(w.get('target_slot', -1))))
    weights = coverage_weights([m['slot_ids'] for m in meta], L)
    for m, u_w in zip(meta, weights):
        m['u'] = u_w
    per, sets = [], [set() for _ in local]
    for y in ys:
        u = coords(a, y)
        row = [(n, variants(lo, hi, L, u) if boundary else variants(lo, hi, L, u)[:1]) for n, (lo, hi) in enumerate(spans)]
        for n, vs in row:
            sets[n].update(vs)
        per.append((y, row))
    return dict(text=a, ys=ys, boundary=bool(boundary), n_windows=len(record['windows']), local=local, meta=meta, per=per,
                need=[sorted(s) for s in sets])


def _plan_for(record, hyps, char_index, boundary):
    """기록에 붙은 계획이 이 호출(전사 · 가설 · 경계 변형 · 창 수)과 맞으면 그것을, 아니면 새로 만든 계획."""
    p = record.get('plan')
    ys = hypotheses_of(record['text'], hyps.get(record['key'], []), char_index)
    if (p is not None and p.get('text') == record['text'] and p.get('ys') == ys and p.get('boundary') == bool(boundary)
            and p.get('n_windows') == len(record['windows'])):
        return p
    return plan_record(record, hyps.get(record['key'], []), char_index, boundary)


def score_windows(records, hyps, model, char_index, device, cand_batch=CANDIDATE_BATCH, *, boundary=False,
                  aggregation='coverage', detail=False, tf32=None):
    """records = [dict(key, text=a, windows=`window.prepare` 의 창 목록[, plan=`plan_record`])] · hyps = {key: [y, …]} → {key: {y: W(y)}}(a 가 맨 앞).
    boundary: 경계 변형(최대 넷)의 logsumexp 를 창 점수로(원고 v1) · 기본은 핵심 부분 문자열 하나 ·
    aggregation: 'coverage'(기본 · 겹침 가중) | 'sum'(원고 v1) · detail: 참이면 (W, 창별 원점수) —
    원점수 = {key: dict(windows=[dict(slot_ids, target, u)], s={y: [창마다 s_w(y) | None]})}(None = 그 창에서 못 잼).
    기록에 미리 만든 계획(`plan_record`)이 있고 이 호출과 맞으면 그것을 쓴다(GPU 프로세스의 직렬 Python 계산을 줄임 · 값은 같다).
    tf32: 인식기 행렬곱 정밀도(`nbest.precision` · None → LEGIT_AR_TF32 · 기본 TF32)."""
    import torch
    from tools.rescore import nbest
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'aggregation must be one of {AGGREGATIONS}')
    dec = model.decoder
    wins, plan = [], []                                   # 창(장 번호 · AR 입력) · 장마다 (계획, 첫 창의 전역 번호)
    for ri, r in enumerate(records):
        p = _plan_for(r, hyps, char_index, boundary)
        off = len(wins)
        wins.extend((ri, r['windows'][k]['image']) for k in p['local'])
        plan.append((p, off))
    by_win = {off + n: vs for p, off in plan for n, vs in enumerate(p['need']) if vs}
    scores, groups = {}, {}
    for wi, (_, arr) in enumerate(wins):
        if wi in by_win:
            groups.setdefault(arr.shape, []).append(wi)
    with nbest.precision(device, tf32):
        for shape, ids in groups.items():
            for s in range(0, len(ids), WINDOW_BATCH):
                part = ids[s:s + WINDOW_BATCH]
                x = torch.from_numpy(np.stack([wins[i][1] for i in part])).permute(0, 3, 1, 2).to(device)
                with torch.inference_mode():
                    mem = model.encoder((x.float() / 255 - .5) / .5)
                req = [(j, v) for j, wi in enumerate(part) for v in by_win[wi]]
                got = forced_scores(dec, mem, req, char_index, cand_batch, MAX_LEN)
                for (j, v), (raw, _) in got.items():
                    scores[(part[j], v)] = raw
    raw = {}
    for r, (p, off) in zip(records, plan):
        s_all = {}
        for y, ws in p['per']:
            each = []
            for n, vs in ws:
                wi = off + n
                vals = [scores[(wi, v)] for v in vs if (wi, v) in scores]
                each.append(float(np.logaddexp.reduce(vals)) if vals else None)
            s_all[y] = each
        # 창 정보는 상수 키로 새로 만든다 — 일꾼에서 풀어 온 dict 의 키는 장마다 다른 문자열 객체라 원점수 pickle 바이트가 달라진다
        raw[r['key']] = dict(windows=[dict(slot_ids=list(m['slot_ids']), target=m['target'], u=m['u']) for m in p['meta']], s=s_all)
    out = aggregate(raw, aggregation)
    return (out, raw) if detail else out


# ---------------------------------------------------------------- 묶음 실행(저장된 attention 기록 · 원본 LMDB)

_W = {}


def _init(dataset, data_root, charout_root, boxes_path, kwargs, stored_geometry=True, planning=None):
    """일꾼 초기화 — dataset 은 기본 셋(None 이면 일감마다 `_prepare(..., dataset=)` 로 받는다 · 셋마다 원본 LMDB 를 처음 쓸 때 연다) ·
    planning = dict(char_index, boundary) 이면 `_prepare` 가 장마다 채점 계획(`plan_record`)까지 만든다."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import cv2
    cv2.setNumThreads(1)
    _W.update(dataset=dataset, data_root=data_root, charout_root=charout_root, kwargs=kwargs, stored_geometry=stored_geometry,
              planning=planning, sources={})
    if dataset is not None:
        _source(dataset)
    if boxes_path:
        import pickle
        with open(boxes_path, 'rb') as fh:
            _W['boxes'] = pickle.load(fh)


def read_image(env, index):
    """원본 LMDB(image-%09d) 한 장 → RGB uint8 배열 | None."""
    import io
    from PIL import Image
    with env.begin() as txn:
        raw = txn.get(b'image-%09d' % int(index))
    return None if raw is None else np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))


def _source(dataset):
    """일꾼 안의 원본 LMDB — 셋마다 한 번 열어 둔다."""
    env = _W['sources'].get(dataset)
    if env is None:
        import lmdb
        env = _W['sources'][dataset] = lmdb.open(str(Path(_W['data_root']) / dataset), readonly=True, lock=False, readahead=False,
                                                 meminit=False)
    return env


def _prepare(indices, hyps=None, dataset=None):
    """장마다 창(`window.prepare`) — hyps({key: [y, …]})와 `_init` 의 planning 이 있으면 채점 계획(`plan_record`)도 붙인다.
    dataset 을 안 주면 `_init` 의 기본 셋."""
    from tools.rescore import window
    dataset = dataset or _W['dataset']
    out = []
    for index in indices:
        try:
            rgb = read_image(_source(dataset), index)
            if rgb is None:
                out.append(dict(index=index, skip='missing_image')); continue
            rec = window.stored_record(dataset, index, root=_W['charout_root'])
            # 미리 계산한 rbox 는 저장 기록(옛 전사)의 기하로 만든 것 — 교사 강제 기록을 읽은 장에는 쓰지 않고 그 자리에서 계산한다.
            # pickle 값이 목록이 아니면(오류 문자열) rbox 실패로 본다(TRW1.5).
            boxes = None if rec is None or rec.get('source') == window.FORCED else _W.get('boxes', {}).get(f"{dataset}#{index}")
            gi, bg0 = (window.paper_inputs(rgb, rec, dataset, index, root=_W['charout_root'])
                       if rec is not None and _W['stored_geometry'] else (None, False))
            got = window.prepare(rgb, rec, boxes=boxes, geometry_inputs=gi, bg0=bg0, **_W['kwargs'])
            got['index'] = index
            planning = _W.get('planning')
            if hyps is not None and planning and 'text' in got and 'windows' in got:
                key = f"{dataset}#{index}"
                try:                                                 # 계획이 실패하면 붙이지 않는다 — GPU 쪽이 다시 만들며 같은 오류를 낸다
                    got['plan'] = plan_record(dict(got, key=key), hyps.get(key, []), planning['char_index'], planning['boundary'])
                except Exception:
                    pass
            out.append(got)
        except Exception as exc:                                     # 한 장의 오류로 묶음 전체를 멈추지 않는다 — 결과에 열쇠가 없다
            out.append(dict(index=index, error=f'{type(exc).__name__}: {exc}'))
    return out


def _batches(pool, indices, batch_size, worker_chunk, hyps=None, dataset=None):
    """입력 순서대로 batch_size 장씩 — 다음 묶음을 미리 하나 제출해 둔다. hyps · dataset 을 주면 일감마다 그 장들의 가설만 넘긴다."""
    def task(part):
        if hyps is None:
            return pool.submit(_prepare, part)
        return pool.submit(_prepare, part, {k: hyps[k] for k in (f'{dataset}#{i}' for i in part) if k in hyps})

    def submit(start):
        stop = min(len(indices), start + batch_size)
        return [task(indices[i:min(i + worker_chunk, stop)]) for i in range(start, stop, worker_chunk)]
    pending = submit(0)
    try:
        for start in range(0, len(indices), batch_size):
            batch = [row for job in pending for row in job.result()]
            pending = submit(start + batch_size)
            yield batch
    finally:
        for job in pending:
            job.cancel()


def _stream(pool, jobs, batch_size, worker_chunk, hyps=None):
    """jobs = [(dataset, 장 번호들, 묶음 번호)] → (job 순번, 묶음, 그 job 의 마지막 묶음인가) — 묶음은 job 안에서만 batch_size 장씩
    자르고(묶음 구성은 job 마다 `_batches` 와 같다), 다음 묶음은 다음 job 의 첫 묶음이라도 미리 하나 제출해 둔다(한 풀을 회차 내내)."""
    order = [(j, s) for j, (_d, idx, _c) in enumerate(jobs) for s in range(0, len(idx), batch_size)]

    def task(dataset, part):
        sub = None if hyps is None else {k: hyps[k] for k in (f'{dataset}#{i}' for i in part) if k in hyps}
        return pool.submit(_prepare, part, sub, dataset)

    def submit(k):
        if k >= len(order):
            return []
        j, start = order[k]
        dataset, idx, _c = jobs[j]
        stop = min(len(idx), start + batch_size)
        return [task(dataset, idx[i:min(i + worker_chunk, stop)]) for i in range(start, stop, worker_chunk)]
    pending = submit(0)
    try:
        for k, (j, _start) in enumerate(order):
            batch = [row for job in pending for row in job.result()]
            pending = submit(k + 1)
            yield j, batch, k + 1 == len(order) or order[k + 1][0] != j
    finally:
        for job in pending:
            job.cancel()


def run(pool_rows, hyps, out, device, *, data_root=None, charout_root=None, boxes_path=None, workers=14, cpu_batch=256,
        shard=(0, 1), region='trw15', hops=1, fallback='trw15', stored_geometry=False, rotate=False, ptl='refined',
        segmentation='nearest', boundary=False, aggregation='coverage', raw=True, tf32=None, checkpoint=None):
    """장 목록 [(dataset, index)] → out/<셋>__c<묶음>.json({key: {y: W}}) · 이미 있는 묶음은 건너뛴다 · 끝나면 out/_done_s<i>of<n>.
    기본은 새 정의(개선 p_tl · 가장 가까운 씨앗 · TRW · 회전 없음 · 핵심 부분 문자열 · 겹침 가중) · 원고 v1 은 `**PAPER_V1_RUN`.
    stored_geometry=True 면 원고 회차처럼 charout 저장소의 p_tl · tl_4 · 분할 행을 먼저 읽는다(`window.paper_inputs` · 원고 v1 정의) ·
    boxes_path 는 원고 회차의 rbox pickle(교사 강제 기록 장에는 쓰지 않음) ·
    raw=True 면 같은 이름의 <묶음>.raw.pkl.gz 에 창별 원점수(`score_windows(detail=True)`)를 함께 쓴다 — 집계 · 계수를 GPU 없이 다시 낼 때.
    CPU 일꾼이 창과 함께 채점 계획(`plan_record` · 편집 정렬 · 부분 문자열)도 만들어, GPU 프로세스는 인코더 · 디코더와 집계만 한다.
    일꾼 풀은 회차 전체에서 하나이고, 셋 · 5,000장 묶음이 바뀌어도 다음 묶음의 창 준비를 GPU 채점과 겹친다(`_stream`).
    tf32: 인식기 행렬곱 정밀도(None → LEGIT_AR_TF32 · 기본 TF32) — 실제로 쓴 값을 원점수 사본의 definition 에 남긴다 ·
    checkpoint: 인식기 가중치(None → `checkpoint_path`)."""
    from tools._io import local_path
    from tools.rescore import nbest, window
    data_root = Path(os.path.expanduser(data_root)) if data_root else Path(local_path('data_root'))
    charout_root = Path(os.path.expanduser(charout_root)) if charout_root else window.charout_root()
    by = OrderedDict()
    for dataset, index in pool_rows:
        by.setdefault(dataset, []).append(int(index))
    jobs = [(d, sorted(v)[c:c + CHUNK], c // CHUNK) for d, v in by.items() for c in range(0, len(v), CHUNK)]
    jobs = [j for n, j in enumerate(jobs) if n % shard[1] == shard[0]]
    model, char_index = load_reader(device, checkpoint)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if stored_geometry and (ptl != 'original' or segmentation != 'watershed'):
        raise ValueError('stored_geometry reads the paper v1 geometry (original p_tl · watershed) — pass ptl/segmentation of v1')
    kwargs = dict(region=region, hops=hops, fallback=fallback, rotate=rotate, ptl=ptl, segmentation=segmentation)
    t0 = time.time()
    todo = []
    for dataset, idx, c in jobs:
        f = out / f"{dataset.replace('/', '__')}__c{c:03d}.json"
        if f.exists():
            print('SKIP', f.name, flush=True)
        else:
            todo.append((dataset, idx, c))

    def finish(dataset, c, res, detail, skipped):
        f = out / f"{dataset.replace('/', '__')}__c{c:03d}.json"
        if raw:                                                       # 원점수 사본을 먼저 — 묶음 JSON 이 있으면 끝난 묶음이다
            import gzip
            import pickle
            rf = f.with_name(f.stem + '.raw.pkl.gz'); rt = rf.with_name(rf.name + '.tmp')
            with gzip.open(rt, 'wb', compresslevel=1) as fh:
                pickle.dump(dict(definition=dict(kwargs, boundary=boundary, aggregation=aggregation, stored_geometry=stored_geometry,
                                                 matmul_tf32=nbest.matmul_tf32(device, tf32)),
                                 rows=detail), fh, protocol=5)
            rt.replace(rf)
        tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(res, ensure_ascii=False)); tmp.replace(f)
        print(f'{dataset}__c{c:03d}: {len(res):,} images · skipped {skipped} · {time.time() - t0:.0f}s so far', flush=True)
    errors = []
    if todo:
        planning = dict(char_index=char_index, boundary=boundary)    # 채점 계획(편집 정렬 · 부분 문자열)도 CPU 일꾼에서
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(None, data_root, charout_root, boxes_path, kwargs, stored_geometry, planning)) as ex:
            res, detail, skipped = {}, {}, 0
            for j, batch, last in _stream(ex, todo, cpu_batch, 2, hyps=hyps):
                dataset, _idx, c = todo[j]
                good = [r for r in batch if 'text' in r and 'windows' in r]
                skipped += len(batch) - len(good)
                errors += [dict(key=f"{dataset}#{r['index']}", error=r['error']) for r in batch if 'error' in r]
                for r in good:
                    r['key'] = f"{dataset}#{r['index']}"
                if good:
                    got, d = score_windows(good, hyps, model, char_index, device, boundary=boundary, aggregation=aggregation,
                                           detail=True, tf32=tf32)
                    res.update(got); detail.update(d)
                if last:                                              # 이 묶음의 마지막 — 쓰는 동안 일꾼은 다음 묶음을 만든다
                    finish(dataset, c, res, detail, skipped)
                    res, detail, skipped = {}, {}, 0
    efile = out / f'_errors_s{shard[0]}of{shard[1]}.json'
    if errors:                    # an exception is not a skip reason: the chunks are kept, the run is not marked complete
        efile.write_text(json.dumps(errors, ensure_ascii=False, indent=1) + '\n')
    if efile.exists():            # also on a resume that skipped the finished chunks: delete their files and this list to redo them
        raise RuntimeError(f'images raised while their windows were built (e.g. {json.loads(efile.read_text())[0]}); '
                           f'not marked done — see {efile}')
    (out / f'_done_s{shard[0]}of{shard[1]}').write_text(time.strftime('%F %T') + '\n')


PAPER_V1_RUN = dict(region='trer', rotate=True, ptl='original', segmentation='watershed', boundary=True, aggregation='sum',
                    stored_geometry=True, tf32=False)   # `run(..., **PAPER_V1_RUN)` — 원고 v1 저장 점수와 같은 실행(fp32)


def pack_record(rec):
    """attention 기록 → 저장 형식 bytes(npz · 지도 float16) — `window.unpack_record` 가 읽는 모양."""
    import io
    buf = io.BytesIO()
    maps = np.asarray(rec['maps'], dtype=np.float16)
    np.savez(buf, text=str(rec['text']), conf=np.asarray(rec.get('conf', []), dtype=np.float32), maps=maps,
             gw=int(rec['gw']), gh=int(rec['gh']), wh0=np.asarray(rec['wh0'], dtype=np.int32), rotated=bool(rec['rotated']))
    return buf.getvalue()


def dump_records(pool_rows, texts, root, device, *, data_root=None, tf32=None, checkpoint=None):
    """저장된 attention 덤프 없이 시작할 때 — 장마다 그림 + 전사(보통 빔의 greedy a)로 `window.attention_record` 를 만들어
    root/<셋>/svtrv2-ar.lmdb 에 저장 형식으로 쓴다. 이어서 `run(..., charout_root=root, stored_geometry=False)`.
    texts = {key: a} · 이미 있는 장은 건너뛴다 · tf32: 인식기 행렬곱 정밀도(`nbest.precision`) · checkpoint: 인식기 가중치."""
    import lmdb
    from tools._io import local_path
    from tools.rescore import nbest, window
    data_root = Path(os.path.expanduser(data_root)) if data_root else Path(local_path('data_root'))
    model, char_index = load_reader(device, checkpoint)
    by = OrderedDict()
    for dataset, index in pool_rows:
        by.setdefault(dataset, []).append(int(index))
    for dataset, idx in by.items():
        dest = Path(root) / dataset / f'{window.READER}.lmdb'
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = lmdb.open(str(Path(data_root) / dataset), readonly=True, lock=False, readahead=False, meminit=False)
        env = lmdb.open(str(dest), map_size=64 << 30, subdir=True, readahead=False)
        with env.begin() as txn:
            have = {int(k) for k in txn.cursor().iternext(values=False) if len(k) == 9 and k.isdigit()}
        rows, t0 = [], time.time()
        for n, index in enumerate(sorted(i for i in idx if i not in have)):
            key = f'{dataset}#{index}'
            rgb = read_image(src, index)
            text = texts.get(key)
            if rgb is None or not text:
                continue
            with nbest.precision(device, tf32):
                rec = window.attention_record(model, rgb, text, char_index, device)
            if rec is not None:
                rows.append((b'%09d' % index, pack_record(rec)))
            if len(rows) >= 500:
                with env.begin(write=True) as txn:
                    for k, v in rows:
                        txn.put(k, v)
                rows = []
        with env.begin(write=True) as txn:
            for k, v in rows:
                txn.put(k, v)
        env.sync(); env.close(); src.close()
        print(f'{dataset}: records for {len(idx):,} images · {time.time() - t0:.0f}s', flush=True)


def main(argv=None):
    from tools._io import read_tsv
    ap = argparse.ArgumentParser(description='Window scores W(y) of beam hypotheses, in chunks of 5,000 images per dataset')
    ap.add_argument('--pool', type=Path, required=True, help='TSV with the columns dataset and sample_idx')
    ap.add_argument('--hyps', type=Path, required=True, help='JSON {key: [y, …]} — key = "<dataset>#<sample_idx>"')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--checkpoint', type=Path, default=None, help='SVTRv2-AR checkpoint (default: LEGIT_AR_CHECKPOINT)')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--boxes', type=Path, default=None,
                    help='precomputed rbox pickle {key: [corners…]} — computed on the spot when absent')
    ap.add_argument('--region', default='trw15', choices=('trer', 'trw15', 'none'))
    ap.add_argument('--hops', type=int, default=1)
    ap.add_argument('--rotate', action='store_true', help='rotate each window upright along the reading tangent (v1)')
    ap.add_argument('--ptl', default='refined', choices=('refined', 'original'))
    ap.add_argument('--segmentation', default='nearest', choices=('nearest', 'watershed'))
    ap.add_argument('--boundary', action='store_true',
                    help='window score = logsumexp over the boundary variants (up to four) (v1)')
    ap.add_argument('--aggregation', default='coverage', choices=AGGREGATIONS)
    ap.add_argument('--stored-geometry', action='store_true',
                    help='read the stored p_tl, tl_4 and segmentation of the attention store first (v1 definition)')
    ap.add_argument('--fresh-geometry', action='store_true', help='(old flag; now the default) do not read stored geometry')
    ap.add_argument('--paper-v1', action='store_true',
                    help='the whole v1 definition: TRER, rotation, original p_tl, watershed, boundary variants, sum, stored geometry')
    ap.add_argument('--no-raw', action='store_true', help='do not write the per-window raw scores (<chunk>.raw.pkl.gz)')
    ap.add_argument('--fp32', action='store_true', help='fp32 matmuls in the recognizer (the stored v1 scores; default TF32 on Ampere or newer)')
    ap.add_argument('--records', type=Path, default=None,
                    help='root of the attention records (<root>/<dataset>/svtrv2-ar.lmdb) — images without one are recorded first '
                         'from the image and the first hypothesis (greedy a); the geometry is then measured afresh')
    a = ap.parse_args(argv)
    rows = [(r['dataset'], int(r['sample_idx'])) for r in read_tsv(a.pool)]
    hyps = json.loads(a.hyps.read_text())
    definition = dict(PAPER_V1_RUN) if a.paper_v1 else dict(
        region=a.region, rotate=a.rotate, ptl=a.ptl, segmentation=a.segmentation, boundary=a.boundary, aggregation=a.aggregation,
        stored_geometry=a.stored_geometry, tf32=False if a.fp32 else None)
    if a.fp32:
        definition['tf32'] = False
    if a.records is not None:
        dump_records(rows, {k: v[0] for k, v in hyps.items() if v}, a.records, a.device, tf32=definition['tf32'],
                     checkpoint=a.checkpoint)
    if a.records is not None or a.fresh_geometry:
        definition['stored_geometry'] = False
    run(rows, hyps, a.out, a.device, charout_root=a.records, boxes_path=a.boxes, workers=a.workers,
        shard=tuple(map(int, a.shard.split('/'))), hops=a.hops, raw=not a.no_raw, checkpoint=a.checkpoint, **definition)


if __name__ == '__main__':
    main()
