"""Image-conditioned extension (manuscript Section III-F, Table 1) — optional; the fixed combination is the main method.

A frozen Qwen3.5-27B-GPTQ-Int4 sees only the image and answers three yes/no questions in one request (`PROMPT`): is a
character occluded (p_O), is the text handwritten (p_H), do clearly visible letters spell a nonstandard string (p_S).
p = P(yes) / (P(yes) + P(no)) at the answer's token. The first satisfied condition, tested in the order of `RULE`, supplies
the coefficients (α window, β length, γ ByT5-XL) of S(y) = Δb + α·ΔV + β·(|y| − |a|) + γ·ΔPLL; τ stays 0. Nonstandard
spelling is accepted only when p_H and p_O are at most 1 − t_S. An image that meets none of the conditions, or whose
answer is invalid, keeps the fixed coefficients `FIXED`.

The judge is asked only when its answer can matter: `needs_judge` decides every image with the four coefficient sets and
marks those whose outputs differ after normalize_36_paper; the others keep the fixed output without a request (6.0 % of
the CB and UB images need it). The decision itself is `decide.Batch.decide` with the chosen coefficients — no other
scoring path.

    rule  = condition.RULE                             # thresholds and coefficients of the manuscript (`ALL_THREE`)
    need  = condition.needs_judge(batch, pll)          # bool per image
    probs = condition.Judge(model_path).ask(images)    # {key: (p_O, p_H, p_S) or None}  (GPU, vLLM)
    out   = condition.decide(batch, pll, probs)        # one output per image, plus the route taken

Selected on development corpora, not on the evaluation sets (handwriting: GNHK train; occlusion: synthetic OST-style
images; spelling: synthetic nonstandard strings). With the stored judge answers of the manuscript this module gives its
values (CB 98.2238 · UB 92.0833; `tools/maintenance/condition_parity.py` in the research tree).
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
import re

import numpy as np

from tools._io import normalize_paper

MODEL = 'Qwen/Qwen3.5-27B-GPTQ-Int4'
MODEL_REVISION = '8f0c09f227ae570e79617c6d9172b59df9c16081'     # snapshot of the manuscript runs
PROMPT = (
    'Inspect the image and answer all three questions. Each answer refers to the same image.\n'
    '1. This image shows a word in a scene-text photograph. Is any character of the word partly or fully hidden by an '
    'occluding object or mark lying on top of it (for example a drawn line, scribble, blob, sticker, or another object in '
    'front of the text)? Letters cut off at the image border do not count, and blur, glare, shadow or low resolution do '
    'not count.\n'
    '2. Look at the text image. Is the text handwritten, with freehand pen or pencil strokes, rather than typeset or '
    'mechanically printed letters?\n'
    '3. Inspect the visible letters exactly as drawn, without correcting the spelling. Are the letters mostly clearly '
    'visible, but their literal sequence contains an unusual or misspelled word, or a random letter string, rather than '
    'a normally spelled word or name? If letters are too obscured or blurry to judge, answer no.\n'
    'Reply with exactly three lowercase words (each yes or no) separated by one space, in the numbered order. Do not add '
    'labels or an explanation.')
PROMPT_SHA256 = 'd5435b28f4c6'                                  # prefix of sha256(PROMPT) — the bytes of the manuscript runs
ANSWERS = ('occlusion', 'handwriting', 'spelling')              # order of the three answers in the reply
ANSWER_REGEX = r'(yes|no) (yes|no) (yes|no)'

FIXED = (.4, 1., .1)                                            # (α window, β length, γ ByT5-XL) of the fixed combination
# (condition, threshold, (α, β, γ)) in the order they are tested — thresholds chosen with the three conditions together
ALL_THREE = (('handwriting', .75, (.05, 1., .2)), ('occlusion', .10, (.3, 1.5, .2)), ('spelling', .65, (.45, 1., -.1)))
RULE = ALL_THREE
# one condition alone (supplementary rows): its own threshold, the other conditions off
SINGLE = {'handwriting': (('handwriting', .75, (.05, 1., .2)),), 'occlusion': (('occlusion', .10, (.3, 1.5, .2)),),
          'spelling': (('spelling', .50, (.45, 1., -.1)),)}


def _coefficients(rule):
    return [FIXED] + [c for _, _, c in rule]


def _decide(batch, pll, coef):
    alpha, beta, gamma = coef
    return list(batch.decide(pll, gamma, beta, 0., alpha))      # Batch.decide(pll, mu=γ, beta=β, tau=τ, omega=α)


def outputs(batch, pll, rule=RULE):
    """Outputs of every coefficient set: {'fixed' | condition: [output per image]}."""
    out = {'fixed': _decide(batch, pll, FIXED)}
    for name, _, coef in rule:
        out[name] = _decide(batch, pll, coef)
    return out


def needs_judge(batch, pll, rule=RULE, out=None):
    """True where some condition's coefficients give another normalize_36_paper output than the fixed ones — only these
    images are sent to the judge (labels and dataset names are not used)."""
    out = out or outputs(batch, pll, rule)
    base = [normalize_paper(y) for y in out['fixed']]
    need = np.zeros(len(base), bool)
    for name, _, _ in rule:
        need |= np.array([normalize_paper(y) != b for y, b in zip(out[name], base)], bool)
    return need


def route(p, rule=RULE):
    """Condition that supplies the coefficients for answers p = (p_O, p_H, p_S), or None (fixed). An invalid answer
    (None, or a non-finite value) never satisfies a condition, and makes spelling unavailable."""
    if p is None:
        return None
    p_o, p_h, p_s = (float('nan') if v is None else float(v) for v in p)
    value = dict(occlusion=p_o, handwriting=p_h, spelling=p_s)
    for name, t, _ in rule:
        v = value[name]
        if not math.isfinite(v) or v < t:
            continue
        if name == 'spelling' and not (math.isfinite(p_o) and math.isfinite(p_h) and p_o <= 1 - t and p_h <= 1 - t):
            continue
        return name
    return None


def decide(batch, pll, probs, rule=RULE, out=None):
    """Final output of every image of the batch: (outputs, routes). probs = {key: (p_O, p_H, p_S)} for the images that
    need the judge; an image that needs it but has no answer keeps the fixed output (route 'fixed', counted in the
    returned routes as 'missing')."""
    out = out or outputs(batch, pll, rule)
    need = needs_judge(batch, pll, rule, out)
    final, routes = [], []
    for i, (key, *_rest) in enumerate(batch.items):
        if not need[i]:
            final.append(out['fixed'][i]); routes.append('skipped')
            continue
        if key not in probs:
            final.append(out['fixed'][i]); routes.append('missing')
            continue
        r = route(probs[key], rule)
        final.append(out[r or 'fixed'][i]); routes.append(r or 'fixed')
    return final, routes


# ---------------------------------------------------------------- the judge (GPU · vLLM)

def judge_png(rgb):
    """The image the judge sees: RGB, upscaled so that the short side is at least 128 px unless the long side would pass
    768 px (never downscaled), BICUBIC, PNG bytes."""
    from PIL import Image
    im = Image.fromarray(np.asarray(rgb, np.uint8)).convert('RGB') if not isinstance(rgb, Image.Image) else rgb.convert('RGB')
    scale = max(1., 128 / max(1, min(im.size)))
    if max(im.size) * scale > 768:
        scale = max(1., 768 / max(im.size))
    size = tuple(max(1, round(v * scale)) for v in im.size)
    if size != im.size:
        im = im.resize(size, Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    im.save(buffer, format='PNG')
    return buffer.getvalue()


def parse(text, tokens, tops):
    """(p_O, p_H, p_S) from one reply: the token where each yes / no starts, P(yes) / (P(yes) + P(no)) over its top
    log-probabilities (case and leading spaces ignored, prefixes such as 'y' or 'n' counted). None for a reply that is
    not three answers, or an answer without mass on yes or no."""
    if not re.fullmatch(r'\s*(yes|no)' + r'\s+(yes|no)' * 2 + r'\s*', text, re.I):
        return None
    starts, offset = [], 0
    for t in tokens:
        starts.append(offset); offset += len(t)
    matches = list(re.finditer(r'(?i)\b(yes|no)\b', ''.join(tokens)))
    if len(matches) != 3:
        return None
    probs = []
    for m in matches:
        pos = max(i for i, s in enumerate(starts) if s <= m.start())
        y = n = 0.
        for token, lp in tops[pos].items():
            token = token.lstrip().lower()
            if token and 'yes'.startswith(token):
                y += math.exp(lp)
            elif token and 'no'.startswith(token):
                n += math.exp(lp)
        probs.append(y / (y + n) if y + n else None)
    return tuple(probs)


class Judge:
    """vLLM runner with the settings of the manuscript runs (vLLM 0.28: bf16 activations on the GPTQ Int4 weights,
    4,096 tokens, 32 sequences, greedy decoding constrained to three lowercase answers, 20 log-probabilities, thinking
    off). Needs `pip install vllm==0.28.0` and VLLM_USE_FLASHINFER_SAMPLER=0 where the FlashInfer sampler cannot compile."""

    def __init__(self, model=MODEL, *, gpu_memory_utilization=.8, max_num_seqs=32, tensor_parallel_size=1):
        assert hashlib.sha256(PROMPT.encode()).hexdigest().startswith(PROMPT_SHA256)
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        # the Hugging Face name is pinned to the snapshot of the manuscript runs; a local path is used as given
        pinned = dict(revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION) if str(model) == MODEL else {}
        self.llm = LLM(model=str(model), dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=gpu_memory_utilization,
                       limit_mm_per_prompt={'image': 1}, trust_remote_code=True, seed=0, max_num_seqs=max_num_seqs,
                       tensor_parallel_size=tensor_parallel_size, **pinned)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0, seed=0, logprobs=20, max_tokens=12,
                                     structured_outputs=StructuredOutputsParams(regex=ANSWER_REGEX))

    @staticmethod
    def message(png):
        url = 'data:image/png;base64,' + base64.b64encode(png).decode()
        return [dict(role='user', content=[dict(type='image_url', image_url=dict(url=url)), dict(type='text', text=PROMPT)])]

    def ask(self, images, chunk=96):
        """images: {key: RGB array or PIL image} → {key: (p_O, p_H, p_S) or None}."""
        keys, out = list(images), {}
        for b in range(0, len(keys), chunk):
            part = keys[b:b + chunk]
            replies = self.llm.chat([self.message(judge_png(images[k])) for k in part], self.params,
                                    chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
            for k, r in zip(part, replies):
                g = r.outputs[0]
                tokens = [self.tok.decode([i]) for i in g.token_ids]
                tops = [{self.tok.decode([i]): float(v.logprob) for i, v in lp.items()} for lp in g.logprobs]
                out[k] = parse(g.text, tokens, tops)
        return out
