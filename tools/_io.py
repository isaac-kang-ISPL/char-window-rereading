"""Common I/O — LMDB images, str_scores TSV files, transcript normalisation, and the locations of the data and weights
that live outside the code (`local_path`).
"""
import csv
import io
import os
import re

# lmdb and PIL are imported inside the functions: the normalisation and TSV half of this module needs neither.


# ── Locations outside the code ──────────────────────────────────────────────
# Each location resolves as: environment variable > `local_paths.json` at the repository root (a research checkout;
# a release does not ship it) > the release default (relative to the repository root). Only JSON is read here — no
# imports — so every package that ships this module can use it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PATHS = os.path.join(REPO_ROOT, 'local_paths.json')
PATH_ENV = {'data_root': 'STR_DATA_ROOT',             # LMDB benchmarks: <root>/<dataset>/data.mdb
            'scores_root': 'STR_SCORES_ROOT',         # <root>/<dataset>/STR_models/<model>/str_scores.tsv
            'charout_root': 'STR_CHAROUT_ROOT',       # stored attention records: <root>/<dataset>/svtrv2-ar.lmdb
            'ar_checkpoint': 'LEGIT_AR_CHECKPOINT'}   # SVTRv2-AR checkpoint (UnionST-U14MFilter)
RELEASE_PATHS = {'data_root': 'data/openocr', 'scores_root': 'data/str_scores', 'charout_root': 'data/charout'}
#                 (no default checkpoint: a release sets LEGIT_AR_CHECKPOINT or passes --checkpoint)
_LOCAL = None


def local_paths():
    """Contents of `local_paths.json` — {} when the file is absent (a release)."""
    global _LOCAL
    if _LOCAL is None:
        import json
        try:
            with open(LOCAL_PATHS, encoding='utf-8') as fh:
                _LOCAL = json.load(fh)
        except FileNotFoundError:
            _LOCAL = {}
    return _LOCAL


def _from_repo(path):
    """`~` expanded; a relative path is taken from the repository root."""
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def local_path(name, env=True):
    """Location `name` as a str — `PATH_ENV[name]` (when env and set) > `local_paths.json` > `RELEASE_PATHS[name]`
    (None if a name has neither). A `local_paths.json` value is a path or a list of candidates: the first that exists,
    else the last. Existence is checked where the path is used (`missing`), not here."""
    var = PATH_ENV.get(name)
    if env and var and os.environ.get(var):
        return os.path.expanduser(os.environ[var])
    value = local_paths().get(name, RELEASE_PATHS.get(name))
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        candidates = [_from_repo(v) for v in value]
        return next((c for c in candidates if os.path.exists(c)), candidates[-1])
    return _from_repo(value)


def missing(name, path, what):
    """FileNotFoundError for a location that does not exist — names the environment variable that sets it."""
    var = PATH_ENV.get(name)
    return FileNotFoundError(f'{what} not found: {path}' + (f' — set {var}' if var else ''))


DATA_ROOT = local_path('data_root')
SCORES_ROOT = local_path('scores_root')

_RE_NON_ALNUM = re.compile(r'[^a-zA-Z0-9]')
_RE_NON_94 = re.compile(r'[^a-zA-Z0-9!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]')


# ── Three transcript yardsticks ─────────────────────────────────────────────
#
# | Question                                        | Function        | Why that yardstick |
# |-------------------------------------------------|-----------------|--------------------|
# | scoring — is this label right                   | `normalize_94`  | the product is a label: case and punctuation are part of it |
# | equivalence — do two readings say the same      | `same_reading`  | no ground truth; a case difference alone is not a different reading |
# | benchmark — compare with published numbers      | `normalize_36`  | STR papers count 36 characters (digits and letters, one case) |
#
# The manuscript's CB / UB accuracies use `normalize_paper` (below), the yardstick of the published numbers.


def normalize_36(text):
    """36-character set: alphanumerics only, lower case — the benchmark yardstick (table above).

    Folds with `normalize_94` first (accents to the base letter, `É` -> `E`) and then keeps [0-9a-z] after lower-casing,
    so `é`, `É` and `e` all become `e`. Filtering ASCII first would delete accented letters instead (`Télég` -> `tlg`).
    Not for scoring labels (`normalize_94`) and not directly for equivalence (`same_reading`).
    """
    return _RE_NON_ALNUM.sub('', normalize_94(text)).lower()


def normalize_paper(text):
    """The manuscript's yardstick — ASCII alphanumerics only, lower case.

    Same as openrec `RecMetric(is_filter=True, is_lower=True, ignore_space=True)`. Accented letters are dropped, not
    folded (`Télég` -> `tlg`). With it the public SVTRv2-AR checkpoint (UnionST-U14MFilter) gives CB 97.84 and UB 91.39,
    the published numbers; `normalize_36` folds accents and gives CB 97.90 (UB is the same), so it is used only for
    equivalence in analyses.
    """
    return _NOT_ASCII_ALNUM.sub('', text).lower()     # keeps [0-9A-Za-z] character by character


_NOT_ASCII_ALNUM = re.compile(r'[^0-9A-Za-z]+')


normalize_36_paper = normalize_paper     # the display name of the yardstick in reports — the same function


def cluster_key(text):
    """Cluster key of the rescoring step that merges hypotheses reading alike — hypotheses with the same
    `normalize_paper` form share a cluster. A hypothesis whose normalized form is empty is not merged: it is a cluster
    of its own string (also when the usual output a has an empty form), so strings without alphanumerics do not pile
    up into one inflated cluster. A normalized form holds alphanumerics only, so it never collides with a key that
    starts with '\x00'. Every place that counts clusters uses this function."""
    q = normalize_paper(text)
    return q if q else '\x00' + text


def same_cluster(y, a):
    """Whether two hypotheses are in one cluster — equal `cluster_key`."""
    return cluster_key(y) == cluster_key(a)

def normalize_68(text):
    """68-character set: case folded, punctuation kept — the default for transcript equivalence.

        94 = 10 (digits) + 26 (upper) + 26 (lower) + 32 (punctuation)
        68 = 10          +              26         + 32       <- case folded
        36 = 10          +              26                    <- punctuation dropped too

    Two transcripts differ in two ways of different weight: case (a reader changing the convention of the whole line,
    `HULU` -> `hulu`), which is not a different reading, and punctuation (`browsers:` -> `browsers`, `1-7` -> `17`),
    which is. 36 characters miss the second and 94 characters count the first; 68 avoids both.
    For equivalence only — scoring is `normalize_94`.
    """
    return normalize_94(text).lower()


def same_reading(a, b):
    """Whether two readings say the same thing — equivalence without ground truth (68 characters: punctuation counts,
    case does not; see `normalize_68`). Used for agreement between runs and between policies, never for scoring.
    """
    return normalize_68(a) == normalize_68(b)


# Recognizer dictionary — every character SVTRv2-AR can output (`strtrain/utils/EN_symbol_dict.txt`, 94 characters).
# Labels are inside it too, so a character outside it in an output is an error, not a yardstick question.
_STR_DICT = None


def str_charset():
    """The SVTRv2-AR character set as a frozenset — from `strtrain/utils/EN_symbol_dict.txt`; a research checkout may
    list further candidate files under "str_dict" in `local_paths.json` (and then runs without normalisation if none is
    found). Without that entry a missing dictionary is an error rather than a silently empty set."""
    global _STR_DICT
    if _STR_DICT is None:
        here = os.path.join(REPO_ROOT, 'strtrain/utils/EN_symbol_dict.txt')
        fallback = local_paths().get('str_dict')
        for p in [here] + [_from_repo(q) for q in (fallback or [])]:
            try:
                _STR_DICT = frozenset(
                    c for c in open(p, encoding='utf-8').read().split('\n') if c)
                break
            except OSError:
                continue
        else:
            if fallback is None:
                raise FileNotFoundError(f'recognizer dictionary not found: {here}')
            _STR_DICT = frozenset()           # research checkout without a dictionary: normalisation is skipped
    return _STR_DICT


# Characters deleted from a transcript — trademark and degree signs are marks, not letters; annotators do not write
# them and the recognizer cannot output them. Dot-like characters (`·`, `•`) are not here: `_MAP` turns them into `.`
# (a character in both would never be mapped, since `_DROP` is applied first).
_DROP = frozenset('®©™§¶†‡′″‴‵°')

# Look-alike characters -> the Latin character with the same glyph. Only what NFKD does not already reduce to a base
# letter (accents `Ü`, full-width `Ａ`, Roman numerals `Ⅰ` and mathematical letters `𝐀` it does). Listed in advance:
# a Cyrillic `А` or Greek `Ο` would otherwise be a silent error that the eye cannot see.
_MAP = {
    # ── Cyrillic (same glyph as a Latin letter) ─────────────────────────
    'А': 'A', 'В': 'B', 'Е': 'E', 'З': '3', 'К': 'K', 'М': 'M', 'Н': 'H',
    'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X', 'Ѕ': 'S',
    'І': 'I', 'Ј': 'J', 'Ү': 'Y', 'Ԛ': 'Q', 'Ԝ': 'W', 'Ғ': 'F', 'Ԁ': 'D',
    'Ц': 'U', 'Ч': '4', 'Ь': 'b', 'Ъ': 'b', 'Э': 'E', 'Я': 'R', 'И': 'N',
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'ѕ': 's', 'і': 'i', 'ј': 'j', 'ԛ': 'q', 'ԝ': 'w', 'в': 'B', 'м': 'M',
    'т': 'T', 'н': 'H', 'к': 'k', 'г': 'r', 'п': 'n', 'ь': 'b',
    # ── Greek ───────────────────────────────────────────────────────────
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K',
    'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
    'Θ': 'O', 'Φ': 'O', 'Λ': 'A', 'Δ': 'A', 'Σ': 'E', 'Ω': 'O', 'Γ': 'r',
    'α': 'a', 'ο': 'o', 'ν': 'v', 'κ': 'k', 'ρ': 'p', 'τ': 't', 'υ': 'u',
    'χ': 'x', 'ι': 'i', 'ϲ': 'c', 'γ': 'y', 'σ': 'o', 'ω': 'w', 'ε': 'e',
    'μ': 'u', 'η': 'n', 'β': 'B', 'θ': 'o', 'φ': 'o', 'π': 'n', 'ζ': 'z',
    # ── Latin with a stroke (NFKD does not strip it) ────────────────────
    'Ø': 'O', 'ø': 'o', 'Đ': 'D', 'đ': 'd', 'Ð': 'D', 'ð': 'd', 'Ł': 'L',
    'ł': 'l', 'Ħ': 'H', 'ħ': 'h', 'Ŧ': 'T', 'ŧ': 't', 'Ɖ': 'D', 'Ɨ': 'I',
    'ı': 'i', 'ſ': 's', 'ɪ': 'I', 'ʟ': 'L', 'ᴀ': 'A', 'ᴏ': 'O', 'ʀ': 'R',
    'Ɵ': 'O', 'Ƶ': 'Z', 'ƶ': 'z', 'Ɛ': 'E', 'Ƨ': 'S', 'Ƽ': '5',
    # ── Currency and letterlike symbols ─────────────────────────────────
    '€': 'E', '£': 'L', '¥': 'Y', '₩': 'W', '₽': 'P', '¢': 'c', 'ℓ': 'l',
    '℮': 'e', 'Ω': 'O', 'Å': 'A', 'µ': 'u', '∅': 'O', '∆': 'A', '∏': 'n',
    # ── Quotes, dashes, division ────────────────────────────────────────
    '’': "'", '‘': "'", 'ʼ': "'", 'ʻ': "'", '´': "'", '`': "'", '＇': "'",
    '“': '"', '”': '"', '„': '"', '«': '"', '»': '"', '‹': "'", '›': "'",
    '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '―': '-', '−': '-',
    '⁄': '/', '∕': '/', '∶': ':', '⁃': '-', '‚': ',',
}
# Dot-like characters become a period (not deleted — a period is part of a label).
_MAP.update({'·': '.', '•': '.', '‧': '.', '․': '.', '∙': '.', '⋅': '.',
             '‥': '..', '…': '...'})


assert not (_DROP & set(_MAP)), 'a character in both _DROP and _MAP would never be mapped'


#: Hole placeholder — no candidate at all for that position.
#:
#: Not `'?'`: `?` is in the 94-character dictionary and is a real reading, so a placeholder and a reading would share a
#: character and could not be told apart in edit distances or agreement counts. A character outside the 94 shows the
#: hole in the string itself; `normalize_94` keeps it, so it also counts as a mismatch in scoring.
HOLE = '\u25a1'   # □


def normalize_94(text):
    """The scoring yardstick (first row of the table above): removes only what neither side can express.

    Case and punctuation are part of a label and stay. Four things change:

      1. Spaces are removed — the recognizer has no space character (`use_space_char: False`), while some labels
         contain spaces; counting them would measure the label convention, not the reader.
      2. Accents are stripped to the base letter (NFKD): `Ü` -> `U`, `İ` -> `I`; full-width letters likewise.
      3. Trademark and degree signs are deleted (`_DROP`).
      4. Look-alike characters become their Latin counterpart (`_MAP`), e.g. `·` -> `.`.

    A character outside the dictionary with no base letter (Hangul, CJK) is kept: it is evidence of an error.
    Not for equivalence (`same_reading`); benchmark comparisons use `normalize_36`.
    """
    import unicodedata as _u
    D = str_charset()
    out = []
    for ch in str(text or ''):
        if ch in _DROP:
            continue
        ch = _MAP.get(ch, ch)
        if not D or ch in D:
            out.append(ch)
            continue
        b = ''.join(c for c in _u.normalize('NFKD', ch) if not _u.combining(c))
        b = ''.join(_MAP.get(c, c) for c in b)
        out.append(b if b and all(c in D for c in b) else ch)
    return ''.join(out).replace(' ', '').strip()


def strip_non94(text):
    """Deletes characters outside the 94 (does not fold them) — benchmark preprocessing.

    Not the scoring yardstick, which is `normalize_94` (folds accents): `Télég` becomes `Tlg` here and `Teleg` there.
    """
    return _RE_NON_94.sub('', str(text))


NORMALIZE = {'36char': normalize_36, '94char': normalize_94}


# ── TSV ─────────────────────────────────────────────────────────────────────
# Two TSV families with opposite quoting rules — the family decides the reader.
#
#   ① TSV files written by this repository (`write_tsv`): values are written as they are; `"` is a character.
#      -> `parse_tsv` / `read_tsv` (split on tabs, no quote handling)
#   ② `<scores root>/**/str_scores.tsv`: written by `csv.DictWriter(delimiter='\t')` (QUOTE_MINIMAL) — a value with
#      `"` is wrapped in quotes and its quotes doubled.
#      -> `read_str_scores` / `iter_str_scores` / `load_preds` (csv defaults)
#
# Reading either family with the other's rule fails silently: ① read as csv swallows lines after a stray `"`, and ②
# read without quote handling keeps the wrapping quotes and splits values that contain line ends.
#
# The column count of ① is checked: splitting never loses a row, so a tab inside a value is the only corruption
# signal of this format, and it raises instead of passing.
def parse_tsv(text, *, path=None, on_bad='error'):
    """A TSV string of family ① -> list[dict]. Quotes are not interpreted (`str_scores` files are family ②:
    `read_str_scores`). `\r\n` and `\r` line ends are accepted.

    A row whose column count differs from the header means a tab inside a value. `on_bad`:
      `'error'` (default)  ValueError — files this repository wrote must be intact
      `'skip'`             drop the row and report the number dropped on stderr
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()                     # the empty line after the final line end
    if not lines:
        return []
    cols = lines[0].split('\t')
    where = f' ({path})' if path else ''
    out, bad = [], 0
    for i, ln in enumerate(lines[1:], 2):
        vals = ln.split('\t')
        if len(vals) != len(cols):
            if on_bad == 'skip':
                bad += 1
                continue
            raise ValueError(
                f'line {i} has {len(vals)} columns, the header {len(cols)}{where} — '
                f'a value contains a tab: {ln[:80]!r}')
        out.append(dict(zip(cols, vals)))
    if bad:
        import sys as _s
        print(f'  ! TSV{where}: dropped {bad} rows with a wrong column count '
              f'(read {len(out)} intact rows)', file=_s.stderr)
    return out


def read_tsv(path, *, expect=None, on_bad='error'):
    """TSV file -> list[dict]. With `expect`, checks the number of rows."""
    import pathlib
    p = pathlib.Path(path)
    rows = parse_tsv(p.read_text(encoding='utf-8'), path=p.name, on_bad=on_bad)
    if expect is not None and len(rows) != expect:
        raise ValueError(f'{p.name}: {len(rows)} rows, expected {expect}')
    return rows


def write_tsv(path, rows, cols=None):
    """list[dict] -> TSV file. A value with a tab or line end is refused before writing: written as it is, it would
    shift the columns of the next read, where its origin can no longer be found.
    """
    import pathlib
    rows = list(rows)
    if cols is None:
        cols = list(rows[0]) if rows else []
    lines = ['\t'.join(cols)]
    for i, r in enumerate(rows, 1):
        vals = []
        for c in cols:
            v = '' if r.get(c) is None else str(r.get(c))
            if '\t' in v or '\n' in v or '\r' in v:
                raise ValueError(f'row {i}, column {c!r} contains a tab or line end: {v[:60]!r}')
            vals.append(v)
        lines.append('\t'.join(vals))
    pathlib.Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return len(rows)


# ── LMDB ────────────────────────────────────────────────────────────────────
def open_lmdb(dataset, root=None, cache=None):
    """dataset is relative to the data root (e.g. 'OST/heavy', 'u14m/curve'); root defaults to DATA_ROOT."""
    if cache is not None and dataset in cache:
        return cache[dataset]
    import lmdb
    path = os.path.join(root or DATA_ROOT, dataset)
    if not os.path.exists(path):
        raise (missing('data_root', path, 'LMDB') if root is None else FileNotFoundError(f'LMDB not found: {path}'))
    env = lmdb.open(path,
                    readonly=True, lock=False, readahead=False, meminit=False,
                    max_readers=32)
    if cache is not None:
        cache[dataset] = env
    return env


def num_samples(env):
    with env.begin() as txn:
        return int(txn.get(b'num-samples'))


def read_sample(env, idx):
    """(image_bytes, label) — (None, None) when the index has no image."""
    with env.begin(write=False) as txn:
        raw = txn.get(f'image-{idx:09d}'.encode())
        lab = txn.get(f'label-{idx:09d}'.encode())
    if raw is None:
        return None, None
    return raw, (lab.decode('utf-8', 'ignore') if lab else '')


def open_image(img_bytes):
    from PIL import Image
    return Image.open(io.BytesIO(img_bytes))


# ── str_scores TSV ──────────────────────────────────────────────────────────
def scores_path(dataset, model, root=None):
    return os.path.join(root or SCORES_ROOT, dataset, 'STR_models', model, 'str_scores.tsv')


STR_SCORES_COLS = ['dataset', 'sample_idx', 'pred', 'gt', 'pred_mean_lp']


def read_str_scores(path):
    """Family ② — yields the rows of one `str_scores.tsv` as dicts; the only reader of these files.

    They were written with `csv.DictWriter(delimiter='\t')` (QUOTE_MINIMAL), so they are read with the csv defaults;
    `QUOTE_NONE` or `read_tsv` would misread every value containing `"`. Yields nothing when the file is missing.
    """
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def write_str_scores(path, rows, cols=None):
    """Writes family ② — `csv.DictWriter(delimiter='\t')`, the rule of the other `str_scores.tsv` files
    (`write_tsv`, family ①, would give this one file a different quoting rule and break its next read).
    """
    rows = list(rows)
    cols = cols or (list(rows[0]) if rows else STR_SCORES_COLS)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def iter_str_scores(dataset, model, root=None, comparison=None, errors_only=False):
    """Rows of the str_scores.tsv of `(dataset, model)` as dicts — on top of `read_str_scores`."""
    norm = NORMALIZE[comparison] if (errors_only and comparison) else None
    for row in read_str_scores(scores_path(dataset, model, root)):
        if norm is not None and norm(row.get('pred', '')) == norm(row.get('gt', '')):
            continue
        yield row


def load_preds(dataset, model, root=None, cap=None, comparison='36char'):
    """{sample_idx: (pred, gt, mean_lp)} — None when the file is missing.

    comparison: '36char' (default, the evaluation yardstick of the trainer) / '94char' (case and punctuation kept) /
    None (raw text).
    """
    path = scores_path(dataset, model, root)
    if not os.path.exists(path):
        return None
    f = NORMALIZE.get(comparison, lambda s: str(s)) if comparison else (lambda s: str(s))
    out = {}
    for i, row in enumerate(iter_str_scores(dataset, model, root)):
        if cap and i >= cap:
            break
        try:
            lp = float(row['pred_mean_lp'])
        except (TypeError, ValueError, KeyError):
            lp = float('nan')
        out[str(row['sample_idx'])] = (f(row['pred']), f(row['gt']), lp)
    return out
