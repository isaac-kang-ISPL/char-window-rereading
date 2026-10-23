#!/usr/bin/env python3
"""Recorded measurements — values the manuscript uses that are not recomputed from the stored inputs.

Timings cannot be measured again to the same bytes, published numbers come from the cited papers, and the run records
describe how the stored beam lists and the validation pool were made. These files are shipped as data; this stage checks
each against its recorded SHA-256 (not by recomputation) and copies it to the output root next to the recomputed results.
The producing scripts named below ran in the research tree and are not part of the release.

    python -m tools.experiments.rescore.recorded
"""
from __future__ import annotations

import hashlib
import shutil

from tools.experiments.rescore import common as C
from tools.rescore import store

# public path (data root) → (SHA-256 of the copies the manuscript cites, what it is, producing script). In the research tree
# `store.at` maps a public path to its file (research_layout.json): the manuscript's copies under <data root>/recorded/ — the
# 10 · 10 TF32 run records, and the cost table of the data root's run (MW3 or segment windows). Several hashes: the MW3 and the
# segment-window runs cite different copies, and the public data archive carries copies with relative paths and server names
# replaced (same values) — the last hash of each entry that has one.
RECORDED = {
    'recorded/cost_per_stage.json':
        (('d8e0f6ab43df5993e71e3d7c1eaf30269511594f036398e04ab5d74613088213',      # MW3 run (main text)
          'a8525300050645cc83dcce86d44cf53e2f2f5c0ae6b03a6503a89158bb831241',      # segment windows (supplement)
          '17800a52b01f4823d3fc1a47ed40ed6c6fe8839b97f277a4a9a2c4f98fcc1c57'),     # public bundle copy of the MW3 file
         'time per stage, each stage alone, 1,000 images, one RTX A6000', 'rescore_cost_e2e.py'),
    'recorded/cost_end_to_end.json':
        (('58d6b320769420ba299d27b97ae8c41084391dcc70f5f4a2f608c733bb277335',
          'd7ed2024db44f217d394bfbb9e3b0c2723f920f7ee641fdbf7a3398ba11fddfa'),
         'time per image of each configuration end to end (Table cost), 1,000 images, one RTX A6000', 'rescore_cost_overlap.py'),
    'recorded/pll_timing.json':
        (('10621a2bdf41625d00439de80f354a9c8b53ee70df8e75b11fac65dba367e833',
          'fcf63e5baadaee836780a824038380c7c6d2fd375f96d4a9b595ee9a5596d322'),
         'ByT5-XL PLL time with and without sharing strings across images', 'pll_timing.py'),
    'recorded/beam_runs/cbub.meta.json':
        (('2c247b62729558d1636f35b25799e9e2ae60da316ae0d4f15d2e610c87811395',
          'e5300c6fedf11c1e3c8c7bcba7370ae522714b9e48107e772d86025da3cdfa2e'), 'beam-run record (beam 10 · 10, TF32), CB + UB except General',
         'nbest.py'),
    'recorded/beam_runs/general_s0.meta.json':
        (('41dbbff9ef7ce8aab5d90452301f834b43e48ef8caf72795802429bb01727c75',
          '6fea58eb9c33b6bd493e0e5382269a81a3acab39eaa1c970e7b8b8e67723b1b2'), 'beam-run record (beam 10 · 10, TF32), General shard 0', 'nbest.py'),
    'recorded/beam_runs/general_s1.meta.json':
        (('4515feafa03d2144a6f8e5fc075391e6eb3831c6d148c6c9358e4d85a3a771a3',
          '06d3a34e226f2fc9ea38525a5f456c90a70c3a529c0050e2701e4b11d3b7af72'), 'beam-run record (beam 10 · 10, TF32), General shard 1', 'nbest.py'),
    'recorded/beam_runs/general_s2.meta.json':
        (('74089baeab18c3db1661f04f5a0baf169161874fb9c2afd9ea35701f5a6ad9f2',
          'f2d8738b9d6a8b095075f6762b35ac9c5e782465168c83d8649fe50ad7d18c0b'), 'beam-run record (beam 10 · 10, TF32), General shard 2', 'nbest.py'),
    'recorded/beam_runs/natdev.meta.json':
        (('d8b1708585268e2db04852e0c7a270bf70c3339e96a7903c8282f53a7b241d99',
          '3e97d638929643ef1d6051a4ebaee48f327d7049a7c9dcb95b5679f0b2c365f3'), 'beam-run record (beam 10 · 10, TF32), validation set', 'nbest.py'),
    'recorded/byt5_refiner_training_pairs.json':
        (('483853db33d51804a5cf1bf2ef5c3d9561eb6ac6d07aa34ea62aff9b360e4fed',), 'training pairs of the fine-tuned ByT5 refiner', 'byt5_finetune.py'),
    'recorded/published_results.json':
        (('aebeafe206f654cc1a2ef155e15effd62d507a7174f0a59d96d1e399c6f108fa',
          '43198648568cdaefe19ce7c9c6d63df733decbc0cd82f76b98f9750953da34ae'), 'published results, extracted from the cited papers',
         'published_sota.py'),
    'recorded/validation_set.meta.json':
        (('d12be3eb703d9d0abca9b23ed508d8c0ae74d525c85a9bb5880d4334ffc399ef',),
         'definition of the original 10,000-image validation pool (Union14M-L-val, seed 20260920); the validation set of '
         'the manuscript replaces 27 of its images (benchmarks/validation-set.json)', None),
    'recorded/validation_set.tsv':
        (('ab55bdbde7b8e67f3ecd6491e091d5608aa7de1b6ac2c1238baa4e39a2540923',),
         'the original 10,000-image validation pool (dataset, index, label; benchmarks/validation-set.json replaces 27 of '
         'them): of the Union14M-L-val images with a label of 1 to 25 printable '
         'ASCII characters (398,228), the first 10,000 in the order of SHA-256("20260920|<dataset>#<index>")', None),
}


def source(rel):
    """The file of a recorded public path under the data root."""
    return store.at(rel)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    bad = []
    for rel, (h, _, _) in RECORDED.items():
        src = source(rel)
        if not src.exists() or sha(src) not in h:
            bad.append(rel)
            continue
        dst = C.out(rel)
        if dst.resolve() != src.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    if bad:
        raise SystemExit('recorded files missing or with another SHA-256:\n  ' + '\n  '.join(bad))
    print(f'{len(RECORDED)} recorded files match their SHA-256', flush=True)


REPRODUCE = (('recorded', 'main', {}, (), None),)        # checked by SHA-256 here, not compared with a reference tree

if __name__ == '__main__':
    main()
