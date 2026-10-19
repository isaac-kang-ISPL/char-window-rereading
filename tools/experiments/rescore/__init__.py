"""Experiments of the manuscript on top of `tools.rescore` — comparisons and ablations of the re-reading method.

Every script reads the stored inputs through `tools.rescore.store` (data root `LEGIT_DATA`) and writes its result
file under the output root (`LEGIT_RESCORE_OUT`, default the data root) at the path the manuscript generator reads,
e.g. `reference/ar_trer_eval/ablation.json`. `reproduce.py` runs them in dependency order and
compares every result with a reference tree byte for byte.
"""
