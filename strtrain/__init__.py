"""STR recognizer training harness, kept apart from the annotation and rescoring tools under `tools/`.

  · `engine/`      configuration loading · training loop (`Trainer`)
  · `data/`        LMDB datasets · samplers · collate
  · `utils/`       character set · checkpoints · EMA · logging
  · `train_rec.py` training entry point
  · `utility.py`   argument parser

Models, losses, pre- and post-processing are in `openrec/` (the upstream OpenOCR package name, hence top level).
Training configurations are in `configs/`. The rescoring release ships only `engine/config.py` (configuration loading)
and `utils/EN_symbol_dict.txt` (the recognizer dictionary).
"""
