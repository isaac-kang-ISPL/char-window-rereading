import copy

from .base_recognizer import BaseRecognizer

__all__ = ['build_model']


def build_model(config):
    """Config -> model. Without `name` the model is `BaseRecognizer`.

    `LegitRecognizer` (a training wrapper; its package `legit` is not part of the rescoring release) wraps a plain
    `BaseRecognizer`, so any model config switches to it with the single `name` line; it is registered below.
    """
    config = copy.deepcopy(config)
    name = config.pop('name', None)
    if name in (None, '', 'BaseRecognizer'):
        return BaseRecognizer(config)
    if name == 'LegitRecognizer':
        from .legit import LegitRecognizer
        return LegitRecognizer(config)
    raise ValueError(f'Unknown Architecture.name: {name!r} '
                     '(available: BaseRecognizer, LegitRecognizer)')
