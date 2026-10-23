import copy

from .base_recognizer import BaseRecognizer

__all__ = ['build_model']


def build_model(config):
    """Config -> model. Without `name` the model is `BaseRecognizer`."""
    config = copy.deepcopy(config)
    name = config.pop('name', None)
    if name in (None, '', 'BaseRecognizer'):
        return BaseRecognizer(config)
    raise ValueError(f'Unknown Architecture.name: {name!r} (available: BaseRecognizer)')
