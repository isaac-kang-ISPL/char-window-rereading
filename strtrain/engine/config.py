import os
import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Mapping

import yaml

__all__ = ['Config']


def _expand_env(s):
    """Expand ${VAR} and ${VAR:-default} occurrences in a string from os.environ.

    Used for _BASE_ paths so that an external caller can swap architecture
    (e.g. `_BASE_: ['./svtrv2_mdiffdecoder_${ARCH:-base}.yml']` resolves to
    `_base.yml` by default and `_small.yml` when ARCH=small is exported).
    """
    if not isinstance(s, str):
        return s

    def repl(m):
        expr = m.group(1)
        if ':-' in expr:
            var, default = expr.split(':-', 1)
            return os.environ.get(var, default)
        return os.environ.get(expr, m.group(0))

    return re.sub(r'\$\{([^}]+)\}', repl, s)


class ArgsParser(ArgumentParser):

    def __init__(self):
        super(ArgsParser,
              self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument('-o',
                          '--opt',
                          nargs='*',
                          action='append',
                          help='set configuration options')
        self.add_argument('--local_rank')

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, 'Please specify --config=configure_file_path.'
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        # action='append' + nargs='*' → list of lists; flatten before processing
        flat = []
        for item in opts:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        for s in flat:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur = cur.setdefault(key, {})
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive."""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


def _merge_dict(config, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested to
    an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        config: dict onto which the merge is executed
        merge_dct: dct merged into config

    Returns: dct
    """
    for key, value in merge_dct.items():
        sub_keys = key.split('.')
        key = sub_keys[0]
        if key in config and len(sub_keys) > 1:
            _merge_dict(config[key], {'.'.join(sub_keys[1:]): value})
        elif key in config and isinstance(config[key], dict) and isinstance(
                value, Mapping):
            _merge_dict(config[key], value)
        else:
            config[key] = value
    return config


def maybe_apply_unionst_transforms(cfg):
    """Auto-swap PARSeqAugPIL → PARSeqAugPILPP when training on UnionST synthetic
    data (paper §9: online resampling 0.1–1× + JPEG compression p=0.2).

    Triggered when any path in Train.dataset.data_dir_list contains 'UnionST'
    (i.e. --data=ust-sp via scripts/_resolve_specs.sh). No-op otherwise.
    """
    train = cfg.get('Train', {}) if isinstance(cfg, dict) else {}
    ds = train.get('dataset', {}) if isinstance(train, dict) else {}
    dirs = ds.get('data_dir_list') or []
    if not any('UnionST' in str(d) for d in dirs):
        return False
    tfs = ds.get('transforms')
    if not isinstance(tfs, list):
        return False
    swapped = False
    for i, op in enumerate(tfs):
        if not isinstance(op, dict):
            continue
        if 'PARSeqAugPIL' in op and 'PARSeqAugPILPP' not in op:
            tfs[i] = {'PARSeqAugPILPP': op['PARSeqAugPIL']}
            swapped = True
    if swapped:
        print('[config] UnionST detected in data_dir_list — '
              'Train.dataset.transforms: PARSeqAugPIL → PARSeqAugPILPP '
              '(adds online resampling 0.1–1× + JPEG compression p=0.2).')
    return swapped


def maybe_force_aug_pp(cfg):
    """Force PARSeqAugPIL → PARSeqAugPILPP when Global.aug_pp is truthy,
    regardless of data (the explicit analogue of the data-driven swap in
    maybe_apply_unionst_transforms). Also injects PP args from Global.aug_pp_*:
        aug_pp_phase_order, aug_pp_o_num_layers, aug_pp_o_magnitude,
        aug_pp_compress_p, aug_pp_resample_p, aug_pp_resample_min,
        aug_pp_resample_max, aug_pp_jpeg_q_low, aug_pp_jpeg_q_high
    Run after maybe_apply_unionst_transforms so UnionST runs also pick up the
    custom phase_order. No-op when Global.aug_pp is unset/false.
    """
    g = cfg.get('Global', {}) if isinstance(cfg, dict) else {}
    if not g.get('aug_pp'):
        return False
    train = cfg.get('Train', {}) if isinstance(cfg, dict) else {}
    ds = train.get('dataset', {}) if isinstance(train, dict) else {}
    tfs = ds.get('transforms')
    if not isinstance(tfs, list):
        return False

    # Collect PP-arg overrides from Global.aug_pp_* keys.
    override = {}
    for arg, key in [('phase_order', 'aug_pp_phase_order'),
                     ('o_num_layers', 'aug_pp_o_num_layers'),
                     ('o_magnitude', 'aug_pp_o_magnitude'),
                     ('compress_p', 'aug_pp_compress_p'),
                     ('resample_p', 'aug_pp_resample_p'),
                     ('resample_min', 'aug_pp_resample_min'),
                     ('resample_max', 'aug_pp_resample_max')]:
        if g.get(key) is not None:
            override[arg] = g[key]
    if g.get('aug_pp_jpeg_q_low') is not None or g.get('aug_pp_jpeg_q_high') is not None:
        override['jpeg_q'] = [int(g.get('aug_pp_jpeg_q_low', 5)),
                              int(g.get('aug_pp_jpeg_q_high', 90))]

    swapped = False
    for i, op in enumerate(tfs):
        if not isinstance(op, dict):
            continue
        if 'PARSeqAugPILPP' in op:
            args = dict(op['PARSeqAugPILPP'] or {})
            args.update(override)
            tfs[i] = {'PARSeqAugPILPP': args or None}
            swapped = True
        elif 'PARSeqAugPIL' in op:
            args = op['PARSeqAugPIL']
            args = dict(args) if isinstance(args, dict) else {}
            args.update(override)
            tfs[i] = {'PARSeqAugPILPP': args or None}
            swapped = True
    if swapped:
        print(f'[config] Global.aug_pp set — Train.dataset.transforms forced to '
              f'PARSeqAugPILPP (overrides: {override or "defaults"}).')
    return swapped


def print_dict(cfg, print_func=print, delimiter=0):
    """Recursively visualize a dict and indenting acrrording by the
    relationship of keys."""
    for k, v in sorted(cfg.items()):
        if isinstance(v, dict):
            print_func('{}{} : '.format(delimiter * ' ', str(k)))
            print_dict(v, print_func, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print_func('{}{} : '.format(delimiter * ' ', str(k)))
            for value in v:
                print_dict(value, print_func, delimiter + 4)
        else:
            print_func('{}{} : {}'.format(delimiter * ' ', k, v))


class Config(object):

    def __init__(self, config_path, BASE_KEY='_BASE_'):
        self.BASE_KEY = BASE_KEY
        self.cfg = self._load_config_with_base(config_path)
        # Snapshot taken BEFORE any -o override: the yml's Optimizer.lr is
        # calibrated against the yml's own Global.total_bs. Recording the pair
        # lets the trainer rescale lr when -o changes total_bs but leaves lr
        # alone (otherwise a smaller batch would keep the larger batch's lr).
        self._yml_total_bs = (self.cfg.get('Global') or {}).get('total_bs')
        self._lr_explicit = False
        if self._yml_total_bs:
            self.cfg.setdefault('Global', {})['lr_base_bs'] = self._yml_total_bs

    @staticmethod
    def _sets_optimizer_lr(args):
        """True if `args` (a parsed -o dict) assigns Optimizer.lr."""
        if not isinstance(args, dict):
            return False
        if 'Optimizer.lr' in args:
            return True
        opt = args.get('Optimizer')
        return isinstance(opt, Mapping) and 'lr' in opt

    def _load_config_with_base(self, file_path):
        """Load config from file.

        Args:
            file_path (str): Path of the config file to be loaded.

        Returns: global config
        """
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], 'only support yaml files for now'

        with open(file_path) as f:
            file_cfg = yaml.load(f, Loader=yaml.Loader)

        # NOTE: cfgs outside have higher priority than cfgs in _BASE_
        if self.BASE_KEY in file_cfg:
            all_base_cfg = AttrDict()
            base_ymls = list(file_cfg[self.BASE_KEY])
            for base_yml in base_ymls:
                base_yml = _expand_env(base_yml)
                if base_yml.startswith('~'):
                    base_yml = os.path.expanduser(base_yml)
                if not base_yml.startswith('/'):
                    base_yml = os.path.join(os.path.dirname(file_path),
                                            base_yml)

                with open(base_yml) as f:
                    base_cfg = self._load_config_with_base(base_yml)
                    all_base_cfg = _merge_dict(all_base_cfg, base_cfg)

            del file_cfg[self.BASE_KEY]
            file_cfg = _merge_dict(all_base_cfg, file_cfg)
        file_cfg['filename'] = os.path.splitext(
            os.path.split(file_path)[-1])[0]
        return file_cfg

    def merge_dict(self, args):
        # Remember whether the caller pinned lr explicitly; if so the trainer
        # must not touch it (an explicit value is taken as already scaled).
        if self._sets_optimizer_lr(args):
            self._lr_explicit = True
        self.cfg = _merge_dict(self.cfg, args)
        if self._yml_total_bs:
            self.cfg.setdefault('Global', {})['lr_explicit'] = self._lr_explicit

    def print_cfg(self, print_func=print):
        """Recursively visualize a dict and indenting acrrording by the
        relationship of keys."""
        print_func('----------- Config -----------')
        print_dict(self.cfg, print_func)
        print_func('---------------------------------------------')

    def save(self, p, cfg=None):
        if cfg is None:
            cfg = self.cfg
        with open(p, 'w') as f:
            yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)
