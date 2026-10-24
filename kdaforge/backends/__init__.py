# -*- coding: utf-8 -*-
"""Inference backends. Five are registered; mock and cpu are fully local and
zero-dependency, openai/vllm/transformers are optional adapters that fall back
to the local engine when their remote dependency is unavailable."""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def list_backends():
    return sorted(_BACKENDS)


def get_backend(name, cfg):
    if name not in _BACKENDS:
        raise KeyError("unknown backend %r (available: %s)" % (name, list_backends()))
    return _BACKENDS[name](cfg)


from . import mock, cpu, openai, vllm, transformers  # noqa: E402,F401
