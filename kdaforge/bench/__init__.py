# -*- coding: utf-8 -*-
"""Benchmark suite: memory, retrieval, load balance, routing recall, speedup."""
_BENCHMARKS = {}


def register_benchmark(name):
    def deco(fn):
        _BENCHMARKS[name] = fn
        return fn
    return deco


def list_benchmarks():
    return sorted(_BENCHMARKS)


def run_all(cfg):
    from . import memory, retrieval, load_balance, routing, speedup  # noqa: F401
    results = {}
    for name in sorted(_BENCHMARKS):
        results[name] = _BENCHMARKS[name](cfg)
    return results


def summarize(results):
    print("=" * 64)
    print("%-16s %10s   %s" % ("benchmark", "value", "note"))
    print("=" * 64)
    for name, res in results.items():
        if isinstance(res, dict):
            for k, v in res.items():
                print("%-16s %10s   %s" % (name, _fmt(v), k))
        else:
            print("%-16s %10s" % (name, _fmt(res)))
    print("=" * 64)


def _fmt(v):
    if isinstance(v, float):
        return "%.4f" % v
    return str(v)
