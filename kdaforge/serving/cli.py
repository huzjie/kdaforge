# -*- coding: utf-8 -*-
"""Command-line interface: doctor / bench / train / serve / memory."""
import argparse
import json

from ..config import load_config
from ..model import KDAForgeModel


def _cfg(args):
    return load_config(args.config if hasattr(args, "config") else None)


def cmd_doctor(args):
    cfg = _cfg(args)
    model = KDAForgeModel(cfg)
    x = model.embed([0, 1, 2, 3, 5, 8])
    out = model.forward(x)
    full, actual = model.attn.kv_memory(cfg.declared.context_len)
    print("[doctor] declared: %.1fT params / %.0fB active / %d experts top-%d / %d ctx" % (
        cfg.declared.total_params_b / 1000, cfg.declared.active_params_b,
        cfg.declared.n_experts, cfg.declared.top_k, cfg.declared.context_len))
    print("[doctor] hybrid stack: %d layers (%d KDA + %d MLA)" % (
        model.attn.n_layers, cfg.attention.kda_layers, cfg.attention.mla_layers))
    print("[doctor] forward OK: seq=%d out_dim=%d" % (len(out), len(out[0])))
    print("[doctor] KV memory @1M ctx: full=%.3e hybrid=%.3e ratio=%.2fx" % (
        float(full), float(actual), full / actual))
    print("[doctor] OK")


def cmd_memory(args):
    cfg = _cfg(args)
    model = KDAForgeModel(cfg)
    for L in (1024, 65536, 1000000):
        full, actual = model.attn.kv_memory(L)
        print("[memory] L=%-9d full=%.3e hybrid=%.3e ratio=%.1fx" % (
            L, float(full), float(actual), full / actual))


def cmd_bench(args):
    from ..bench import run_all, summarize
    cfg = _cfg(args)
    results = run_all(cfg)
    summarize(results)


def cmd_train(args):
    from ..train.mopd_trainer import train_mopd
    cfg = _cfg(args)
    train_mopd(cfg)


def cmd_serve(args):
    from .server import run
    cfg = _cfg(args)
    run(cfg.serving.host, cfg.serving.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kdaforge", description="KDA hybrid linear attention + sparse MoE")
    ap.add_argument("--config", default=None, help="path to config yaml")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    sub.add_parser("memory").set_defaults(func=cmd_memory)
    sub.add_parser("bench").set_defaults(func=cmd_bench)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("serve").set_defaults(func=cmd_serve)
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
