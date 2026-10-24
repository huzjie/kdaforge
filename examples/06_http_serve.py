# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
from kdaforge.serving.server import run

cfg = load_config()
print("start serving, then try:  curl http://%s:%d/v1/models" % (
    cfg.serving.host, cfg.serving.port))
run(cfg.serving.host, cfg.serving.port)
