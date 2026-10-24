# -*- coding: utf-8 -*-
"""Top-level CLI re-export (``python -m kdaforge ...`` or ``kdaforge``)."""
from .serving.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
