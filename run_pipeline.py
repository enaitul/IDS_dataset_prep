#!/usr/bin/env python
"""python run_pipeline.py [--datasets NSL-KDD InSDN CICIDS2017] [--steps inspect clean split select export figures report]"""
import argparse
from src.utils import load_config
from src import pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--steps", nargs="*", default=None, choices=pipeline.STEPS + ["figures", "report"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    core = [s for s in (a.steps or pipeline.STEPS) if s in pipeline.STEPS]
    if core:
        pipeline.run(cfg, a.datasets, core)
    if a.steps is None or "figures" in a.steps:
        from src import visualization; visualization.make_all(cfg)
    if a.steps is None or "report" in a.steps:
        from src import reporting; reporting.write_report(cfg)
