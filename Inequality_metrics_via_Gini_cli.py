#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI wrapper for Inequality_metrics_via_Gini.py

This wrapper reads the original script, applies command-line overrides to
top-level variables (via regex on assignments), then executes the modified code.
The original file is left unchanged.

Common overrides:
  --data-path /path/to/input.xlsx   -> sets data_path
  --out-dir-fig ./figs              -> sets out_dir_fig
  --out-dir-tab ./tables            -> sets out_dir_tab
  --countries ESP            -> sets COUNTRIES = {'ESP'}
  --set KEY=VAL                     -> arbitrary override; repeatable
"""
from __future__ import annotations

import argparse, logging, re, sys
from pathlib import Path
from typing import Dict

ORIGINAL = Path(r"/mnt/data/Inequality_metrics_via_Gini.py")

ASSIGN_RE_TMPL = r"^(\s*){var}\s*=\s*.*?$"

def _repr_value(val: str, *, as_code: bool = False) -> str:
    if as_code:
        return val
    low = val.lower()
    if low in {"true", "false"}:
        return "True" if low == "true" else "False"
    try:
        return str(int(val))
    except ValueError:
        pass
    try:
        return repr(float(val))
    except ValueError:
        pass
    return repr(val)

def _repr_set_from_csv(values: str) -> str:
    parts = [p.strip() for p in values.split(",") if p.strip()]
    inner = ",".join(repr(p) for p in parts)
    return "{" + inner + "}"

def apply_overrides(src: str, overrides: Dict[str, str]) -> str:
    for var, value_src in overrides.items():
        pattern = re.compile(ASSIGN_RE_TMPL.format(var=re.escape(var)), flags=re.MULTILINE)
        if not pattern.search(src):
            logging.warning("Variable '%s' not found at top level; skipping.", var)
            continue
        replacement = r"\\1{var} = {value}".format(var=var, value=value_src)
        src = pattern.sub(replacement, src, count=1)
    return src

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, default=ORIGINAL, help="Path to Inequality_metrics_via_Gini.py")
    p.add_argument("--data-path", type=str, help="Set data_path")
    p.add_argument("--out-dir-fig", type=str, help="Set out_dir_fig")
    p.add_argument("--out-dir-tab", type=str, help="Set out_dir_tab")
    p.add_argument("--countries", type=str, help="CSV to set COUNTRIES (e.g., ESP)")
    p.add_argument("--set", dest="sets", action="append", default=[],
                   help="Arbitrary overrides like VAR=VALUE; VALUE can be Python literal")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG")
    args = p.parse_args()

    log_level = logging.WARNING - (10 * min(args.verbose, 2))
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if not args.original.exists():
        p.error(f"Original script not found: {args.original}")

    src = args.original.read_text(encoding="utf-8")

    overrides: Dict[str, str] = {}
    if args.data_path:
        overrides["data_path"] = _repr_value(args.data_path)
    if args.out_dir_fig:
        overrides["out_dir_fig"] = _repr_value(args.out_dir_fig)
    if args.out_dir_tab:
        overrides["out_dir_tab"] = _repr_value(args.out_dir_tab)
    if args.countries:
        overrides["COUNTRIES"] = _repr_set_from_csv(args.countries)

    for kv in (args.sets or []):
        if "=" not in kv:
            p.error(f"--set expects VAR=VALUE, got: {kv}")
        var, value = kv.split("=", 1)
        overrides[var.strip()] = _repr_value(value.strip(), as_code=True)

    if overrides:
        logging.info("Applying overrides: %s", overrides)
        src = apply_overrides(src, overrides)
    else:
        logging.info("No overrides provided. Running original as-is.")

    g = {"__name__": "__main__", "__file__": str(args.original)}
    exec(compile(src, str(args.original), "exec"), g, g)

if __name__ == "__main__":
    main()
