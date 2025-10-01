#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI wrapper for Data_center_WWTP_matching.py

This wrapper reads the original script, applies command-line overrides to
top-level variables (via regex on assignments), then executes the modified code.
The original file is left unchanged.

Common overrides:
  --countries ESP       -> sets COUNTRIES = {'ESP'}
  --distance-km 107              -> sets DISTANCE_THRESHOLD_KM = 107
  --result-subdir results_v1    -> sets RESULT_SUBDIR = "results_v1"
  --set KEY=VAL                 -> arbitrary override; repeatable

Usage examples:
  python Data_center_WWTP_matching_cli.py --countries ESP --distance-km 107
  python Data_center_WWTP_matching_cli.py --set LOG_LEVEL=20 --set USE_PYBIND=True
"""
from __future__ import annotations

import argparse, ast, logging, re, sys
from pathlib import Path
from typing import Dict

# Default path to the original script
ORIGINAL = Path(r"/mnt/data/Data_center_WWTP_matching.py")

ASSIGN_RE_TMPL = r"^(\s*){var}\s*=\s*.*?$"  # matches a whole-line assignment

def _repr_value(val: str, *, as_code: bool = False) -> str:
    """
    Convert a CLI string into a Python literal source.
    - as_code=True means we assume val is already Python code (e.g., "{'ESP'}")
    """
    if as_code:
        return val
    # booleans
    low = val.lower()
    if low in {"true", "false"}:
        return "True" if low == "true" else "False"
    # ints
    try:
        i = int(val)
        return str(i)
    except ValueError:
        pass
    # floats
    try:
        f = float(val)
        return repr(f)
    except ValueError:
        pass
    # fallback to string literal
    return repr(val)

def _repr_set_from_csv(values: str) -> str:
    """Turn 'ESP' into a Python set literal like {'ESP'}"""
    parts = [p.strip() for p in values.split(",") if p.strip()]
    inner = ",".join(repr(p) for p in parts)
    return "{" + inner + "}"

def apply_overrides(src: str, overrides: Dict[str, str]) -> str:
    """Replace top-level assignments in the source with new values."""
    for var, value_src in overrides.items():
        pattern = re.compile(ASSIGN_RE_TMPL.format(var=re.escape(var)), flags=re.MULTILINE)
        if not pattern.search(src):
            logging.warning("Variable '%s' not found at top level; skipping.", var)
            continue
        replacement = r"\1{var} = {value}".format(var=var, value=value_src)
        src = pattern.sub(replacement, src, count=1)
    return src

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, default=ORIGINAL, help="Path to Data_center_WWTP_matching.py")
    p.add_argument("--countries", type=str, help="CSV like ESP to set COUNTRIES")
    p.add_argument("--distance-km", type=float, help="Set DISTANCE_THRESHOLD_KM (float)")
    p.add_argument("--result-subdir", type=str, help="Set RESULT_SUBDIR (string)")
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

    if args.countries:
        overrides["COUNTRIES"] = _repr_set_from_csv(args.countries)
    if args.distance_km is not None:
        overrides["DISTANCE_THRESHOLD_KM"] = _repr_value(str(args.distance_km))
    if args.result_subdir:
        overrides["RESULT_SUBDIR"] = _repr_value(args.result_subdir)

    # Arbitrary VAR=VALUE pairs
    for kv in (args.sets or []):
        if "=" not in kv:
            p.error(f"--set expects VAR=VALUE, got: {kv}")
        var, value = kv.split("=", 1)
        var = var.strip()
        value = value.strip()
        # VALUE is treated as Python literal
        overrides[var] = _repr_value(value, as_code=True)

    if overrides:
        logging.info("Applying overrides: %s", overrides)
        src = apply_overrides(src, overrides)
    else:
        logging.info("No overrides provided. Running original as-is.")

    g = {"__name__": "__main__", "__file__": str(args.original)}
    exec(compile(src, str(args.original), "exec"), g, g)

if __name__ == "__main__":
    main()
