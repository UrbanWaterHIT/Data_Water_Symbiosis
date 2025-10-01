#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI wrapper for: WWTP+DClujingjisuan20250622all00USA000808shangchuan.py
This script reads the original script, applies argparse-provided variable overrides
to its source (via regex on top-level assignments), then executes the modified code.
It leaves the original file unchanged, providing a reviewer-friendly CLI.
"""
from __future__ import annotations

import argparse, ast, logging, re, sys
from pathlib import Path

ORIGINAL = Path(r"/mnt/data/WWTP+DClujingjisuan20250622all00USA000808shangchuan.py")

def _parse_set_values(values: str):
    # Accept list like "ESP,USA" -> {'ESP','USA'}, or a Python literal like "{'ESP','USA'}"
    values = values.strip()
    try:
        # try Python literal (set/list/tuple)
        lit = ast.literal_eval(values)
        if isinstance(lit, (set, list, tuple)):
            return "{{{}}}".format(",".join(repr(str(v)) for v in lit))
        elif isinstance(lit, str):
            return repr(lit)
        else:
            return repr(lit)
    except Exception:
        # fallback: comma-separated
        parts = [p.strip() for p in values.split(",") if p.strip()]
        return "{{{}}}".format(",".join(repr(p) for p in parts))

def _repr_value(v: str, as_code: bool = False):
    if as_code:
        return v
    # Try to parse as Python literal for proper typing (int/float/bool/None/list/dict/...)
    try:
        lit = ast.literal_eval(v)
        return repr(lit)
    except Exception:
        return repr(v)

def apply_overrides(src: str, overrides: dict[str, str]) -> str:
    # Replace top-level assignments: var = ...
    for var, rhs_code in overrides.items():
        pattern = rf"^(\s*){var}\s*=.*$"
        repl = rf"\1{var} = {rhs_code}"
        src_new, n = re.subn(pattern, repl, src, flags=re.MULTILINE)
        if n == 0:
            logging.warning("Variable '%s' not found for override.", var)
        src = src_new
    return src

def main():
    parser = argparse.ArgumentParser(prog="wwtp_dc_mcf_cli.py", description="CLI for WWTP+DClujingjisuan20250622all00USA000808shangchuan.py")

    parser.add_argument("--dc-path", help="Path to data center input file (e.g., Excel/CSV)")
    parser.add_argument("--wwtp-path", help="Path to WWTP input file (e.g., Excel/CSV)")
    parser.add_argument("--outdir", help="Directory to write results")
    parser.add_argument("--countries", help="Comma-separated ISO3 list; overrides COUNTRIES/countries")
    parser.add_argument("--distance-km", type=float, help="Max assignment distance in kilometers")
    parser.add_argument("--unmet-penalty", type=float, help="Penalty cost per unit unmet demand")
    parser.add_argument("--result-subdir", help="Result subdir name (replaces RESULT_SUBDIR if present)")

    parser.add_argument("--set", metavar="VAR=VALUE", action="append",
                        help="Additional overrides, repeatable. VALUE can be a Python literal or plain string.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    if not ORIGINAL.exists():
        logging.error("Original script not found: %s", ORIGINAL)
        sys.exit(1)

    src = ORIGINAL.read_text(encoding="utf-8")

    overrides = {}

    if args.dc_path:
        overrides["DC_PATH"] = _repr_value(args.dc_path)
        overrides["dc_path"] = _repr_value(args.dc_path)
    if args.wwtp_path:
        overrides["WWTP_PATH"] = _repr_value(args.wwtp_path)
        overrides["wwtp_path"] = _repr_value(args.wwtp_path)
    if args.outdir:
        overrides["RESULT_DIR"] = _repr_value(args.outdir)
        overrides["OUT_DIR"] = _repr_value(args.outdir)
    if args.result_subdir:
        overrides["RESULT_SUBDIR"] = _repr_value(args.result_subdir)
    if args.countries:
        cn_set = _parse_set_values(args.countries)
        overrides["COUNTRIES"] = cn_set
        overrides["countries"] = cn_set
    if args.distance_km is not None:
        overrides["MAX_DISTANCE_KM"] = _repr_value(str(args.distance_km))
        overrides["DISTANCE_KM"] = _repr_value(str(args.distance_km))
    if args.unmet_penalty is not None:
        overrides["UNMET_PENALTY"] = _repr_value(str(args.unmet_penalty))
        overrides["PENALTY"] = _repr_value(str(args.unmet_penalty))


    # Generic --set overrides (late to take precedence)
    if args.set:
        for kv in args.set:
            if "=" not in kv:
                logging.error("--set expects VAR=VALUE, got: %s", kv)
                sys.exit(2)
            var, val = kv.split("=", 1)
            overrides[var.strip()] = _repr_value(val.strip(), as_code=False)

    logging.debug("Applying overrides: %s", overrides)

    modified = apply_overrides(src, overrides)

    # Execute the modified script in an isolated global namespace
    g = {"__name__": "__main__", "__file__": str(ORIGINAL)}
    exec(compile(modified, str(ORIGINAL), "exec"), g, g)

if __name__ == "__main__":
    main()
