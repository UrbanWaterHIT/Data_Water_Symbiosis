#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cooling-energy allocation (vectorized + OR-Tools min-cost flow, auto-compat with new/old APIs + bidirectional slack)
--------------------------------------------------------------------------------
Dependencies:
    pip install numpy pandas openpyxl "ortools==9.13.*"

Changes in this version:
  • "Allocations" sheet: add "Satisfaction (%)"
  • "Unmet demand" sheet: add "Satisfaction (%)"
  • "Remaining supply" sheet: add "Remaining (%)"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 0. OR-Tools dynamic import (compat with ≥9.10 pybind and ≤9.9 SWIG APIs)
# ------------------------------------------------------------
try:  # ≥ 9.10  pybind API
    from ortools.graph.python import min_cost_flow as mcf_mod

    PYBIND_API = True
except ImportError:  # ≤ 9.9  SWIG API
    from ortools.graph import pywrapgraph as mcf_mod

    PYBIND_API = False


# ------------------------------------------------------------
# 1. Unified wrapper: hide API diffs and expose required methods
# ------------------------------------------------------------
class MCF:
    """Lightweight wrapper around SimpleMinCostFlow, mapping API names automatically."""

    def __init__(self):
        self._mcf = mcf_mod.SimpleMinCostFlow()

        # Method mapping (pybind vs. SWIG)
        self._add_arc = (
            self._mcf.add_arc_with_capacity_and_unit_cost
            if PYBIND_API
            else self._mcf.AddArcWithCapacityAndUnitCost
        )
        self._set_supply = (
            self._mcf.set_node_supply if PYBIND_API else self._mcf.SetNodeSupply
        )
        self._flow = self._mcf.flow if PYBIND_API else self._mcf.Flow
        self._tail = self._mcf.tail if PYBIND_API else self._mcf.Tail
        self._head = self._mcf.head if PYBIND_API else self._mcf.Head
        self._num_arcs = self._mcf.num_arcs if PYBIND_API else self._mcf.NumArcs
        self._optimal_cost = (
            self._mcf.optimal_cost if PYBIND_API else self._mcf.OptimalCost
        )

        self._solve_impl = self._mcf.solve if PYBIND_API else self._mcf.Solve
        self._OPTIMAL = self._mcf.OPTIMAL
        self._last_status: int | None = None

    # ---------- Basic interface ----------
    def add_arc(self, tail: int, head: int, cap: int, cost: int) -> None:
        self._add_arc(tail, head, cap, cost)

    def set_supply(self, node: int, amount: int) -> None:
        self._set_supply(node, amount)

    def solve(self) -> bool:
        self._last_status = self._solve_impl()
        return self._last_status == self._OPTIMAL

    def status(self) -> int | None:  # keep camelCase for API symmetry
        return self._last_status

    # ---------- Results ----------
    def optimal_cost(self) -> int:
        return self._optimal_cost()

    def num_arcs(self) -> int:
        return self._num_arcs()

    def flow(self, i: int) -> int:
        return self._flow(i)

    def tail(self, i: int) -> int:
        return self._tail(i)

    def head(self, i: int) -> int:
        return self._head(i)


# ------------------------------------------------------------
# 2. Global constants
# ------------------------------------------------------------
PUE_FACTOR = 1          # load → IT power PUE correction
HOURS_PER_YEAR = 8_760
DISTANCE_THRESHOLD_KM = 107  # None ⇒ no distance limit

COST_FACTOR = 1  # cost per km (integerized 1:1)
SCALE = 1        # scale factor for cooling energy; e.g., use 1_000 for MWh→GWh

BASE_DIR = Path()
RESULT_SUBDIR = (  )  # fill as needed, e.g., Path("results")

countries = {
    'ESP'
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DC_COLS: Dict[str, str] = {
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Demand": "demand",
    "ID1": "DC_ID",
}
SP_COLS: Dict[str, str] = {
    "LAT_WWTP": "latitude",
    "LON_WWTP": "longitude",
    "Cool energy potential": "supply",  # keep exact source column name
    "WASTE_ID": "SP_ID",
}


# ------------------------------------------------------------
# 3. Haversine (vectorized)
# ------------------------------------------------------------
def haversine_np(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km) between two coordinate sets.
    Returns an array with shape (len(lat1), len(lat2)).
    """
    R = 6_371.0  # Earth mean radius (km)
    lat1_r, lon1_r = np.radians(lat1)[:, None], np.radians(lon1)[:, None]
    lat2_r, lon2_r = np.radians(lat2)[None, :], np.radians(lon2)[None, :]
    dlat, dlon = lat1_r - lat2_r, lon1_r - lon2_r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------
# 4. Process each country independently
# ------------------------------------------------------------
def process_country(code: str) -> None:
    """Read Excel, build & solve min-cost flow, and write outputs."""

    logging.info("▶ Processing %s", code)

    dc_path = BASE_DIR / "0728D" / f"D_{code}_data.xlsx"
    sp_path = BASE_DIR / "0713W" / f"W_{code}_data.xlsx"
    out_dir = BASE_DIR / RESULT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"cooling_allocation_{code}.xlsx"

    if not dc_path.exists() or not sp_path.exists():
        logging.warning("Missing data files, skipping %s", code)
        return

    # ---------- Read data ----------
    dcs = pd.read_excel(dc_path)[list(DC_COLS)].rename(columns=DC_COLS)
    sps = pd.read_excel(sp_path)[list(SP_COLS)].rename(columns=SP_COLS)

    # Load → annual cooling energy (MWh) → /SCALE
    dcs["demand"] = dcs["demand"] * PUE_FACTOR * HOURS_PER_YEAR / SCALE
    sps["supply"] = sps["supply"] / SCALE

    # ---------- Distance matrix & valid edges ----------
    dist = haversine_np(
        sps.latitude.values, sps.longitude.values, dcs.latitude.values, dcs.longitude.values
    )
    mask = (
        dist <= DISTANCE_THRESHOLD_KM
        if DISTANCE_THRESHOLD_KM is not None
        else np.ones_like(dist, bool)
    )
    sp_idx, dc_idx = np.nonzero(mask)
    logging.info("Valid edges: %d", len(sp_idx))

    n_sp, n_dc = len(sps), len(dcs)
    SP0, DC0, SLACK = 0, n_sp, n_sp + n_dc  # node offsets

    # OR-Tools uses integers only
    supply = np.rint(sps.supply.values).astype("int64")
    demand = np.rint(dcs.demand.values).astype("int64")

    mcf = MCF()

    # ---------- SP → DC ----------
    for k in range(len(sp_idx)):
        mcf.add_arc(
            SP0 + int(sp_idx[k]),
            DC0 + int(dc_idx[k]),
            int(supply[sp_idx[k]]),
            int(dist[sp_idx[k], dc_idx[k]] * COST_FACTOR),
        )

    # ---------- Bidirectional slack ----------
    DUMP_COST = 0  # no penalty for dumping leftover supply
    PENALTY = int(np.ceil(dist.max() * COST_FACTOR)) * 1_000  # unmet demand penalty

    for sp in range(n_sp):
        mcf.add_arc(SP0 + sp, SLACK, int(supply[sp]), DUMP_COST)
    for dc in range(n_dc):
        mcf.add_arc(SLACK, DC0 + dc, int(demand[dc]), PENALTY)

    # Slack node net supply
    net = supply.sum() - demand.sum()  # positive = surplus; negative = deficit
    mcf.set_supply(SLACK, -int(net))

    # Regular node supplies
    for sp, s in enumerate(supply):
        mcf.set_supply(SP0 + sp, int(s))
    for dc, d in enumerate(demand):
        mcf.set_supply(DC0 + dc, -int(d))

    # ---------- Solve ----------
    logging.info("Total supply: %d   Total demand: %d", supply.sum(), demand.sum())

    ok = mcf.solve()
    logging.info(
        "Solve ok? %s   raw status = %s (0=OPTIMAL,1=INFEASIBLE,2=UNBALANCED)",
        ok,
        mcf.status(),
    )
    if not ok:
        logging.error("Solve failed for %s; aborting this country", code)
        return

    logging.info("Optimal cost: %s", mcf.optimal_cost())

    # ---------- Collect results ----------
    alloc: List[Dict[str, Any]] = []
    unmet: List[Dict[str, Any]] = []
    leftover = supply.astype(float).copy()
    fulfilled = np.zeros(n_dc, dtype=int)  # actual cooling received per DC

    for i in range(mcf.num_arcs()):
        f = mcf.flow(i)
        if f == 0:
            continue
        tail, head = mcf.tail(i), mcf.head(i)

        if tail < DC0 and DC0 <= head < SLACK:  # SP ➜ DC normal allocation
            sp = tail - SP0
            dc = head - DC0
            alloc.append(
                {
                    "WWTP_ID": sps.SP_ID.iat[sp],
                    "DC_ID": dcs.DC_ID.iat[dc],
                    "Allocated cooling (MWh/yr)": f,
                    "Distance (km)": dist[sp, dc],
                }
            )
            leftover[sp] -= f
            fulfilled[dc] += f

        elif tail == SLACK and DC0 <= head < SLACK:  # unmet demand (SLACK ➜ DC)
            dc = head - DC0
            unmet_amount = f
            total_demand = demand[dc]
            sats_pct = round((total_demand - unmet_amount) / total_demand * 100, 2)
            unmet.append(
                {
                    "DC_ID": dcs.DC_ID.iat[dc],
                    "Unmet demand (MWh/yr)": unmet_amount,
                    "Satisfaction (%)": sats_pct,
                }
            )

    # Remaining supply (SP ➜ SLACK)
    remain: List[Dict[str, Any]] = []
    for i, v in enumerate(leftover):
        if v > 0:
            remain.append(
                {
                    "WWTP_ID": sps.SP_ID.iat[i],
                    "Remaining supply (MWh/yr)": v,
                    "Remaining (%)": round(v / supply[i] * 100, 2),
                }
            )

    # ---------- Write outputs ----------
    alloc_df = pd.DataFrame(alloc)
    if not alloc_df.empty:
        sats_map = {
            dcs.DC_ID.iat[i]: round(fulfilled[i] / demand[i] * 100, 2)
            for i in range(n_dc)
        }
        alloc_df["Satisfaction (%)"] = alloc_df["DC_ID"].map(sats_map)

    unmet_df = pd.DataFrame(unmet)
    remain_df = pd.DataFrame(remain)

    with pd.ExcelWriter(out_xlsx) as xw:
        alloc_df.to_excel(xw, sheet_name="Allocations", index=False)
        unmet_df.to_excel(xw, sheet_name="Unmet demand", index=False)
        remain_df.to_excel(xw, sheet_name="Remaining supply", index=False)

    logging.info("Saved results to %s", out_xlsx)


# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    logging.info("========== Batch cooling allocation started ==========")
    for c in sorted(countries):
        try:
            process_country(c)
        except Exception as e:
            logging.exception("Exception while processing %s: %s", c, e)
    logging.info("========== All countries processed ==========")
