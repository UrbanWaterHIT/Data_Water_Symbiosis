---

# Data‑Water Infrastructure Symbiosis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.28622486.svg)](https://doi.org/10.6084/m9.figshare.28622486)

A research codebase and reproducible demo for coupling data center cooling with water infrastructure, exposure estimation, and inequality metrics.

---

## Table of contents

1. [Overview](#overview)
2. [System requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick start (Demo)](#quick-start-demo)
5. [Use on your own data](#use-on-your-own-data)
6. [Module‑level details](#module-level-details)

   * [1) Data center ↔ WWTP matching](#1-data-center-wwtp-matching)
   * [2) Noise‑affected population](#2-noise-affected-population)
   * [3) Inequality via Gini](#3-inequality-via-gini)
7. [Configuration examples](#configuration-examples)
8. [Reproduce paper artifacts](#reproduce-paper-artifacts)
9. [Troubleshooting & FAQ](#troubleshooting--faq)
10. [Contributing](#contributing)
11. [License & citation](#license--citation)
12. [Contact](#contact)

---

## Overview

### 1) Data center ↔ WWTP matching (uses `requirements_1.txt`)

This module performs country‑by‑country allocation of treated water cooling supply from **wastewater treatment plants (WWTPs)** to **data centers** cooling demand via a minimum CO₂ model (OR‑Tools SimpleMinCostFlow).

* **Inputs (per country)**: two tables — WWTP supply nodes and data center demand nodes — each with `data_center_ID, Latitude, Longitude, Cooling_energy_demand (MWh yr-1)` (see schema below).
* **Distance filter**: great‑circle distances computed by vectorized Haversine; links allowed only if `distance_km ≤ max_link_km`.
* **Objective**: minimize ∑_{i,j} (E_ij × d_ij) × f_ij, where E_ij is per-unit energy on arc i→j (e.g., MWh) and d_ij is distance. This is a pure energy×distance objective;
* **Slack handling**: a super‑sink/source pair allows **unused supply** (zero cost) and **unmet demand** (large penalty), ensuring feasibility under imbalance.
* **Outputs**: per‑country Excel workbook summarizing

  1. optimal link allocations and each data center satisfied share,
  2. unmet demand with satisfaction percentages, and
  3. residual supply with remaining percentages. Loads are converted to annual cooling energy using a configurable  and **8,760 h·yr⁻¹**.

### 2) Noise‑affected population estimation (uses `requirements_2.txt`)

Indexes population coordinates with a **BallTree (Haversine metric)** and queries circular buffers around point sources (e.g., facilities) with per‑source radii.

* **De‑duplication**: sources processed in descending radius with a boolean mask so each population cell is attributed **at most once**.
* **Outputs**: per‑source affected population counts and a global CSV mask flagging whether each population cell is affected.

### 3) Inequality metrics via Gini (uses `requirements_2.txt`)

Computes population‑weighted per‑capita values at ADM1 from a GID_2 table and evaluates **Gini** before/after (e.g., scenario datacenter Pre→datacenter Post) via a weighted Lorenz–trapezoid implementation.

* **Outputs**: `Gini_Pre`, `Gini_Post`, `Gini_Diff` per country, with diagnostic logs for edge cases (single‑ADM1, zero/NA weights).

---

## System requirements

* **OS**: Windows 10/11, or Ubuntu 20.04/22.04
* **CPU & RAM**: ≥ 8 cores, 16–32 GB RAM recommended for country‑scale runs
* **Python**: 3.10–3.11 (see `requirements_1.txt`, `requirements_2.txt`, `requirements_1.yml`, or `requirements_2.yml`)
* **Tested on**: Python 3.10 + Ubuntu 22.04

> Tip: On Windows, installing via **conda** is often smoother for geopandas/pyproj dependencies.

---

## Installation

Typical install time on a normal desktop: **~15 minutes**.

### Option A — pip

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
# Core library in editable mode
pip install -e .
# Module‑specific deps (choose what you need)
pip install -r requirements_1.txt   # matching
pip install -r requirements_2.txt   # population & gini
```

### Option B — conda

```bash
conda env create -f requirements_1.yml
conda activate requirements_1
conda env create -f requirements_2.yml
conda activate requirements_2
```

---

## Quick start (Demo)

A tiny demo dataset is provided under `demo/data/`. You can also use the **Spain demo** below to exercise all three modules end‑to‑end.

### Spain demo — end‑to‑end (Modules 1–3)

**Files (already laid out under `demo/`):**

* `demo/config_spain.yaml` — parameters for Data center↔WWTP matching
* `demo/data/spain/wwtp_supply.csv`
* `demo/data/spain/datacenter_demand.csv`
* `demo/data/spain/sources_spain.csv` (noise buffers; sources are the Datacenters)
* `demo/data/population/global_population_demo.csv` (lightweight global points)
* `demo/data/inequality/adm2_metrics_spain.csv`

> Schemas
>
> * `wwtp_supply.csv`: `WASTE_ID, Latitude, Longitude, WASTE_DIS (m3 d-1)`
> * `datacenter_demand.csv`: `data_center_ID, Latitude, Longitude, Total_power(MW), Cooling_energy_demand (MWh yr-1)`
> * `sources_spain.csv`: `src_id, Latitude, Longitude, radius_m`
> * `global_population_demo.csv`: `Latitude, Longitude, GID_0, Population`
> * `adm2_metrics_spain.csv`: `GID_2, ADM2_name, Population, Datacenter_pre_kwh_per_cap, Datacenter_post_kwh_per_cap`

#### 1) Run Data center ↔ WWTP matching (minimum‑carbon proxy)

```bash
python -m datawater.match \
  --config demo/config_spain.yaml \
  --out demo/output
```

**Outputs** (under `demo/output/`):

* `links.xlsx` (or workbook): optimal allocations per WWTP–Datacenter pair
* `datacenter_summary.csv`: `data_center_ID, demand_m3_per_yr, served_m3_per_yr, pct_served`
* `wwtp_summary.csv`: `WASTE_ID, supply_m3_per_yr, used_m3_per_yr, pct_used`

#### 2) Run noise‑affected population

```bash
python -m datawater.noise \
  --sources demo/data/spain/sources_spain.xlsx:Sources \
  --pop demo/data/population/global_population_demo.xlsx:Population \
  --out demo/output
```

**Outputs**:

* `affected_by_source.csv`: population totals within each source’s radius
* `population_mask.csv`: `Latitude, Longitude, affected_bool`

#### 3) Compute inequality (Gini) at ADM1

```bash
python -m datawater.gini \
  --adm2 demo/data/inequality/adm2_metrics_spain.xlsx:ADM2 \
  --out demo/output
```

**Outputs**:

* `gini_summary.csv`: `ISO3, Gini_Pre, Gini_Post, Gini_Diff`
* `plots/` with bar charts where |ΔGini| exceeds a threshold

**Expected runtime**: ~20 minutes on a normal desktop.

---

## Use on your own data

```python
from datawater import run_pipeline
run_pipeline("path/to/your.csv", "path/to/out_dir")
```

**Input CSV** must contain columns: `data_center_ID, Latitude, Longitude, Total_power, Cooling_energy_demand` (see module schemas below). Outputs include `summary.csv` with metrics and a `run.log`.

---

## Module‑level details

### 1) Data center ↔ WWTP matching

**Input schemas**

* `wwtp_supply.csv` — `WASTE_ID, Latitude, Longitude, WASTE_DIS (m3 d-1), Cooling_energy_potential (MWh yr-1)`
* `datacenter_demand.csv` — `data_center_ID, Latitude, Longitude, Total_power, Cooling_energy_demand`

**Key parameters**

* `max_link_km` (default: 107)
* `unmet_demand_penalty` (large positive)
* `unused_supply_penalty` (usually 0)
* `hours_per_year` (default: 8760)

**CLI**

```bash
python -m datawater.match \
  --supply data/<ISO3>/wwtp_supply.csv \
  --demand data/<ISO3>/datacenter_demand.csv \
  --out out/<ISO3> \
  --max-link-km 107
```

**Outputs** (Excel workbook)

* `links`: `WASTE_ID, data_center_ID, distance_km, flow_m3_per_yr, cost`
* `datacenter_summary`: `data_center_ID, demand_m3_per_yr, served_m3_per_yr, pct_served`
* `wwtp_summary`: `WASTE_ID, supply_m3_per_yr, used_m3_per_yr, pct_used`

### 2) Noise‑affected population

**Inputs**

* `sources.csv` — `src_id, Latitude, Longitude, radius_m`
* `population.parquet/csv` — `Latitude, Longitude, Population, GID_0`

**CLI**

```bash
python -m datawater.noise \
  --sources data/<ISO3>/sources.csv \
  --pop data/population/<ISO3>.parquet \
  --out out/<ISO3>
```

**Outputs**

* `affected_by_source.csv` — per‑source totals
* `population_mask.csv` — `cell_id, affected_bool`

### 3) Inequality via Gini

**Inputs**

* `adm2_metrics.csv` — `GID_2, ADM2_name, Population, Datacenter Pre, Datacenter Post`

**CLI**

```bash
python -m datawater.gini \
  --adm2 data/<ISO3>/adm2_metrics.csv \
  --out out/<ISO3>
```

**Outputs**

* `gini_summary.csv` — `ISO3, Gini_Pre, Gini_Post, Gini_Diff`
* `plots/` — bar charts for countries with |ΔGini| above a threshold

---

## Configuration examples

Minimal YAML for matching:

```yaml
# demo/config_demo.yaml
country: Spain
max_link_km: 107
hours_per_year: 8760
unmet_demand_penalty: 1e6
unused_supply_penalty: 0
paths:
  supply: demo/data/spain/wwtp_supply.xlsx:Wastewater
  demand: demo/data/spain/datacenter_demand.xlsx:DataCenters
  outdir: demo/output
```

---

## Reproduce paper artifacts

```bash
bash scripts/reproduce_figs.sh
```

Creates `results/` with figure data and panels (placeholder script for now).

---

## Troubleshooting & FAQ

* **ImportError (geopandas/pyproj)** → Prefer `conda` install or ensure geopandas is present.
* **Memory pressure on big countries** → Run per‑country; increase RAM; for BallTree set `batch_size`.
* **Coordinate systems** → All lat/lon should be **WGS84 (EPSG:4326)**; BallTree radii in **radians**.
* **Reproducibility** → Set `PYTHONHASHSEED=0` and pin versions via `environment.yml`.

---

## Contributing

PRs welcome! Please:

1. Run `ruff`/`black` on code and `pytest -q` before submitting.
2. Add/extend unit tests for new features.
3. Update `CHANGELOG.md` and docs.

---

## License & citation

* **License**: MIT (see [LICENSE](LICENSE))
* **Citation**: see [CITATION.cff](CITATION.cff) or cite the DOI above.

**BibTeX**

```bibtex
@misc{datawater2025,
  title        = {Data–Water Infrastructure Symbiosis},
  author       = {Congchao Zhang},
  year         = {2025},
  doi          = {10.6084/m9.figshare.28622486},
  url          = {https://doi.org/10.6084/m9.figshare.28622486}
}
```

---

## Contact

Maintainer: <Congchao Zhang> ([congchaozhang@stu.hit.edu.cn](mailto:congchaozhang@stu.hit.edu.cn))
Repository: [https://github.com/UrbanWaterHIT/](https://github.com/UrbanWaterHIT/)

---



