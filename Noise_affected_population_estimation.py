#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cold-energy allocation (vectorized + OR-Tools min-cost flow, auto-compatible with new/old APIs + bidirectional slack)
--------------------------------------------------------------------
Dependencies:
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm

For the given COUNTRIES list (ISO3), compute the population affected by each noise source by country.

Assumptions:
- EXCEL_PATH Sheet1 contains columns: CNTRY_ISO, Latitude, Longitude
- POP_PATH CSV contains columns: latitude, longitude, population, GID_0.
"""

# ---------------- Constants ----------------
EARTH_RADIUS = 6371.0  # km
KILOMETERS_PER_METER = 0.001  # m → km

# ---------- File paths (edit as needed) ----------
EXCEL_PATH = r''
POP_PATH = r''
RESULT_XLSX = r''
POP_AFFECTED_CSV = r''

# ---------- Country allowlist ----------
COUNTRIES = {
    'ESP'
}

# ---------------- Load data ----------------
print("Reading noise sources...")
excel_data = pd.read_excel(EXCEL_PATH, sheet_name='')
excel_data = excel_data[excel_data['CNTRY_ISO'].isin(COUNTRIES)].reset_index(drop=True)

print("Reading population data...")
population_df = pd.read_csv(POP_PATH)
required_cols = {'latitude', 'longitude', 'population', 'GID_0'}
missing = required_cols - set(population_df.columns)
if missing:
    raise ValueError(f"Population CSV is missing columns: {missing}")

population_df['affected'] = False  # Global affected flag

# Results list
affected_results = []

# ---------------- Main loop ----------------
for country in sorted(COUNTRIES):
    centers_country = excel_data[excel_data['CNTRY_ISO'] == country]
    if centers_country.empty:
        continue  # No noise sources in this country

    pop_idx = population_df.index[population_df['GID_0'] == country]
    if pop_idx.empty:
        # No population data for this country; still pad zeros for later merge
        for _, c_row in centers_country.iterrows():
            affected_results.append({
                'CNTRY_ISO': country,
                'Longitude': c_row['Longitude'],
                'Latitude': c_row['Latitude'],
                'dis1': c_row['dis1'],
                'dis1 Population': 0
            })
        continue

    # Build BallTree
    pop_coords = np.deg2rad(population_df.loc[pop_idx, ['latitude', 'longitude']].values)
    tree = BallTree(pop_coords, metric='haversine')
    pop_mask = np.ones(len(pop_idx), dtype=bool)

    # Process larger radii first to avoid being overshadowed by smaller circles
    centers_country = centers_country.sort_values('dis1', ascending=False)

    for _, n_row in centers_country.iterrows():
        center_coord = np.deg2rad([[n_row['Latitude'], n_row['Longitude']]])
        radius_rad = n_row['dis1'] * KILOMETERS_PER_METER / EARTH_RADIUS
        idxs = tree.query_radius(center_coord, r=radius_rad)[0]
        valid = idxs[pop_mask[idxs]]

        global_idxs = pop_idx[valid]
        population_df.loc[global_idxs, 'affected'] = True
        total_pop = population_df.loc[global_idxs, 'population'].sum()

        pop_mask[valid] = False  # Update mask

        affected_results.append({
            'CNTRY_ISO': country,
            'Longitude': n_row['Longitude'],
            'Latitude': n_row['Latitude'],
            'dis1': n_row['dis1'],
            'dis1 Population': int(total_pop)
        })

# ---------------- Merge & output ----------------
print("Writing results...")
results_df = pd.DataFrame(affected_results)
final_df = excel_data.merge(
    results_df,
    on=['CNTRY_ISO', 'Longitude', 'Latitude', 'dis1'],
    how='left'
)

final_df.to_excel(RESULT_XLSX, index=False)
population_df.to_csv(POP_AFFECTED_CSV, index=False)

print(f"Done! Results saved to: {RESULT_XLSX}")
print(f"Affected population flags saved to: {POP_AFFECTED_CSV}")
