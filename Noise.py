import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm

EARTH_RADIUS = 6371
KILOMETERS_PER_METER = 0.001

data_centers = pd.read_excel(r'D:\data\datacenters.xlsx', sheet_name='Sheet3')
unique_centers = data_centers.drop_duplicates(subset=['Latitude', 'Longitude', 'Radius']).reset_index(drop=True)

population_data = pd.read_csv(r'E:\data\population_data.csv')

population_coords = np.deg2rad(population_data[['latitude', 'longitude']].values)

population_tree = BallTree(population_coords, metric='haversine')

population_mask = np.ones(len(population_data), dtype=bool)

results = []

unique_centers = unique_centers.sort_values(by='Radius', ascending=False).reset_index(drop=True)

population_data['affected'] = False

for idx, row in tqdm(unique_centers.iterrows(), total=len(unique_centers), desc="Processing Data Centers"):
    center_lat = row['Latitude']
    center_lon = row['Longitude']
    radius_meters = row['Radius']
    radius_radian = radius_meters * KILOMETERS_PER_METER / EARTH_RADIUS
    center_coord = np.deg2rad([[center_lat, center_lon]])

    indices = population_tree.query_radius(center_coord, r=radius_radian)[0]

    valid_indices = indices[population_mask[indices]]

    population_data.loc[valid_indices, 'affected'] = True

    total_population = population_data.iloc[valid_indices]['population'].sum()

    population_mask[valid_indices] = False

    results.append({
        'Longitude': center_lon,
        'Latitude': center_lat,
        'Radius': radius_meters,
        'Affected Population': total_population
    })

final_result = data_centers.merge(
    pd.DataFrame(results),
    on=['Longitude', 'Latitude', 'Radius'],
    how='left'
)

final_result_path = r'E:\data\final_result.xlsx'
final_result.to_excel(final_result_path, index=False)

affected_population_path = r'E:\data\affected_population.csv'
population_data.to_csv(affected_population_path, index=False)

print(f"Final results saved to: {final_result_path}")
print(f"Affected population data saved to: {affected_population_path}")
