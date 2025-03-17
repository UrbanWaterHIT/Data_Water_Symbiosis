import pandas as pd
import numpy as np
from geopy.distance import geodesic
import pulp
import os

country = ''
print(f"Processing country: {country}")

base_dir = r'D:\DataStoragePath'
data_center_path = os.path.join(base_dir, 'D', f"D_{country}_data.xlsx")
sewage_plant_path = os.path.join(base_dir, 'W', f"W_{country}_data.xlsx")

if not os.path.exists(data_center_path):
    print(f"Data center file not found: {data_center_path}")
else:
    if not os.path.exists(sewage_plant_path):
        print(f"Sewage treatment plant file not found: {sewage_plant_path}")
    else:
        data_centers = pd.read_excel(data_center_path)
        sewage_plants = pd.read_excel(sewage_plant_path)

        data_centers = data_centers[['125Latitude', '125Longitude', 'buquanhouPW', 'ID1']]
        data_centers.columns = ['latitude', 'longitude', 'demand', 'DC_ID']
        data_centers['demand'] = data_centers['demand'] * 0.4 * 8760

        sewage_plants = sewage_plants[['LAT_WWTP', 'LON_WWTP', 'cool potential mw', 'WASTE_ID']]
        sewage_plants.columns = ['latitude', 'longitude', 'supply', 'SP_ID']

        def calculate_distance(lat1, lon1, lat2, lon2):
            return geodesic((lat1, lon1), (lat2, lon2)).kilometers

        distance_matrix = pd.DataFrame(index=sewage_plants['SP_ID'], columns=data_centers['DC_ID'])

        for sp_id, sp_row in sewage_plants.iterrows():
            for dc_id, dc_row in data_centers.iterrows():
                distance = calculate_distance(
                    sp_row['latitude'], sp_row['longitude'],
                    dc_row['latitude'], dc_row['longitude']
                )
                distance_matrix.at[sp_row['SP_ID'], dc_row['DC_ID']] = distance

        distance_matrix = distance_matrix.astype(float)

        prob = pulp.LpProblem(f"Cold_Energy_Distribution_{country}", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            "allocate",
            ((sp, dc) for sp in sewage_plants['SP_ID'] for dc in data_centers['DC_ID']),
            lowBound=0,
            cat='Continuous'
        )

        prob += pulp.lpSum([
            distance_matrix.at[sp, dc] * x[(sp, dc)]
            for sp in sewage_plants['SP_ID']
            for dc in data_centers['DC_ID']
        ]), "Total_Transportation_Cost"

        for sp in sewage_plants['SP_ID']:
            supply_val = sewage_plants.loc[sewage_plants['SP_ID'] == sp, 'supply'].values[0]
            prob += (
                pulp.lpSum([x[(sp, dc)] for dc in data_centers['DC_ID']]) <= supply_val,
                f"Supply_Constraint_{sp}"
            )

        for dc in data_centers['DC_ID']:
            demand_val = data_centers.loc[data_centers['DC_ID'] == dc, 'demand'].values[0]
            prob += (
                pulp.lpSum([x[(sp, dc)] for sp in sewage_plants['SP_ID']]) >= demand_val,
                f"Demand_Constraint_{dc}"
            )

        prob.solve()

        status = pulp.LpStatus[prob.status]
        print(f"Solution status ({country}): {status}")

        if status in ['Optimal', 'Feasible']:
            allocation = []
            for sp in sewage_plants['SP_ID']:
                for dc in data_centers['DC_ID']:
                    allocated = x[(sp, dc)].varValue
                    if allocated > 0:
                        dist = distance_matrix.at[sp, dc]
                        allocation.append({
                            'Sewage_Plant_ID': sp,
                            'Data_Center_ID': dc,
                            'Allocated_Cold_Energy': allocated,
                            'Distance_km': dist
                        })

            allocation_df = pd.DataFrame(allocation)

            result_path = os.path.join(
                os.path.dirname(data_center_path),
                f"Cold_Energy_Distribution_{country}.xlsx"
            )
            with pd.ExcelWriter(result_path) as writer:
                allocation_df.to_excel(writer, sheet_name='Allocation_Result', index=False)

            print(f"Results saved to: {result_path}")
        else:
            print(f"No feasible allocation found for {country}. Check supply-demand data.")

print("Processing completed for all countries.")
