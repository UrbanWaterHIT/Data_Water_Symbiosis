import pandas as pd
import geopy.distance

# Load Excel file
data = pd.read_excel('USA.xlsx')

# Function to calculate distance between two latitude-longitude points
def calculate_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geopy.distance.distance(coords_1, coords_2).km

# Function to calculate population within a given radius
def calculate_population_within_radius(center_lat, center_lon, radius, population_chunk, processed_points):
    total_population = 0
    for _, row in population_chunk.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        population = row['population']
        
        if (latitude, longitude) in processed_points:
            continue
        
        distance = calculate_distance(center_lat, center_lon, latitude, longitude)
        
        if distance <= radius * 0.001:
            total_population += population
            processed_points.add((latitude, longitude))
    
    return total_population

# Process each data center
population_results = []
chunk_size = 1000000
processed_points = set()

for index, row in data.iterrows():
    longitude_center = row['Longitude']
    latitude_center = row['Latitude']
    radius = row['Radius']
    
    print(f"Processing Data Center {index + 1}: Longitude = {longitude_center}, Latitude = {latitude_center}")
    
    total_population = 0
    
    for chunk in pd.read_csv('USA1.csv', chunksize=chunk_size):
        total_population += calculate_population_within_radius(latitude_center, longitude_center, radius, chunk, processed_points)
    
    population_results.append({
        'Longitude': longitude_center,
        'Latitude': latitude_center,
        'Population Within Radius': total_population
    })
    
    data.at[index, 'Population Within Radius'] = total_population
    data.to_excel('USA_updated.xlsx', index=False)
    
    print(f"Data Center {index + 1} processing completed.")

print("All results saved to file: USA_updated.xlsx")
