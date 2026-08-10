import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of records
n_records = 100

# ---------------------------------------------------------
# 1. Basic identification
# ---------------------------------------------------------

trip_ids = [f"TRIP-{i:05d}" for i in range(1, n_records + 1)]

truck_ids = [
    f"TRK-{np.random.randint(1, 51):03d}"
    for _ in range(n_records)
]

driver_ids = [
    f"DRV-{np.random.randint(1, 101):03d}"
    for _ in range(n_records)
]

# ---------------------------------------------------------
# 2. Mining and destination information
# ---------------------------------------------------------

mines = [
    "Mine_A",
    "Mine_B",
    "Mine_C",
    "Mine_D"
]

destinations = [
    "Power_Plant_A",
    "Power_Plant_B",
    "Railway_Yard_A",
    "Railway_Yard_B"
]

mine_data = np.random.choice(
    mines,
    size=n_records
)

destination_data = np.random.choice(
    destinations,
    size=n_records
)

# ---------------------------------------------------------
# 3. Date
# ---------------------------------------------------------

dates = pd.date_range(
    start="2026-01-01",
    periods=181,
    freq="D"
)

date_data = np.random.choice(
    dates,
    size=n_records
)

# ---------------------------------------------------------
# 4. Transportation information
# ---------------------------------------------------------

distance_data = np.round(
    np.random.uniform(20, 150, n_records),
    2
)

coal_weight_data = np.round(
    np.random.uniform(15, 40, n_records),
    2
)

avg_speed_data = np.round(
    np.random.uniform(25, 65, n_records),
    2
)

# ---------------------------------------------------------
# 5. Fuel consumption
# ---------------------------------------------------------

# Fuel depends partly on distance and coal weight
fuel_data = (
    distance_data * 0.18
    + coal_weight_data * 0.25
    + np.random.normal(0, 2, n_records)
)

fuel_data = np.round(
    np.maximum(fuel_data, 5),
    2
)

# ---------------------------------------------------------
# 6. Loading and unloading
# ---------------------------------------------------------

loading_time_data = np.round(
    np.random.uniform(15, 60, n_records),
    2
)

unloading_time_data = np.round(
    np.random.uniform(10, 40, n_records),
    2
)

# ---------------------------------------------------------
# 7. Weather
# ---------------------------------------------------------

weather_data = np.random.choice(
    ["Clear", "Cloudy", "Rain", "Heavy_Rain"],
    size=n_records,
    p=[0.45, 0.25, 0.20, 0.10]
)

# ---------------------------------------------------------
# 8. Traffic
# ---------------------------------------------------------

traffic_data = np.random.choice(
    ["Low", "Medium", "High"],
    size=n_records,
    p=[0.35, 0.45, 0.20]
)

# ---------------------------------------------------------
# 9. Driver experience
# ---------------------------------------------------------

driver_experience_data = np.random.randint(
    1,
    21,
    n_records
)

# ---------------------------------------------------------
# 10. Previous driver delays
# ---------------------------------------------------------

previous_delays_data = np.random.randint(
    0,
    11,
    n_records
)

# ---------------------------------------------------------
# 11. Safety score
# ---------------------------------------------------------

safety_score_data = np.round(
    np.random.uniform(60, 100, n_records),
    2
)

# ---------------------------------------------------------
# 12. Create delay probability
# ---------------------------------------------------------

delay_score = (
    (distance_data > 100) * 0.15
    + (avg_speed_data < 35) * 0.20
    + (weather_data == "Rain") * 0.15
    + (weather_data == "Heavy_Rain") * 0.30
    + (traffic_data == "High") * 0.20
    + (loading_time_data > 45) * 0.15
    + (previous_delays_data > 5) * 0.15
    + (driver_experience_data < 3) * 0.10
)

# Add random variation
delay_probability = np.clip(
    0.15 + delay_score + np.random.normal(0, 0.05, n_records),
    0.02,
    0.95
)

# Generate target variable
delay_data = np.random.binomial(
    1,
    delay_probability
)

# ---------------------------------------------------------
# 13. Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame({
    "Trip_ID": trip_ids,
    "Truck_ID": truck_ids,
    "Driver_ID": driver_ids,
    "Date": date_data,
    "Mine": mine_data,
    "Destination": destination_data,
    "Distance_km": distance_data,
    "Coal_Weight_Ton": coal_weight_data,
    "Fuel_Used_Liters": fuel_data,
    "Average_Speed_kmh": avg_speed_data,
    "Loading_Time_Min": loading_time_data,
    "Unloading_Time_Min": unloading_time_data,
    "Weather": weather_data,
    "Traffic_Level": traffic_data,
    "Driver_Experience_Years": driver_experience_data,
    "Previous_Delays": previous_delays_data,
    "Safety_Score": safety_score_data,
    "Delay": delay_data
})

# ---------------------------------------------------------
# 14. Sort data by date
# ---------------------------------------------------------

df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------
# 15. Save dataset
# ---------------------------------------------------------

output_path = "data/raw/coal_transport_data_100.csv"

df.to_csv(
    output_path,
    index=False
)

# ---------------------------------------------------------
# 16. Display results
# ---------------------------------------------------------

print("\nDataset created successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDelay distribution:")
print(df["Delay"].value_counts())

print(f"\nDataset saved to: {output_path}")