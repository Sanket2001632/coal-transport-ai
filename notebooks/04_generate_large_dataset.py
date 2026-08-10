import pandas as pd
import numpy as np

# Reproducible random data
np.random.seed(42)

print("Large dataset generator started!")

# Number of transportation trips
N = 10000

print("Number of trips to generate:", N)

# ---------------------------------------------------------
# Generate Trip, Truck and Driver IDs
# ---------------------------------------------------------

trip_ids = [f"TRIP-{i:05d}" for i in range(1, N + 1)]

truck_ids = np.random.choice(
    [f"TRK-{i:03d}" for i in range(1, 101)],
    size=N
)

driver_ids = np.random.choice(
    [f"DRV-{i:03d}" for i in range(1, 201)],
    size=N
)

print("Trip IDs, Truck IDs and Driver IDs generated.")

# ---------------------------------------------------------
# Generate transportation dates
# ---------------------------------------------------------

dates = pd.date_range(
    start="2026-01-01",
    periods=365,
    freq="D"
)

trip_dates = np.random.choice(dates, size=N)

print("Trip dates generated.")


# ---------------------------------------------------------
# Mine and Destination
# ---------------------------------------------------------

mines = [
    "Mine_A",
    "Mine_B",
    "Mine_C",
    "Mine_D"
]

destinations = [
    "Railway_Yard_A",
    "Railway_Yard_B",
    "Power_Plant_A",
    "Power_Plant_B"
]

mine_data = np.random.choice(
    mines,
    size=N,
    p=[0.25, 0.25, 0.25, 0.25]
)

destination_data = np.random.choice(
    destinations,
    size=N,
    p=[0.30, 0.25, 0.25, 0.20]
)

print("Mine and destination data generated.")

# ---------------------------------------------------------
# Distance
# ---------------------------------------------------------

distance_data = np.random.uniform(
    20,
    150,
    size=N
)

print("Distance data generated.")

# ---------------------------------------------------------
# Coal Weight
# ---------------------------------------------------------

coal_weight_data = np.random.uniform(
    15,
    40,
    size=N
)

print("Coal weight data generated.")

# ---------------------------------------------------------
# Weather
# ---------------------------------------------------------

weather_conditions = [
    "Clear",
    "Cloudy",
    "Rain",
    "Heavy_Rain"
]

weather_data = np.random.choice(
    weather_conditions,
    size=N,
    p=[0.45, 0.25, 0.20, 0.10]
)

print("Weather data generated.")

# ---------------------------------------------------------
# Traffic
# ---------------------------------------------------------

traffic_levels = [
    "Low",
    "Medium",
    "High"
]

traffic_data = np.random.choice(
    traffic_levels,
    size=N,
    p=[0.40, 0.40, 0.20]
)

print("Traffic data generated.")

# ---------------------------------------------------------
# Driver Experience
# ---------------------------------------------------------

driver_experience_data = np.random.randint(
    1,
    21,
    size=N
)

print("Driver experience data generated.")

# ---------------------------------------------------------
# Previous Delays
# ---------------------------------------------------------

previous_delays_data = np.random.poisson(
    lam=3,
    size=N
)

# Keep the maximum at 10
previous_delays_data = np.clip(
    previous_delays_data,
    0,
    10
)

print("Previous delay data generated.")

# ---------------------------------------------------------
# Loading Time
# ---------------------------------------------------------

loading_time_data = np.random.normal(
    loc=35,
    scale=8,
    size=N
)

loading_time_data = np.clip(
    loading_time_data,
    15,
    60
)

print("Loading time data generated.")

# ---------------------------------------------------------
# Unloading Time
# ---------------------------------------------------------

unloading_time_data = np.random.normal(
    loc=25,
    scale=5,
    size=N
)

unloading_time_data = np.clip(
    unloading_time_data,
    10,
    40
)

print("Unloading time data generated.")

# ---------------------------------------------------------
# Safety Score
# ---------------------------------------------------------

safety_score_data = np.random.normal(
    loc=82,
    scale=8,
    size=N
)

safety_score_data = np.clip(
    safety_score_data,
    50,
    100
)

print("Safety score data generated.")

# ---------------------------------------------------------
# Average Speed
# ---------------------------------------------------------

base_speed = np.random.normal(
    loc=55,
    scale=7,
    size=N
)

# Weather impact
weather_speed_penalty = np.select(
    [
        weather_data == "Clear",
        weather_data == "Cloudy",
        weather_data == "Rain",
        weather_data == "Heavy_Rain"
    ],
    [
        0,
        5,
        12,
        20
    ],
    default=0
)

# Traffic impact
traffic_speed_penalty = np.select(
    [
        traffic_data == "Low",
        traffic_data == "Medium",
        traffic_data == "High"
    ],
    [
        0,
        7,
        15
    ],
    default=0
)

average_speed_data = (
    base_speed
    - weather_speed_penalty
    - traffic_speed_penalty
)

average_speed_data = np.clip(
    average_speed_data,
    20,
    70
)

average_speed_data = np.round(
    average_speed_data,
    2
)

print("Average speed data generated.")

# ---------------------------------------------------------
# Fuel Consumption
# ---------------------------------------------------------

fuel_data = (
    distance_data * 0.16
    + coal_weight_data * 0.22
    + (average_speed_data < 35) * 2.5
    + (weather_data == "Heavy_Rain") * 2
    + np.random.normal(0, 2, N)
)

fuel_data = np.maximum(
    fuel_data,
    5
)

fuel_data = np.round(
    fuel_data,
    2
)

print("Fuel consumption data generated.")

# ---------------------------------------------------------
# Delay Risk Score
# ---------------------------------------------------------

delay_risk = np.zeros(N)

# Weather impact
delay_risk += np.select(
    [
        weather_data == "Clear",
        weather_data == "Cloudy",
        weather_data == "Rain",
        weather_data == "Heavy_Rain"
    ],
    [
        0.00,
        0.05,
        0.15,
        0.30
    ],
    default=0
)

# Traffic impact
delay_risk += np.select(
    [
        traffic_data == "Low",
        traffic_data == "Medium",
        traffic_data == "High"
    ],
    [
        0.00,
        0.08,
        0.20
    ],
    default=0
)

# Long-distance trips
delay_risk += np.where(
    distance_data > 100,
    0.12,
    0
)

# Long loading time
delay_risk += np.where(
    loading_time_data > 45,
    0.15,
    0
)

# Previous delays
delay_risk += np.where(
    previous_delays_data >= 5,
    0.12,
    0
)

# Less experienced drivers
delay_risk += np.where(
    driver_experience_data <= 3,
    0.08,
    0
)

# Low average speed
delay_risk += np.where(
    average_speed_data < 35,
    0.12,
    0
)

print("Delay risk factors calculated.")

# ---------------------------------------------------------
# Convert Risk Score Into Delay Probability
# ---------------------------------------------------------

delay_probability = (
    0.08
    + delay_risk
)

# Add small random variation
delay_probability += np.random.normal(
    0,
    0.03,
    N
)

# Keep probability between 2% and 95%
delay_probability = np.clip(
    delay_probability,
    0.02,
    0.95
)

print("Delay probabilities calculated.")

# ---------------------------------------------------------
# Generate Delay Target
# ---------------------------------------------------------

delay_data = np.random.binomial(
    1,
    delay_probability,
    N
)

print("Delay target generated.")

# ---------------------------------------------------------
# Create Final DataFrame
# ---------------------------------------------------------

df = pd.DataFrame({
    "Trip_ID": trip_ids,
    "Truck_ID": truck_ids,
    "Driver_ID": driver_ids,
    "Date": trip_dates,
    "Mine": mine_data,
    "Destination": destination_data,
    "Distance_km": np.round(distance_data, 2),
    "Coal_Weight_Ton": np.round(coal_weight_data, 2),
    "Fuel_Used_Liters": fuel_data,
    "Average_Speed_kmh": average_speed_data,
    "Loading_Time_Min": np.round(loading_time_data, 2),
    "Unloading_Time_Min": np.round(unloading_time_data, 2),
    "Weather": weather_data,
    "Traffic_Level": traffic_data,
    "Driver_Experience_Years": driver_experience_data,
    "Previous_Delays": previous_delays_data,
    "Safety_Score": np.round(safety_score_data, 2),
    "Delay": delay_data
})

print("Final DataFrame created.")
print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# Sort by Date
# ---------------------------------------------------------

df = df.sort_values(
    "Date"
).reset_index(drop=True)

print("Dataset sorted by date.")

# ---------------------------------------------------------
# Save Dataset
# ---------------------------------------------------------

output_path = "data/raw/coal_transport_data_10000.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"Dataset saved successfully to: {output_path}")

# ---------------------------------------------------------
# Basic Quality Check
# ---------------------------------------------------------

print("\n========== FINAL DATASET CHECK ==========")

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDelay distribution:")
print(df["Delay"].value_counts())

print("\nDelay percentage:")
print(df["Delay"].value_counts(normalize=True) * 100)

print("\nFirst 5 rows:")
print(df.head())