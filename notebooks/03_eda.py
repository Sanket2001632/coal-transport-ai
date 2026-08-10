import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

file_path = "data/raw/coal_transport_data_100.csv"

df = pd.read_csv(file_path)

# Convert Date from string to datetime
df["Date"] = pd.to_datetime(df["Date"])

print("Dataset loaded successfully.")

# ---------------------------------------------------------
# 1. Delay by Weather
# ---------------------------------------------------------

weather_delay = (
    df.groupby("Weather")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY WEATHER ==========")
print(weather_delay * 100)

# ---------------------------------------------------------
# 2. Delay by Traffic
# ---------------------------------------------------------

traffic_delay = (
    df.groupby("Traffic_Level")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY TRAFFIC ==========")
print(traffic_delay * 100)

# ---------------------------------------------------------
# 3. Delay by Mine
# ---------------------------------------------------------

mine_delay = (
    df.groupby("Mine")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY MINE ==========")
print(mine_delay * 100)

# ---------------------------------------------------------
# 4. Delay by Destination
# ---------------------------------------------------------

destination_delay = (
    df.groupby("Destination")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY DESTINATION ==========")
print(destination_delay * 100)

# ---------------------------------------------------------
# 5. Average fuel consumption by mine
# ---------------------------------------------------------

fuel_by_mine = (
    df.groupby("Mine")["Fuel_Used_Liters"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== AVERAGE FUEL BY MINE ==========")
print(fuel_by_mine)

# ---------------------------------------------------------
# 6. Average loading time by mine
# ---------------------------------------------------------

loading_by_mine = (
    df.groupby("Mine")["Loading_Time_Min"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== AVERAGE LOADING TIME BY MINE ==========")
print(loading_by_mine)

# ---------------------------------------------------------
# 7. Visualization - Delay by Weather
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    x=weather_delay.index,
    y=weather_delay.values
)

plt.title("Delay Rate by Weather")
plt.xlabel("Weather")
plt.ylabel("Delay Rate")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. Visualization - Delay by Traffic
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    x=traffic_delay.index,
    y=traffic_delay.values
)

plt.title("Delay Rate by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Delay Rate")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Distance vs Fuel
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Distance_km",
    y="Fuel_Used_Liters",
    hue="Delay"
)

plt.title("Distance vs Fuel Consumption")
plt.xlabel("Distance (km)")
plt.ylabel("Fuel Used (Liters)")

plt.tight_layout()
plt.show()