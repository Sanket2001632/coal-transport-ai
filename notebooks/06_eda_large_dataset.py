import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

file_path = "data/raw/coal_transport_data_10000.csv"

df = pd.read_csv(file_path)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

print("Large dataset loaded successfully.")
print("Shape:", df.shape)

# ---------------------------------------------------------
# Delay Rate by Weather
# ---------------------------------------------------------

weather_delay = (
    df.groupby("Weather")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY WEATHER ==========")
print(weather_delay * 100)


# ---------------------------------------------------------
# Delay Rate by Traffic
# ---------------------------------------------------------

traffic_delay = (
    df.groupby("Traffic_Level")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY TRAFFIC ==========")
print(traffic_delay * 100)


# ---------------------------------------------------------
# Delay Rate by Mine
# ---------------------------------------------------------

mine_delay = (
    df.groupby("Mine")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY MINE ==========")
print(mine_delay * 100)


# ---------------------------------------------------------
# Delay Rate by Destination
# ---------------------------------------------------------

destination_delay = (
    df.groupby("Destination")["Delay"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== DELAY RATE BY DESTINATION ==========")
print(destination_delay * 100)

# ---------------------------------------------------------
# Fuel Consumption by Mine
# ---------------------------------------------------------

fuel_by_mine = (
    df.groupby("Mine")["Fuel_Used_Liters"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== AVERAGE FUEL BY MINE ==========")
print(fuel_by_mine)


# ---------------------------------------------------------
# Fuel Efficiency
# ---------------------------------------------------------

df["Fuel_Efficiency_km_per_liter"] = (
    df["Distance_km"] /
    df["Fuel_Used_Liters"]
)

fuel_efficiency_by_mine = (
    df.groupby("Mine")["Fuel_Efficiency_km_per_liter"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== FUEL EFFICIENCY BY MINE ==========")
print(fuel_efficiency_by_mine)

# ---------------------------------------------------------
# Loading Time by Mine
# ---------------------------------------------------------

loading_by_mine = (
    df.groupby("Mine")["Loading_Time_Min"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== AVERAGE LOADING TIME BY MINE ==========")
print(loading_by_mine)


# ---------------------------------------------------------
# Delay Rate by Loading Time Group
# ---------------------------------------------------------

df["Loading_Time_Group"] = pd.cut(
    df["Loading_Time_Min"],
    bins=[0, 30, 40, 50, 100],
    labels=[
        "Under 30 min",
        "30-40 min",
        "40-50 min",
        "Over 50 min"
    ]
)

loading_delay = (
    df.groupby(
        "Loading_Time_Group",
        observed=True
    )["Delay"]
    .mean()
)

print("\n========== DELAY RATE BY LOADING TIME ==========")
print(loading_delay * 100)

# ---------------------------------------------------------
# Delay Rate by Driver Experience
# ---------------------------------------------------------

df["Experience_Group"] = pd.cut(
    df["Driver_Experience_Years"],
    bins=[0, 3, 7, 12, 20],
    labels=[
        "1-3 years",
        "4-7 years",
        "8-12 years",
        "13-20 years"
    ]
)

experience_delay = (
    df.groupby(
        "Experience_Group",
        observed=True
    )["Delay"]
    .mean()
)

print("\n========== DELAY RATE BY DRIVER EXPERIENCE ==========")
print(experience_delay * 100)

# ---------------------------------------------------------
# Save analysis-ready dataset
# ---------------------------------------------------------

analysis_path = "data/processed/coal_transport_analysis.csv"

df.to_csv(
    analysis_path,
    index=False
)

print(
    f"\nAnalysis dataset saved to: {analysis_path}"
)

