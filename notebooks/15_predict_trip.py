import joblib
import pandas as pd


print("\n========== COAL TRANSPORT SINGLE TRIP PREDICTION ==========")


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/coal_transport_delay_model.pkl"

model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")


# ============================================================
# 2. ENTER NEW TRIP DETAILS
# ============================================================

trip = {
    "Distance_km": 140.0,
    "Coal_Weight_Ton": 30.0,
    "Fuel_Used_Liters": 30.0,
    "Average_Speed_kmh": 25.0,
    "Loading_Time_Min": 45.0,
    "Unloading_Time_Min": 30.0,
    "Weather": "Heavy_Rain",
    "Traffic_Level": "High",
    "Driver_Experience_Years": 5,
    "Previous_Delays": 5,
    "Safety_Score": 70.0
}


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

trip_df = pd.DataFrame([trip])


# ============================================================
# 4. CREATE DERIVED FEATURE
# ============================================================

trip_df["Fuel_Efficiency_km_per_liter"] = (
    trip_df["Distance_km"] /
    trip_df["Fuel_Used_Liters"]
)


print("\n========== TRIP DETAILS ==========")

for column, value in trip.items():

    print(
        f"{column}: {value}"
    )

print(
    "Fuel_Efficiency_km_per_liter:",
    round(
        trip_df["Fuel_Efficiency_km_per_liter"].iloc[0],
        2
    )
)


# ============================================================
# 5. PREDICT DELAY PROBABILITY
# ============================================================

delay_probability = model.predict_proba(
    trip_df
)[0][1]


prediction = model.predict(
    trip_df
)[0]


# ============================================================
# 6. ASSIGN BUSINESS RISK
# ============================================================

if delay_probability >= 0.70:

    risk_level = "CRITICAL"

elif delay_probability >= 0.50:

    risk_level = "HIGH"

elif delay_probability >= 0.30:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# 7. IDENTIFY RISK FACTORS
# ============================================================

risk_factors = []


if trip["Weather"] == "Heavy_Rain":

    risk_factors.append(
        "Heavy rain"
    )


if trip["Traffic_Level"] == "High":

    risk_factors.append(
        "High traffic"
    )


if trip["Average_Speed_kmh"] < 30:

    risk_factors.append(
        "Low average speed"
    )


if trip["Previous_Delays"] >= 4:

    risk_factors.append(
        "High number of previous delays"
    )


if trip["Loading_Time_Min"] > 50:

    risk_factors.append(
        "Long loading time"
    )


if trip["Distance_km"] > 120:

    risk_factors.append(
        "Long transport distance"
    )


if trip["Safety_Score"] < 70:

    risk_factors.append(
        "Low safety score"
    )


# ============================================================
# 8. RECOMMENDED ACTION
# ============================================================

if risk_level == "CRITICAL":

    recommended_action = (
        "Immediate operational intervention"
    )

elif risk_level == "HIGH":

    recommended_action = (
        "Increase trip monitoring and prepare contingency"
    )

elif risk_level == "MEDIUM":

    recommended_action = (
        "Monitor trip closely"
    )

else:

    recommended_action = (
        "Normal operation"
    )


# ============================================================
# 9. DISPLAY RESULT
# ============================================================

print("\n========== PREDICTION RESULT ==========")

print(
    f"Delay Probability: "
    f"{delay_probability * 100:.2f}%"
)

print(
    f"Model Prediction: "
    f"{'DELAY' if prediction == 1 else 'NO DELAY'}"
)

print(
    f"Risk Level: {risk_level}"
)


print("\n========== RISK FACTORS ==========")

if risk_factors:

    for factor in risk_factors:

        print(
            f"- {factor}"
        )

else:

    print(
        "No major risk factors identified."
    )


print("\n========== RECOMMENDED ACTION ==========")

print(
    recommended_action
)


print(
    "\n========== STEP 15 COMPLETED =========="
)