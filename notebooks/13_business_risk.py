import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


print("\n========== BUSINESS RISK OPTIMIZATION ==========")


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/raw/coal_transport_data_10000.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. CREATE FUEL EFFICIENCY FEATURE
# ============================================================

df["Fuel_Efficiency_km_per_liter"] = (
    df["Distance_km"] / df["Fuel_Used_Liters"]
)

print("\nFuel efficiency feature created.")


# ============================================================
# 3. FEATURES
# ============================================================

features = [
    "Distance_km",
    "Coal_Weight_Ton",
    "Fuel_Used_Liters",
    "Average_Speed_kmh",
    "Loading_Time_Min",
    "Unloading_Time_Min",
    "Weather",
    "Traffic_Level",
    "Driver_Experience_Years",
    "Previous_Delays",
    "Safety_Score",
    "Fuel_Efficiency_km_per_liter"
]

target = "Delay"

X = df[features]
y = df[target]


# ============================================================
# 4. CATEGORICAL / NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "Weather",
    "Traffic_Level"
]

numerical_features = [
    "Distance_km",
    "Coal_Weight_Ton",
    "Fuel_Used_Liters",
    "Average_Speed_kmh",
    "Loading_Time_Min",
    "Unloading_Time_Min",
    "Driver_Experience_Years",
    "Previous_Delays",
    "Safety_Score",
    "Fuel_Efficiency_km_per_liter"
]


# ============================================================
# 5. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)


# ============================================================
# 7. GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

print("\nTraining Gradient Boosting model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 8. PREDICT DELAY PROBABILITY
# ============================================================

probabilities = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probabilities)

print("\n========== MODEL PERFORMANCE ==========")
print(f"ROC-AUC: {auc:.4f}")


# ============================================================
# 9. CREATE RISK DATASET
# ============================================================

risk_df = X_test.copy()

risk_df["Delay_Probability"] = probabilities

risk_df["Actual_Delay"] = y_test.values


# ============================================================
# 10. RISK LEVEL
# ============================================================

def assign_risk(probability):

    if probability >= 0.70:
        return "CRITICAL"

    elif probability >= 0.50:
        return "HIGH"

    elif probability >= 0.30:
        return "MEDIUM"

    else:
        return "LOW"


risk_df["Risk_Level"] = risk_df["Delay_Probability"].apply(
    assign_risk
)


# ============================================================
# 11. RECOMMENDED ACTION
# ============================================================

def recommend_action(row):

    probability = row["Delay_Probability"]
    weather = row["Weather"]
    traffic = row["Traffic_Level"]
    speed = row["Average_Speed_kmh"]
    loading = row["Loading_Time_Min"]

    if probability >= 0.70:

        if weather == "Heavy_Rain":
            return "Critical monitoring - Heavy rain risk"

        elif traffic == "High":
            return "Critical monitoring - High traffic"

        elif speed < 30:
            return "Critical monitoring - Low vehicle speed"

        elif loading > 50:
            return "Critical monitoring - Excessive loading time"

        else:
            return "Immediate operational intervention"

    elif probability >= 0.50:

        return "Increase trip monitoring"

    elif probability >= 0.30:

        return "Monitor trip"

    else:

        return "Normal operation"


risk_df["Recommended_Action"] = risk_df.apply(
    recommend_action,
    axis=1
)


# ============================================================
# 12. RISK DISTRIBUTION
# ============================================================

print("\n========== RISK DISTRIBUTION ==========")

risk_distribution = (
    risk_df["Risk_Level"]
    .value_counts()
)

print(risk_distribution)


# ============================================================
# 13. RISK PERCENTAGE
# ============================================================

risk_percentage = (
    risk_df["Risk_Level"]
    .value_counts(normalize=True)
    * 100
)

print("\n========== RISK PERCENTAGE ==========")

print(risk_percentage.round(2))


# ============================================================
# 14. ACTUAL DELAY RATE BY RISK
# ============================================================

risk_delay_rate = (
    risk_df
    .groupby("Risk_Level")["Actual_Delay"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\n========== ACTUAL DELAY RATE BY RISK ==========")

print(risk_delay_rate.round(2))


# ============================================================
# 15. HIGH-RISK TRIPS
# ============================================================

high_risk = risk_df[
    risk_df["Risk_Level"].isin(
        ["CRITICAL", "HIGH"]
    )
]

print("\n========== HIGH-RISK TRIPS ==========")

print(
    "Number of high-risk trips:",
    len(high_risk)
)


# ============================================================
# 16. HIGH-RISK DELAY RATE
# ============================================================

if len(high_risk) > 0:

    high_risk_delay_rate = (
        high_risk["Actual_Delay"].mean() * 100
    )

else:

    high_risk_delay_rate = 0


print(
    f"High-risk actual delay rate: "
    f"{high_risk_delay_rate:.2f}%"
)


# ============================================================
# 17. POTENTIALLY MISSED DELAYS
# ============================================================

missed_delays = risk_df[
    (risk_df["Actual_Delay"] == 1)
    &
    (risk_df["Risk_Level"].isin(["LOW", "MEDIUM"]))
]

print("\n========== MISSED HIGH-RISK OPPORTUNITIES ==========")

print(
    "Actual delayed trips classified LOW/MEDIUM:",
    len(missed_delays)
)


# ============================================================
# 18. TOP 20 CRITICAL TRIPS
# ============================================================

top_critical = (
    risk_df
    .sort_values(
        "Delay_Probability",
        ascending=False
    )
    .head(20)
)

print("\n========== TOP 20 HIGHEST-RISK TRIPS ==========")

print(
    top_critical[
        [
            "Distance_km",
            "Average_Speed_kmh",
            "Loading_Time_Min",
            "Weather",
            "Traffic_Level",
            "Previous_Delays",
            "Delay_Probability",
            "Risk_Level",
            "Recommended_Action",
            "Actual_Delay"
        ]
    ].to_string(index=False)
)


# ============================================================
# 19. SAVE BUSINESS RISK DATA
# ============================================================

OUTPUT_PATH = (
    "data/processed/"
    "coal_transport_business_risk.csv"
)

risk_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nBusiness risk dataset saved to:")
print(OUTPUT_PATH)


# ============================================================
# 20. PROJECT STEP COMPLETED
# ============================================================

print("\n========== PROJECT STEP COMPLETED ==========")
print("Step 13 completed successfully!")