import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

file_path = "data/processed/coal_transport_analysis.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ---------------------------------------------------------
# Select features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Define feature types
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ---------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Logistic Regression Pipeline
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")

# ---------------------------------------------------------
# Extract Feature Names
# ---------------------------------------------------------

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

print("\n========== MODEL FEATURES ==========")

for feature in feature_names:
    print(feature)

    # ---------------------------------------------------------
# Extract Logistic Regression Coefficients
# ---------------------------------------------------------

coefficients = model.named_steps["classifier"].coef_[0]

coefficient_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by coefficient
coefficient_df = coefficient_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\n========== FEATURE COEFFICIENTS ==========")

print(
    coefficient_df.to_string(index=False)
)