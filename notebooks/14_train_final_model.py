import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


print("\n========== FINAL MODEL TRAINING ==========")


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/raw/coal_transport_data_10000.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. CREATE FEATURE
# ============================================================

df["Fuel_Efficiency_km_per_liter"] = (
    df["Distance_km"] / df["Fuel_Used_Liters"]
)

print("Fuel efficiency feature created.")


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


print("\n========== FEATURES ==========")

for feature in features:
    print(feature)

print("\nTarget:")
print(target)


# ============================================================
# 4. CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Weather",
    "Traffic_Level"
]


# ============================================================
# 5. NUMERICAL FEATURES
# ============================================================

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
# 6. PREPROCESSOR
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
# 7. TRAIN / TEST SPLIT
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
# 8. CREATE FINAL MODEL
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


# ============================================================
# 9. TRAIN
# ============================================================

print("\nTraining final Gradient Boosting model...")

pipeline.fit(X_train, y_train)

print("Final model training completed!")


# ============================================================
# 10. EVALUATE
# ============================================================

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n========== FINAL MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"ROC-AUC: {roc_auc:.4f}")


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "No Delay",
            "Delay"
        ]
    )
)


print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# ============================================================
# 11. CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

MODEL_PATH = (
    "models/"
    "coal_transport_delay_model.pkl"
)


joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n========== MODEL SAVED ==========")

print(
    f"Model saved successfully to: {MODEL_PATH}"
)


# ============================================================
# 13. SAVE FEATURE CONFIGURATION
# ============================================================

CONFIG_PATH = (
    "models/"
    "model_features.txt"
)


with open(
    CONFIG_PATH,
    "w"
) as file:

    file.write(
        "Coal Transport Delay Prediction Model\n\n"
    )

    file.write(
        "Features:\n"
    )

    for feature in features:

        file.write(
            f"{feature}\n"
        )

    file.write(
        "\nTarget:\n"
    )

    file.write(
        target
    )


print(
    f"Feature configuration saved to: {CONFIG_PATH}"
)


# ============================================================
# 14. TEST SAVED MODEL
# ============================================================

print("\n========== TESTING SAVED MODEL ==========")

loaded_model = joblib.load(
    MODEL_PATH
)

sample_predictions = loaded_model.predict(
    X_test.head(5)
)

sample_probabilities = loaded_model.predict_proba(
    X_test.head(5)
)[:, 1]


print("\nSample predictions:")

for i in range(5):

    print(
        f"Trip {i + 1}: "
        f"Prediction={sample_predictions[i]}, "
        f"Delay Probability={sample_probabilities[i]:.4f}"
    )


print("\n========== STEP 14 COMPLETED ==========")