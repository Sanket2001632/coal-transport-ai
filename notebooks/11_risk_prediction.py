import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


print("========== COAL TRANSPORT RISK PREDICTION ==========")


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

DATA_PATH = "data/processed/coal_transport_analysis.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ---------------------------------------------------------
# 2. Features
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
# 3. Feature types
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
# 4. Preprocessing
# ---------------------------------------------------------

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_transformer,
            numerical_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 6. Gradient Boosting model
# ---------------------------------------------------------

gradient_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            gradient_model
        )
    ]
)


print("\nTraining Gradient Boosting model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ---------------------------------------------------------
# 7. Probability prediction
# ---------------------------------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


print("\nProbability predictions generated!")


# ---------------------------------------------------------
# 8. Test different thresholds
# ---------------------------------------------------------

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


print("\n========== THRESHOLD ANALYSIS ==========")

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    delay_recall = report["1"]["recall"]

    delay_precision = report["1"]["precision"]

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Delay Precision: {delay_precision:.4f}"
    )

    print(
        f"Delay Recall: {delay_recall:.4f}"
    )


# ---------------------------------------------------------
# 9. Business risk categories
# ---------------------------------------------------------

risk_threshold = 0.40

risk_predictions = (
    probabilities >= risk_threshold
).astype(int)


print("\n========== BUSINESS THRESHOLD ==========")

print(
    f"Selected threshold: {risk_threshold}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        risk_predictions
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        risk_predictions
    )
)


# ---------------------------------------------------------
# 10. Create risk levels
# ---------------------------------------------------------

def calculate_risk(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.60:
        return "MEDIUM"

    elif probability < 0.80:
        return "HIGH"

    else:
        return "CRITICAL"


# ---------------------------------------------------------
# 11. Create prediction dataframe
# ---------------------------------------------------------

results = X_test.copy()

results["Actual_Delay"] = y_test.values

results["Delay_Probability"] = probabilities

results["Risk_Level"] = [
    calculate_risk(p)
    for p in probabilities
]


# ---------------------------------------------------------
# 12. Sort highest risk first
# ---------------------------------------------------------

results = results.sort_values(
    "Delay_Probability",
    ascending=False
)


print("\n========== TOP 20 HIGH-RISK TRIPS ==========")

print(
    results.head(20)[
        [
            "Distance_km",
            "Average_Speed_kmh",
            "Loading_Time_Min",
            "Weather",
            "Traffic_Level",
            "Previous_Delays",
            "Delay_Probability",
            "Risk_Level",
            "Actual_Delay"
        ]
    ]
)


# ---------------------------------------------------------
# 13. Save results
# ---------------------------------------------------------

OUTPUT_PATH = (
    "data/processed/"
    "coal_transport_risk_predictions.csv"
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nRisk prediction dataset saved to: "
    f"{OUTPUT_PATH}"
)


print("\n========== PROJECT STEP COMPLETED ==========")