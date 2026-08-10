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


print("========== ADVANCED MODEL EXPERIMENT ==========")

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

DATA_PATH = "data/processed/coal_transport_analysis.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ---------------------------------------------------------
# 2. Select ML features
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


print("\n========== FEATURES ==========")

print(features)

print("\nTarget:")
print(target)


# ---------------------------------------------------------
# 3. Define feature types
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
# 4. Numerical preprocessing
# ---------------------------------------------------------

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ---------------------------------------------------------
# 5. Categorical preprocessing
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 6. Combined preprocessing
# ---------------------------------------------------------

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
# 7. Train / test split
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 8. Gradient Boosting Model
# ---------------------------------------------------------

gradient_boosting = GradientBoostingClassifier(
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
            gradient_boosting
        )
    ]
)


print("\nGradient Boosting pipeline created.")


# ---------------------------------------------------------
# 9. Train
# ---------------------------------------------------------

print("\nTraining Gradient Boosting model...")

model.fit(
    X_train,
    y_train
)

print("Gradient Boosting training completed!")


# ---------------------------------------------------------
# 10. Predictions
# ---------------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]

print("\nPredictions generated successfully!")


# ---------------------------------------------------------
# 11. Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n========== GRADIENT BOOSTING RESULTS ==========")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ---------------------------------------------------------
# 12. Classification report
# ---------------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ---------------------------------------------------------
# 13. Confusion matrix
# ---------------------------------------------------------

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)