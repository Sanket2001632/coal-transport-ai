import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
# ---------------------------------------------------------
# Load analysis dataset
# ---------------------------------------------------------

file_path = "data/processed/coal_transport_analysis.csv"

df = pd.read_csv(file_path)

print("ML dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ---------------------------------------------------------
# Select ML features
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

print("\n========== ML FEATURES ==========")
print(X.columns.tolist())

print("\n========== TARGET ==========")
print(target)

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ---------------------------------------------------------
# Identify categorical and numerical features
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

print("\n========== CATEGORICAL FEATURES ==========")
print(categorical_features)

print("\n========== NUMERICAL FEATURES ==========")
print(numerical_features)

# ---------------------------------------------------------
# Preprocessing
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

print("\nPreprocessor created successfully.")

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

print("\n========== TRAIN / TEST SPLIT ==========")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\n========== TARGET DISTRIBUTION ==========")

print("Overall:")
print(y.value_counts(normalize=True) * 100)

print("\nTraining:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting:")
print(y_test.value_counts(normalize=True) * 100)

# ---------------------------------------------------------
# Logistic Regression Model
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

print("\nLogistic Regression pipeline created.")

# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully.")

# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== MODEL ACCURACY ==========")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# ---------------------------------------------------------
# ROC-AUC
# ---------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== ROC-AUC ==========")
print(f"ROC-AUC: {roc_auc:.4f}")

# ---------------------------------------------------------
# Random Forest Model
# ---------------------------------------------------------

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nRandom Forest pipeline created.")

# ---------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------

print("\nTraining Random Forest model...")

random_forest.fit(
    X_train,
    y_train
)

print("Random Forest training completed successfully.")

# ---------------------------------------------------------
# Random Forest Predictions
# ---------------------------------------------------------

rf_pred = random_forest.predict(X_test)

rf_probability = random_forest.predict_proba(X_test)[:, 1]

print("\nRandom Forest predictions generated successfully.")

# ---------------------------------------------------------
# Random Forest Evaluation
# ---------------------------------------------------------

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")

print(f"ROC-AUC: {rf_auc:.4f}")

print("\n========== RANDOM FOREST CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

print("\n========== RANDOM FOREST CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)

