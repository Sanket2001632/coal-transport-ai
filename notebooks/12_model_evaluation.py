import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/raw/coal_transport_data_10000.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MODEL EVALUATION & EXPLAINABILITY")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. CREATE DERIVED FEATURES
# ============================================================

# Fuel efficiency was created in Step 09, but the raw dataset
# does not contain it. Therefore calculate it here.

df["Fuel_Efficiency_km_per_liter"] = (
    df["Distance_km"] / df["Fuel_Used_Liters"]
)

print("\nFuel efficiency feature created successfully!")

print(
    df[
        [
            "Distance_km",
            "Fuel_Used_Liters",
            "Fuel_Efficiency_km_per_liter"
        ]
    ].head()
)


# ============================================================
# 3. FEATURES AND TARGET
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
print(features)

print("\nTarget:")
print(target)


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
        )
    ],
    remainder="passthrough"
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
# 8. PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully!")


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"ROC-AUC: {roc_auc:.4f}")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Delay",
            "Delay"
        ]
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# 12. BUSINESS ERROR ANALYSIS
# ============================================================

print("\n========== BUSINESS ERROR ANALYSIS ==========")

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    "\nFalse Negatives represent delayed trips "
    "that the model failed to identify."
)


# ============================================================
# 13. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Gradient Boosting (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Coal Transport Delay Prediction"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/roc_curve_gradient_boosting.png",
    dpi=150
)

plt.close()


# ============================================================
# 14. PRECISION-RECALL CURVE
# ============================================================

precision, recall, pr_thresholds = precision_recall_curve(
    y_test,
    y_probability
)

average_precision = average_precision_score(
    y_test,
    y_probability
)

print("\n========== PRECISION-RECALL ==========")

print(
    f"Average Precision Score: "
    f"{average_precision:.4f}"
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Delay Prediction"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/precision_recall_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = pipeline.named_steps[
    "model"
].feature_importances_

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 16. FEATURE IMPORTANCE CHART
# ============================================================

top_features = (
    feature_importance
    .head(15)
    .sort_values(by="Importance")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top Features Influencing Delay Prediction"
)

plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png",
    dpi=150
)

plt.close()


# ============================================================
# 17. CROSS VALIDATION
# ============================================================

print("\n========== CROSS VALIDATION ==========")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("ROC-AUC scores:")

print(cv_scores)

print(
    f"\nMean ROC-AUC: "
    f"{cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: "
    f"{cv_scores.std():.4f}"
)


# ============================================================
# 18. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nAccuracy       : {accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC        : {roc_auc:.4f}"
)

print(
    f"Avg Precision  : {average_precision:.4f}"
)

print(
    f"CV Mean AUC    : {cv_scores.mean():.4f}"
)

print("\nFiles created:")

print(
    "- data/processed/"
    "roc_curve_gradient_boosting.png"
)

print(
    "- data/processed/"
    "precision_recall_curve.png"
)

print(
    "- data/processed/"
    "feature_importance.png"
)

print(
    "- data/processed/"
    "feature_importance.csv"
)