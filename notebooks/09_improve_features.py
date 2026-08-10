import pandas as pd

# ---------------------------------------------------------
# Load Analysis Dataset
# ---------------------------------------------------------

file_path = "data/processed/coal_transport_analysis.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\n========== ORIGINAL COLUMNS ==========")

for column in df.columns:
    print(column)

    # ---------------------------------------------------------
# Create Time-Based Features
# ---------------------------------------------------------

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract day of week
df["Day_of_Week"] = df["Date"].dt.dayofweek

# Extract month
df["Month"] = df["Date"].dt.month

# Identify weekends
df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)


print("\n========== NEW TIME FEATURES ==========")

print(
    df[
        [
            "Date",
            "Day_of_Week",
            "Month",
            "Is_Weekend"
        ]
    ].head(10)
)

# ---------------------------------------------------------
# Create Transportation Features
# ---------------------------------------------------------

# Total loading + unloading time
df["Total_Handling_Time_Min"] = (
    df["Loading_Time_Min"] +
    df["Unloading_Time_Min"]
)

# Estimated travel time in hours
df["Estimated_Travel_Time_Hours"] = (
    df["Distance_km"] /
    df["Average_Speed_kmh"]
)

# Coal transported per kilometer
df["Coal_per_Km"] = (
    df["Coal_Weight_Ton"] /
    df["Distance_km"]
)

print("\n========== TRANSPORTATION FEATURES ==========")

print(
    df[
        [
            "Distance_km",
            "Coal_Weight_Ton",
            "Average_Speed_kmh",
            "Loading_Time_Min",
            "Unloading_Time_Min",
            "Total_Handling_Time_Min",
            "Estimated_Travel_Time_Hours",
            "Coal_per_Km"
        ]
    ].head(10)
)

# ---------------------------------------------------------
# Define Improved ML Features
# ---------------------------------------------------------

improved_features = [
    "Distance_km",
    "Coal_Weight_Ton",
    "Average_Speed_kmh",
    "Loading_Time_Min",
    "Unloading_Time_Min",
    "Total_Handling_Time_Min",
    "Estimated_Travel_Time_Hours",
    "Coal_per_Km",
    "Weather",
    "Traffic_Level",
    "Driver_Experience_Years",
    "Previous_Delays",
    "Safety_Score",
    "Fuel_Efficiency_km_per_liter",
    "Day_of_Week",
    "Month",
    "Is_Weekend"
]

target = "Delay"


print("\n========== IMPROVED ML FEATURES ==========")

for feature in improved_features:
    print(feature)

print("\nTotal features:", len(improved_features))

print("\nTarget:", target)

# ---------------------------------------------------------
# Check Feature Correlations
# ---------------------------------------------------------

numerical_features_for_check = [
    "Distance_km",
    "Coal_Weight_Ton",
    "Average_Speed_kmh",
    "Loading_Time_Min",
    "Unloading_Time_Min",
    "Total_Handling_Time_Min",
    "Estimated_Travel_Time_Hours",
    "Coal_per_Km",
    "Driver_Experience_Years",
    "Previous_Delays",
    "Safety_Score",
    "Fuel_Efficiency_km_per_liter",
    "Day_of_Week",
    "Month",
    "Is_Weekend"
]

correlation_matrix = df[numerical_features_for_check].corr()

print("\n========== FEATURE CORRELATION WITH DELAY ==========")

delay_correlations = df[
    numerical_features_for_check + ["Delay"]
].corr()["Delay"].sort_values(
    ascending=False
)

print(delay_correlations)

# ---------------------------------------------------------
# Step 17.6 - Train Improved Logistic Regression Model
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ---------------------------------------------------------
# Create Feature Matrix and Target
# ---------------------------------------------------------

X = df[improved_features]
y = df[target]

print("\n========== IMPROVED ML DATA ==========")
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ---------------------------------------------------------
# Define Categorical and Numerical Features
# ---------------------------------------------------------

categorical_features = [
    "Weather",
    "Traffic_Level"
]

numerical_features = [
    "Distance_km",
    "Coal_Weight_Ton",
    "Average_Speed_kmh",
    "Loading_Time_Min",
    "Unloading_Time_Min",
    "Total_Handling_Time_Min",
    "Estimated_Travel_Time_Hours",
    "Coal_per_Km",
    "Driver_Experience_Years",
    "Previous_Delays",
    "Safety_Score",
    "Fuel_Efficiency_km_per_liter",
    "Day_of_Week",
    "Month",
    "Is_Weekend"
]

print("\n========== CATEGORICAL FEATURES ==========")
print(categorical_features)

print("\n========== NUMERICAL FEATURES ==========")
print(numerical_features)

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_transformer,
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

# ---------------------------------------------------------
# Create Improved Logistic Regression Pipeline
# ---------------------------------------------------------

improved_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

print("\nImproved Logistic Regression pipeline created.")

# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

print("\nTraining improved Logistic Regression model...")

improved_model.fit(
    X_train,
    y_train
)

print("Improved model training completed successfully!")

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = improved_model.predict(X_test)
y_probability = improved_model.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully.")

# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== IMPROVED MODEL RESULTS ==========")

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

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ---------------------------------------------------------
# Compare With Previous Model
# ---------------------------------------------------------

previous_accuracy = 0.7180
previous_roc_auc = 0.7303

print("\n========== MODEL COMPARISON ==========")

print(
    f"Previous Accuracy: {previous_accuracy * 100:.2f}%"
)

print(
    f"Improved Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Accuracy Change: {(accuracy - previous_accuracy) * 100:+.2f} percentage points"
)

print(
    f"Previous ROC-AUC: {previous_roc_auc:.4f}"
)

print(
    f"Improved ROC-AUC: {roc_auc:.4f}"
)

print(
    f"ROC-AUC Change: {roc_auc - previous_roc_auc:+.4f}"
)

# ---------------------------------------------------------
# Step 17.7 - Scaled Logistic Regression
# ---------------------------------------------------------

from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Scaled Numerical Preprocessor
# ---------------------------------------------------------

scaled_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ---------------------------------------------------------
# Categorical Preprocessor
# ---------------------------------------------------------

scaled_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# ---------------------------------------------------------
# Combined Preprocessor
# ---------------------------------------------------------

scaled_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            scaled_numeric_transformer,
            numerical_features
        ),
        (
            "categorical",
            scaled_categorical_transformer,
            categorical_features
        )
    ]
)

# ---------------------------------------------------------
# Scaled Logistic Regression
# ---------------------------------------------------------

scaled_model = Pipeline(
    steps=[
        ("preprocessor", scaled_preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)

print("\n========== SCALED LOGISTIC REGRESSION ==========")

print("Training scaled Logistic Regression model...")

scaled_model.fit(
    X_train,
    y_train
)

print("Scaled model training completed successfully!")

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

scaled_pred = scaled_model.predict(X_test)

scaled_probability = scaled_model.predict_proba(
    X_test
)[:, 1]

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

scaled_accuracy = accuracy_score(
    y_test,
    scaled_pred
)

scaled_roc_auc = roc_auc_score(
    y_test,
    scaled_probability
)

print("\n========== SCALED MODEL RESULTS ==========")

print(
    f"Accuracy: {scaled_accuracy:.4f}"
)

print(
    f"Accuracy: {scaled_accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC: {scaled_roc_auc:.4f}"
)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        scaled_pred
    )
)

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        scaled_pred
    )
)

# ---------------------------------------------------------
# Final Comparison
# ---------------------------------------------------------

print("\n========== THREE MODEL COMPARISON ==========")

print(
    f"Original Logistic Regression ROC-AUC: "
    f"{previous_roc_auc:.4f}"
)

print(
    f"Improved Logistic Regression ROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"Scaled Logistic Regression ROC-AUC: "
    f"{scaled_roc_auc:.4f}"
)