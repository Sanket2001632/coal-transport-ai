import pandas as pd

# Load the dataset
file_path = "data/raw/coal_transport_data_100.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== FIRST 10 ROWS ==========")
print(df.head(10))

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe())

print("\n========== DELAY DISTRIBUTION ==========")
print(df["Delay"].value_counts())

print("\n========== DELAY PERCENTAGE ==========")
print(df["Delay"].value_counts(normalize=True) * 100)