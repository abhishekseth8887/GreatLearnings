import os
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome"
}

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Upload tourism.csv into tourism_project/data/ first."
    )

raw_df = pd.read_csv(DATA_PATH)

# Drop auto-generated index column if present.
if "Unnamed: 0" in raw_df.columns:
    raw_df = raw_df.drop(columns=["Unnamed: 0"])

missing_columns = EXPECTED_COLUMNS.difference(raw_df.columns)
extra_columns = set(raw_df.columns).difference(EXPECTED_COLUMNS)

if missing_columns:
    raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")

print("Dataset validation passed.")
print(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
print("Columns:")
print(sorted(raw_df.columns))

if extra_columns:
    print(f"Additional columns detected (allowed): {sorted(extra_columns)}")

print("\nMissing values per column:")
print(raw_df.isna().sum())

print("\nTarget distribution (ProdTaken):")
print(raw_df["ProdTaken"].value_counts(normalize=True).rename("ratio"))
