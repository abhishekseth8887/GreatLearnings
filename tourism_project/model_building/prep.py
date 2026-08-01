import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COL = "ProdTaken"

# Load from repository data folder.
df = pd.read_csv(DATA_PATH)

# Drop auto-generated index column and identifier column.
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Standardize text-like fields.
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()

df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})

# Convert target to integer.
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")

# Drop rows with missing target.
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed.")
print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
