from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

with open("params.yaml","r") as f:
    params = yaml.safe_load(f)

RANDOM_STATE = params['split']['random_state']
TEST_SIZE = params['split']['test_size']

data_path = Path("data/raw/breast_cancer.csv")
output_dir = Path("data/processed")

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(data_path)
print(f"Loaded dataset from: {data_path}")
print(f"Loaded dataset: {df.shape}")


# -----------------------------
# Flip target
#
# original sklearn:
# 0 = malignant
# 1 = benign
#
# new:
# 1 = malignant
# 0 = benign
# -----------------------------

df["target"] = (
    df["target"] == 0
).astype(int)


# -----------------------------
# Split
# -----------------------------

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["target"],
    random_state=RANDOM_STATE,
)


# -----------------------------
# Save
# -----------------------------

train_path = (
    output_dir / "train.csv"
)

test_path = (
    output_dir / "test.csv"
)


train_df.to_csv(
    train_path,
    index=False,
)

test_df.to_csv(
    test_path,
    index=False,
)


print(
    f"Train: {train_df.shape}"
)

print(
    f"Test: {test_df.shape}"
)

print(
    f"Saved: {train_path}"
)

print(
    f"Saved: {test_path}"
)