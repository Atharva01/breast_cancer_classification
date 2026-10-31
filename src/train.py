from pathlib import Path
import joblib
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


with open("params.yaml","r") as f:
    params = yaml.safe_load(f)

C = params["train"]["C"]
MAX_ITER = params["train"]["max_iter"]
RANDOM_STATE = params["split"]["random_state"]


# -----------------------------
# Paths
# -----------------------------

train_path = Path(
    "data/processed/train.csv"
)

model_dir = Path("models")

model_dir.mkdir(
    exist_ok=True
)

model_path = (
    model_dir /
    "model.pkl"
)


# -----------------------------
# Load training data
# -----------------------------

train_df = pd.read_csv(
    train_path
)

X_train = train_df.drop(
    columns=["target"]
)

y_train = train_df["target"]


print(
    f"Training data: "
    f"{X_train.shape}"
)


# -----------------------------
# Model
# -----------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    )
])


# -----------------------------
# Train
# -----------------------------

model.fit(
    X_train,
    y_train
)


# -----------------------------
# Save model
# -----------------------------

joblib.dump(
    model,
    model_path
)


print(
    f"Model saved to: "
    f"{model_path}"
)