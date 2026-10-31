from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


###################################
# 1. Paths
###################################

test_path = Path("data/processed/test.csv")
model_path = Path("models/model.pkl")

metrics_dir = Path("metrics")
metrics_dir.mkdir(exist_ok=True)

metrics_path = metrics_dir / "final_metrics.json"


###################################
# 2. Load test data
###################################

test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

print(f"Test data shape: {X_test.shape}")


###################################
# 3. Load model
###################################

model = joblib.load(model_path)

print(f"Loaded model from: {model_path}")


###################################
# 4. Predictions
###################################

y_pred = model.predict(X_test)


# ROC-AUC needs a continuous score,
# not the final class prediction.

if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]

elif hasattr(model, "decision_function"):
    y_score = model.decision_function(X_test)

else:
    raise ValueError(
        "Model does not provide predict_proba "
        "or decision_function."
    )


###################################
# 5. Metrics
###################################

accuracy = accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
)

recall = recall_score(
    y_test,
    y_pred,
)

f1 = f1_score(
    y_test,
    y_pred,
)

roc_auc = roc_auc_score(
    y_test,
    y_score,
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
).ravel()


###################################
# 6. Save metrics
###################################

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
}


with open(metrics_path, "w") as f:
    json.dump(
        metrics,
        f,
        indent=4,
    )


print("\nFinal metrics:")

for name, value in metrics.items():
    print(f"{name}: {value}")


print(
    f"\nMetrics saved to: "
    f"{metrics_path}"
)