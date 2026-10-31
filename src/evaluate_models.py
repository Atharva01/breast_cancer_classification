from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
RANDOM_STATE = 42

##################################
# 1. Path
##################################

test_path = Path("data/processed/test.csv")
model_path = Path("models/model.pkl")

metrics_dir = Path("src/metrics")
metrics_dir.mkdir(exist_ok=True)

metrics_path = metrics_dir / "metrics.json"

##################################
# 2. Create final houldout set
##################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

###################################
# 3. Create models  
###################################

models = {
    "Logistic Regression": Pipeline([
        ("scaler",StandardScaler()),
        ("classifier",LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE)),
    ]),
    
    "RBF SVM": Pipeline([
        ("scaler",StandardScaler()),
        ("classifier",SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
        )),
    ]),
    
    "KNN": Pipeline([
        ("scaler",StandardScaler()),
        ("classifier",KNeighborsClassifier(
            n_neighbors=5,
        )),
    ]),
    
    "Random Forest": Pipeline([
        ("scaler",StandardScaler()),
        ("classifier",RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )
}

###################################
# 4. Cross validation
###################################

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=RANDOM_STATE,   
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
    ]

results = []

for model_name,model in models.items():
    
    print(f"\nEvaluating model: {model_name}")
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    
    result = {
        "model":model_name,
        "accuracy":scores["test_accuracy"].mean(),
        "precision":scores["test_precision"].mean(),
        "recall":scores["test_recall"].mean(),
        "f1":scores["test_f1"].mean(),
        "roc_auc":scores["test_roc_auc"].mean(),
    }
    print(f"Results: {result}")
    results.append(result)
    
###################################
# 5. Results
###################################

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False
)

print(f"\nModel Comparison")
print(results_df.to_string(
    index=False
))

###################################
# 6. Save metrics
###################################

Path("src/metrics").mkdir(
    exist_ok=True
)

results_df.to_csv("src/metrics/model_comparison.csv",
                  index=False,)

print(f"\nResults saved to: metrics/model_comparison.csv")

###################################
# 7. Conclusion
###################################

print(f"Model selected: {results_df["model"].iloc[0]} "
      f"with ROC AUC: {results_df["roc_auc"].iloc[0]:.4f}")

