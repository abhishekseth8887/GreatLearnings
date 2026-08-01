import os
import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

EXPERIMENT_NAME = "tourism-package-classification"
MODEL_OUTPUT_PATH = "tourism_project/deployment/tourism_package_model.joblib"

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

ytrain = pd.to_numeric(ytrain, errors="coerce").astype(int)
ytest = pd.to_numeric(ytest, errors="coerce").astype(int)

numeric_cols = Xtrain.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in Xtrain.columns if c not in numeric_cols]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

pos = int((ytrain == 1).sum())
neg = int((ytrain == 0).sum())
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="xgb-grid-search"):
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1",
        verbose=1,
    )
    search.fit(Xtrain, ytrain)

    results = search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"candidate-{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_cv_f1", float(results["std_test_score"][i]))

    best_model = search.best_estimator_
    mlflow.log_params(search.best_params_)

    train_pred = best_model.predict(Xtrain)
    test_pred = best_model.predict(Xtest)
    test_proba = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(ytrain, train_pred),
        "test_accuracy": accuracy_score(ytest, test_pred),
        "test_precision": precision_score(ytest, test_pred, zero_division=0),
        "test_recall": recall_score(ytest, test_pred, zero_division=0),
        "test_f1": f1_score(ytest, test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(ytest, test_proba),
    }

    mlflow.log_metrics(metrics)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    mlflow.log_artifact(MODEL_OUTPUT_PATH, artifact_path="model")

print("Best params:", search.best_params_)
print("Model saved to:", MODEL_OUTPUT_PATH)
