import os
import json
import time
import subprocess
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from huggingface_hub import hf_hub_download, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def main():
    project_dir = os.getenv("PROJECT_DIR", "tourism_project")
    mlruns_dir = os.path.join(project_dir, "mlruns")
    art_dir = os.path.join(project_dir, "model_artifacts")
    os.makedirs(mlruns_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    # MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "visit-with-us-wellness-tourism_experiment"))

    # Data from HF dataset repo
    dataset_repo_id = os.getenv("HF_DATASET_REPO_ID", "akanshasalampuria/visit-with-us-wellness-tourism")
    repo_type = os.getenv("HF_DATASET_REPO_TYPE", "dataset")

    Xtrain = pd.read_csv(hf_hub_download(dataset_repo_id, "processed_xy/Xtrain.csv", repo_type=repo_type))
    Xtest  = pd.read_csv(hf_hub_download(dataset_repo_id, "processed_xy/Xtest.csv", repo_type=repo_type))
    ytrain = pd.read_csv(hf_hub_download(dataset_repo_id, "processed_xy/ytrain.csv", repo_type=repo_type)).iloc[:, 0]
    ytest  = pd.read_csv(hf_hub_download(dataset_repo_id, "processed_xy/ytest.csv", repo_type=repo_type)).iloc[:, 0]

    # Preprocess
    cat_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in Xtrain.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    models_and_grids = [
        ("DecisionTree", DecisionTreeClassifier(random_state=42), {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"],
        }),
        ("Bagging", BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42), {
            "model__n_estimators": [50, 100],
            "model__max_samples": [0.7, 1.0],
            "model__estimator__max_depth": [None, 5, 10],
        }),
        ("RandomForest", RandomForestClassifier(random_state=42, n_jobs=-1), {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"],
        }),
        ("AdaBoost", AdaBoostClassifier(random_state=42), {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1, 0.5],
        }),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42), {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        }),
        ("XGBoost", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1), {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }),
    ]

    best_overall = {"model_name": None, "best_estimator": None, "best_params": None, "test_f1": -1.0, "run_id": None, "test_metrics": None}

    for model_name, model, param_grid in models_and_grids:
        with mlflow.start_run(run_name=model_name) as run:
            pipe = Pipeline([("preprocess", preprocess), ("model", model)])

            grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1, return_train_score=True)
            grid.fit(Xtrain, ytrain)

            best_est = grid.best_estimator_
            best_params = grid.best_params_

            ypred = best_est.predict(Xtest)

            try:
                yprob = best_est.predict_proba(Xtest)[:, 1]
                roc = roc_auc_score(ytest, yprob)
            except Exception:
                roc = None

            test_metrics = {
                "test_accuracy": float(accuracy_score(ytest, ypred)),
                "test_precision": float(precision_score(ytest, ypred, zero_division=0)),
                "test_recall": float(recall_score(ytest, ypred, zero_division=0)),
                "test_f1": float(f1_score(ytest, ypred, zero_division=0)),
                "best_cv_f1": float(grid.best_score_),
            }
            if roc is not None:
                test_metrics["test_roc_auc"] = float(roc)

            mlflow.log_params(best_params)
            mlflow.log_metrics(test_metrics)

            cv_results = pd.DataFrame(grid.cv_results_)
            cv_path = os.path.join(art_dir, f"{model_name}_cv_results.csv")
            cv_results.to_csv(cv_path, index=False)
            mlflow.log_artifact(cv_path, artifact_path="tuning")

            report = classification_report(ytest, ypred)
            report_path = os.path.join(art_dir, f"{model_name}_classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path, artifact_path="reports")

            input_example = Xtrain.head(5)
            mlflow.sklearn.log_model(
              sk_model=best_est,
              name="model",
              input_example=input_example
            )

            if test_metrics["test_f1"] > best_overall["test_f1"]:
                best_overall.update({
                    "model_name": model_name,
                    "best_estimator": best_est,
                    "best_params": best_params,
                    "test_f1": test_metrics["test_f1"],
                    "run_id": run.info.run_id,
                    "test_metrics": test_metrics
                })

    # Save best model locally
    best_model_path = os.path.join(art_dir, "best_model.joblib")
    joblib.dump(best_overall["best_estimator"], best_model_path)

    # Register to HF Model Hub
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set. Needed to push model to HF Model Hub.")

    model_repo_id = os.getenv("HF_MODEL_REPO_ID", "akanshasalampuria/visit-with-us-wellness-model")
    api = HfApi(token=hf_token)

    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo_id, repo_type="model", private=False, token=hf_token)

    summary = {
        "best_model_name": best_overall["model_name"],
        "best_params": best_overall["best_params"],
        "best_test_metrics": best_overall["test_metrics"],
        "best_run_id": best_overall["run_id"],
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
    }
    summary_path = os.path.join(art_dir, "best_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    readme_path = os.path.join(art_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(
            "# Visit with Us – Wellness Tourism Purchase Predictor\n\n"
            f"**Best Model:** {best_overall['model_name']}\n\n"
            f"**Best Test F1:** {best_overall['test_f1']:.4f}\n\n"
            "## Best Params\n"
            f"```json\n{json.dumps(best_overall['best_params'], indent=2)}\n```\n"
        )

    req_path = os.path.join(art_dir, "requirements.txt")
    with open(req_path, "w") as f:
        f.write("pandas\nscikit-learn\nxgboost\njoblib\nmlflow\nhuggingface_hub\n")

    for local_path, hf_path in {
        best_model_path: "model.joblib",
        summary_path: "best_model_summary.json",
        readme_path: "README.md",
        req_path: "requirements.txt",
    }.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=model_repo_id,
            repo_type="model",
            commit_message=f"Add/Update {hf_path}"
        )

    print("Done. Best model pushed to HF:", model_repo_id)


if __name__ == "__main__":
    main()
