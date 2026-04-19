from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from fraud_detection.config import load_project_config, resolve_local_path, resolve_tracking_uri
from fraud_detection.data import (
    build_precision_recall_curve,
    compute_binary_metrics,
    compute_business_metrics,
    get_feature_columns,
    load_dataset,
    split_features_and_target,
)
from fraud_detection.pipeline import build_training_pipeline, compute_scale_pos_weight, recommend_smote_neighbors


def _prepare_scored_frame(
    df: pd.DataFrame,
    indices: pd.Index,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    split_name: str,
    target_column: str,
) -> pd.DataFrame:
    scored = df.loc[indices].copy()
    scored["predicted_probability"] = probabilities
    scored["predicted_label"] = predictions
    scored["actual_label"] = scored[target_column].astype(int)
    scored["data_split"] = split_name
    return scored


def train_project(project_root: str | Path, config_path: str | Path) -> dict[str, str | float]:
    root = Path(project_root).resolve()
    config = load_project_config(config_path)

    data_config = config["data"]
    training_config = config["training"]
    experiment_config = config["experiment"]
    business_config = config["business"]
    artifacts_config = config["artifacts"]

    data_path = resolve_local_path(root, data_config["input_path"])
    artifacts_dir = resolve_local_path(root, artifacts_config["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(str(data_path))
    target_column = data_config["target_column"]
    id_column = data_config["id_column"]
    feature_columns = get_feature_columns(df, target_column, id_column)
    features, target = split_features_and_target(df, feature_columns, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=target,
    )

    threshold = float(training_config["decision_threshold"])
    xgb_params = dict(training_config["xgb_params"])
    scale_pos_weight = compute_scale_pos_weight(y_train.to_numpy())
    smote_neighbors = recommend_smote_neighbors(y_train.to_numpy())

    pipeline = build_training_pipeline(
        feature_columns=feature_columns,
        random_state=data_config["random_state"],
        sampling_strategy=float(training_config["sampling_strategy"]),
        smote_k_neighbors=smote_neighbors,
        scale_pos_weight=scale_pos_weight,
        xgb_params=xgb_params,
    )

    cv = StratifiedKFold(
        n_splits=int(training_config["cv_folds"]),
        shuffle=True,
        random_state=data_config["random_state"],
    )
    scoring = {
        "average_precision": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    cv_scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
    cv_metrics = {
        metric_name.replace("test_", "cv_"): float(np.mean(metric_values))
        for metric_name, metric_values in cv_scores.items()
        if metric_name.startswith("test_")
    }

    pipeline.fit(X_train, y_train)

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_metrics = compute_binary_metrics(y_test, test_predictions, test_probabilities)

    full_probabilities = pipeline.predict_proba(features)[:, 1]
    full_predictions = (full_probabilities >= threshold).astype(int)

    full_scored = _prepare_scored_frame(
        df=df,
        indices=features.index,
        probabilities=full_probabilities,
        predictions=full_predictions,
        split_name="train_or_test",
        target_column=target_column,
    )
    full_scored.loc[X_train.index, "data_split"] = "train"
    full_scored.loc[X_test.index, "data_split"] = "test"

    test_scored = _prepare_scored_frame(
        df=df,
        indices=X_test.index,
        probabilities=test_probabilities,
        predictions=test_predictions,
        split_name="test",
        target_column=target_column,
    )
    business_metrics = compute_business_metrics(
        scored_frame=test_scored,
        investigation_cost=float(business_config["investigation_cost"]),
    )

    metrics_payload = {
        "threshold": threshold,
        "feature_columns": feature_columns,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "business_metrics": business_metrics,
        "scale_pos_weight": scale_pos_weight,
        "smote_k_neighbors": smote_neighbors,
    }

    metrics_path = artifacts_dir / "metrics.json"
    model_path = artifacts_dir / "fraud_model.joblib"
    full_scored_path = artifacts_dir / "scored_transactions.csv"
    test_scored_path = artifacts_dir / "test_scored_transactions.csv"
    pr_curve_path = artifacts_dir / "precision_recall_curve.csv"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    build_precision_recall_curve(y_test, test_probabilities).to_csv(pr_curve_path, index=False)
    full_scored.to_csv(full_scored_path, index=False)
    test_scored.to_csv(test_scored_path, index=False)
    joblib.dump(pipeline, model_path)

    mlflow.set_tracking_uri(resolve_tracking_uri(root, experiment_config["tracking_uri"]))
    mlflow.set_experiment(experiment_config["name"])

    with mlflow.start_run(run_name="xgboost-smote-fraud-model"):
        for parameter_name, parameter_value in xgb_params.items():
            mlflow.log_param(f"xgb_{parameter_name}", parameter_value)

        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_param("sampling_strategy", training_config["sampling_strategy"])
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("smote_k_neighbors", smote_neighbors)

        for metric_name, metric_value in {**cv_metrics, **test_metrics, **business_metrics}.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(pr_curve_path))
        mlflow.log_artifact(str(test_scored_path))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return {
        "data_path": str(data_path),
        "artifacts_dir": str(artifacts_dir),
        "model_path": str(model_path),
        "average_precision": test_metrics["average_precision"],
        "recall": test_metrics["recall"],
        "estimated_savings": business_metrics["estimated_savings"],
    }

