from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.data import compute_binary_metrics, get_feature_columns, split_features_and_target
from fraud_detection.demo_data import generate_demo_transactions
from fraud_detection.pipeline import build_training_pipeline, compute_scale_pos_weight, recommend_smote_neighbors


def test_demo_dataset_has_imbalanced_target() -> None:
    dataset = generate_demo_transactions(n_samples=5000, random_state=7)
    fraud_rate = dataset["is_fraud"].mean()

    assert 0.01 <= fraud_rate <= 0.03


def test_training_pipeline_fits_and_scores() -> None:
    dataset = generate_demo_transactions(n_samples=2500, random_state=11)
    feature_columns = get_feature_columns(dataset, target_column="is_fraud", id_column="transaction_id")
    features, target = split_features_and_target(dataset, feature_columns, target_column="is_fraud")

    pipeline = build_training_pipeline(
        feature_columns=feature_columns,
        random_state=11,
        sampling_strategy=0.1,
        smote_k_neighbors=recommend_smote_neighbors(target.to_numpy()),
        scale_pos_weight=compute_scale_pos_weight(target.to_numpy()),
        xgb_params={
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
    )

    pipeline.fit(features, target)
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = compute_binary_metrics(target, predictions, probabilities)

    assert probabilities.shape == (len(features),)
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0

