from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_feature_columns(df: pd.DataFrame, target_column: str, id_column: str) -> list[str]:
    ignored = {target_column, id_column}
    return [column for column in df.columns if column not in ignored]


def split_features_and_target(df: pd.DataFrame, feature_columns: Iterable[str], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    features = df.loc[:, list(feature_columns)].copy()
    target = df.loc[:, target_column].astype(int).copy()
    return features, target


def compute_binary_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def build_precision_recall_curve(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    threshold_values = np.append(thresholds, np.nan)
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": threshold_values,
        }
    )


def compute_business_metrics(scored_frame: pd.DataFrame, investigation_cost: float) -> dict[str, float]:
    alerted = scored_frame.loc[scored_frame["predicted_label"] == 1]
    true_positives = alerted.loc[alerted["actual_label"] == 1]
    false_negatives = scored_frame.loc[
        (scored_frame["predicted_label"] == 0) & (scored_frame["actual_label"] == 1)
    ]

    blocked_fraud_amount = float(true_positives["amount"].sum())
    missed_fraud_amount = float(false_negatives["amount"].sum())
    operational_cost = float(len(alerted) * investigation_cost)
    estimated_savings = float(blocked_fraud_amount - operational_cost)

    return {
        "alerts_volume": float(len(alerted)),
        "blocked_fraud_amount": blocked_fraud_amount,
        "missed_fraud_amount": missed_fraud_amount,
        "operational_cost": operational_cost,
        "estimated_savings": estimated_savings,
    }

