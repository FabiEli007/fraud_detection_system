from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _normalize_shap_values(raw_values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(raw_values, list):
        if len(raw_values) == 1:
            return np.asarray(raw_values[0])
        return np.asarray(raw_values[-1])
    return np.asarray(raw_values)


def _normalize_expected_value(raw_expected_value: float | np.ndarray | list[float]) -> float:
    array_value = np.asarray(raw_expected_value)
    if array_value.ndim == 0:
        return float(array_value)
    return float(array_value.reshape(-1)[-1])


def _transform_features(pipeline, feature_frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_frame)
    transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    feature_names = list(preprocessor.get_feature_names_out())
    return transformed_array, feature_names


def explain_transaction(pipeline, feature_frame: pd.DataFrame) -> dict:
    transformed_array, feature_names = _transform_features(pipeline, feature_frame)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = _normalize_shap_values(explainer.shap_values(transformed_array))
    expected_value = _normalize_expected_value(explainer.expected_value)

    row_values = shap_values[0]
    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": row_values,
            "abs_shap_value": np.abs(row_values),
        }
    ).sort_values("abs_shap_value", ascending=False)

    positive_share = float(
        contributions.loc[contributions["shap_value"] > 0, "abs_shap_value"].sum()
        / max(contributions["abs_shap_value"].sum(), 1e-12)
    )

    return {
        "base_probability": float(1.0 / (1.0 + np.exp(-expected_value))),
        "positive_contribution_share": positive_share,
        "contributions": contributions,
    }


def global_shap_importance(pipeline, feature_frame: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    sampled_frame = feature_frame.sample(
        n=min(sample_size, len(feature_frame)),
        random_state=42,
        replace=False,
    )
    transformed_array, feature_names = _transform_features(pipeline, sampled_frame)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = _normalize_shap_values(explainer.shap_values(transformed_array))

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

