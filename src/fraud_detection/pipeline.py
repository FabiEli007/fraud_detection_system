from __future__ import annotations

import math

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBClassifier


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))
    if positive_count == 0:
        return 1.0
    return max(1.0, negative_count / positive_count)


def recommend_smote_neighbors(y_train: np.ndarray) -> int:
    positive_count = int(np.sum(y_train == 1))
    if positive_count <= 2:
        return 1
    return min(5, positive_count - 1)


def build_training_pipeline(
    feature_columns: list[str],
    random_state: int,
    sampling_strategy: float,
    smote_k_neighbors: int,
    scale_pos_weight: float,
    xgb_params: dict,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SklearnPipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        **xgb_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "sampler",
                SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=smote_k_neighbors,
                ),
            ),
            ("model", model),
        ]
    )

