from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.config import load_project_config, resolve_local_path
from fraud_detection.explainability import explain_transaction, global_shap_importance
from fraud_detection.feedback import append_feedback


st.set_page_config(page_title="Fraud Radar Console", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(172, 44, 44, 0.12), transparent 24%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
    }
    .hero {
        background: linear-gradient(135deg, rgba(124, 45, 18, 0.9), rgba(127, 29, 29, 0.86));
        border-radius: 22px;
        color: white;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 40px rgba(127, 29, 29, 0.18);
    }
    .hero p {
        margin-bottom: 0;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets() -> tuple[dict, object, dict, pd.DataFrame, pd.DataFrame, Path]:
    config = load_project_config(PROJECT_ROOT / "configs" / "train_config.yaml")
    artifacts_dir = resolve_local_path(PROJECT_ROOT, config["artifacts"]["dir"])
    model_path = artifacts_dir / "fraud_model.joblib"
    metrics_path = artifacts_dir / "metrics.json"
    scored_path = artifacts_dir / "scored_transactions.csv"
    pr_curve_path = artifacts_dir / "precision_recall_curve.csv"

    if not all(path.exists() for path in [model_path, metrics_path, scored_path, pr_curve_path]):
        raise FileNotFoundError("Training artifacts are missing. Run scripts/train_model.py first.")

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    return (
        config,
        joblib.load(model_path),
        metrics,
        pd.read_csv(scored_path),
        pd.read_csv(pr_curve_path),
        artifacts_dir,
    )


def load_feedback_frame(feedback_path: Path) -> pd.DataFrame:
    if feedback_path.exists():
        return pd.read_csv(feedback_path)
    return pd.DataFrame(
        columns=[
            "timestamp",
            "transaction_id",
            "analyst_name",
            "analyst_decision",
            "predicted_label",
            "risk_score",
            "comment",
        ]
    )


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


try:
    config, model, metrics, scored_transactions, pr_curve, artifacts_dir = load_assets()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

feature_columns = metrics["feature_columns"]
feedback_path = resolve_local_path(PROJECT_ROOT, config["feedback"]["output_path"])
feedback_frame = load_feedback_frame(feedback_path)
test_slice = scored_transactions.loc[scored_transactions["data_split"] == "test"].copy()

st.markdown(
    """
    <div class="hero">
        <h1>Fraud Radar Console</h1>
        <p>Detection, explicabilite SHAP et boucle de feedback pour les equipes risque.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analyse")
    transaction_options = scored_transactions["transaction_id"].astype(str).tolist()
    selected_transaction_id = st.selectbox("Transaction ID", transaction_options)
    st.caption("Selection par identifiant pour inspection rapide.")

recall_value = metrics["test_metrics"]["recall"]
precision_value = metrics["test_metrics"]["precision"]
estimated_savings = metrics["business_metrics"]["estimated_savings"]
alerts_volume = int(metrics["business_metrics"]["alerts_volume"])

metric_columns = st.columns(4)
metric_columns[0].metric("Recall", f"{recall_value:.1%}")
metric_columns[1].metric("Precision", f"{precision_value:.1%}")
metric_columns[2].metric("Estimated savings", format_currency(estimated_savings))
metric_columns[3].metric("Alerts volume", f"{alerts_volume}")

overview_left, overview_right = st.columns([1.2, 1.0])

with overview_left:
    st.subheader("Precision-Recall tradeoff")
    pr_figure = px.line(
        pr_curve,
        x="recall",
        y="precision",
        title="Precision-Recall curve",
        markers=False,
    )
    pr_figure.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(pr_figure, use_container_width=True)

with overview_right:
    st.subheader("Global risk mix")
    histogram = px.histogram(
        test_slice,
        x="predicted_probability",
        color=test_slice["actual_label"].map({0: "legit", 1: "fraud"}),
        nbins=30,
        barmode="overlay",
        color_discrete_map={"legit": "#2563eb", "fraud": "#b91c1c"},
        title="Distribution of risk scores on the test set",
    )
    histogram.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), legend_title_text="Actual label")
    st.plotly_chart(histogram, use_container_width=True)

selected_row = scored_transactions.loc[
    scored_transactions["transaction_id"].astype(str) == str(selected_transaction_id)
].iloc[0]
selected_frame = pd.DataFrame([selected_row[feature_columns]]).apply(pd.to_numeric, errors="coerce")
explanation = explain_transaction(model, selected_frame)
contributions = explanation["contributions"].head(10).copy()
contributions["direction"] = np.where(contributions["shap_value"] >= 0, "toward fraud", "toward legit")

local_left, local_right = st.columns([1.0, 1.1])

with local_left:
    st.subheader(f"Transaction {selected_transaction_id}")
    score = float(selected_row["predicted_probability"])
    predicted_alert = int(selected_row["predicted_label"])
    actual_label = int(selected_row["actual_label"])

    st.metric("Risk score", f"{score:.1%}", delta="alert" if predicted_alert == 1 else "no alert")
    st.write(
        "Top driver: "
        f"**{contributions.iloc[0]['feature']}**. "
        f"Positive fraud contribution share: **{explanation['positive_contribution_share']:.0%}**."
    )

    details = pd.DataFrame(
        [
            {"field": "Amount", "value": f"${selected_row['amount']:,.2f}"},
            {"field": "Actual label", "value": "fraud" if actual_label == 1 else "legit"},
            {"field": "Predicted label", "value": "alert" if predicted_alert == 1 else "clear"},
            {"field": "Distance from home", "value": f"{selected_row['distance_from_home_km']:.1f} km"},
            {"field": "Velocity 1h", "value": int(selected_row["velocity_1h"])},
            {"field": "International", "value": "yes" if int(selected_row["international"]) == 1 else "no"},
        ]
    )
    st.dataframe(details, use_container_width=True, hide_index=True)

with local_right:
    st.subheader("Local SHAP explanation")
    local_figure = px.bar(
        contributions.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"toward fraud": "#b91c1c", "toward legit": "#15803d"},
        title="Feature impact for the selected transaction",
    )
    local_figure.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
    st.plotly_chart(local_figure, use_container_width=True)

st.subheader("Global SHAP importance")
global_importance = global_shap_importance(model, scored_transactions[feature_columns], sample_size=500).head(10)
global_figure = px.bar(
    global_importance.sort_values("mean_abs_shap"),
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    color="mean_abs_shap",
    color_continuous_scale="Reds",
    title="Most influential drivers across transactions",
)
global_figure.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False)
st.plotly_chart(global_figure, use_container_width=True)

st.subheader("Feedback loop")
with st.form("feedback_form", clear_on_submit=True):
    analyst_name = st.text_input("Analyst name", value="risk_ops")
    analyst_decision = st.radio(
        "Decision",
        options=["confirmed_fraud", "false_positive"],
        horizontal=True,
    )
    comment = st.text_area("Comment", placeholder="Short analyst note for future retraining.")
    submitted = st.form_submit_button("Save analyst feedback")

if submitted:
    append_feedback(
        feedback_path=feedback_path,
        transaction_id=str(selected_transaction_id),
        analyst_decision=analyst_decision,
        comment=comment,
        risk_score=float(selected_row["predicted_probability"]),
        predicted_label=int(selected_row["predicted_label"]),
        analyst_name=analyst_name,
    )
    st.success("Feedback saved in data/feedback/analyst_feedback.csv")
    st.cache_resource.clear()
    st.rerun()

st.dataframe(load_feedback_frame(feedback_path).tail(10), use_container_width=True, hide_index=True)
