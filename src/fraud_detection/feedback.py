from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def append_feedback(
    feedback_path: str | Path,
    transaction_id: str,
    analyst_decision: str,
    comment: str,
    risk_score: float,
    predicted_label: int,
    analyst_name: str = "ops_analyst",
) -> None:
    output_path = Path(feedback_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feedback_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "transaction_id": transaction_id,
                "analyst_name": analyst_name,
                "analyst_decision": analyst_decision,
                "predicted_label": predicted_label,
                "risk_score": risk_score,
                "comment": comment.strip(),
            }
        ]
    )

    file_exists = output_path.exists()
    feedback_row.to_csv(output_path, mode="a", header=not file_exists, index=False)

