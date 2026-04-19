from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.config import load_project_config, resolve_local_path
from fraud_detection.demo_data import generate_demo_transactions
from fraud_detection.training import train_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bank fraud detection project.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
        help="Path to the project config file.",
    )
    args = parser.parse_args()

    config = load_project_config(args.config)
    data_path = resolve_local_path(PROJECT_ROOT, config["data"]["input_path"])

    if not data_path.exists():
        row_count = int(config["training"].get("n_demo_rows_if_missing", 15000))
        data_path.parent.mkdir(parents=True, exist_ok=True)
        demo_dataset = generate_demo_transactions(
            n_samples=row_count,
            random_state=int(config["data"]["random_state"]),
        )
        demo_dataset.to_csv(data_path, index=False)
        print(f"No input dataset found. Demo dataset generated at: {data_path}")

    summary = train_project(PROJECT_ROOT, args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

