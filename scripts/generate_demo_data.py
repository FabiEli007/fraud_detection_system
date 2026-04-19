from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.config import load_project_config, resolve_local_path
from fraud_detection.demo_data import generate_demo_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an imbalanced fraud dataset for demo usage.")
    parser.add_argument("--rows", type=int, default=15000, help="Number of rows to generate.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
        help="Path to the project config file.",
    )
    args = parser.parse_args()

    config = load_project_config(args.config)
    data_path = resolve_local_path(PROJECT_ROOT, config["data"]["input_path"])
    data_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_demo_transactions(n_samples=args.rows, random_state=int(config["data"]["random_state"]))
    dataset.to_csv(data_path, index=False)

    print(f"Demo dataset written to: {data_path}")
    print(f"Fraud rate: {dataset['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()

