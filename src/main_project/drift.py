"""
drift.py

Offline prediction drift monitoring using Evidently.

This script compares a reference set of model predictions with
current (production) predictions logged by the FastAPI /predict endpoint.
It generates an Evidently HTML report showing drift in predicted class
distribution and confidence scores.

How to run:
-----------
1. Make sure the FastAPI service has been running and predictions
   have been logged to `predictions.csv`.       --- uv run uvicorn src.main_project.api:app --reload --port 8000

2. Create a reference dataset (for example, an earlier snapshot):
   cp predictions.csv reference.csv

3. Run the drift report:
uv run python -m main_project.drift \
     --reference_data reference.csv \
     --current_data predictions.csv

Output:
-------
An HTML drift report will be saved to the specified output directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

SCRIPT_DIR = Path(__file__).resolve().parent  # src/main_project

def main():
    ap = argparse.ArgumentParser(description="Data Drift Detection using Evidently")
    ap.add_argument("--reference_data", type=Path, required=True, help="Path to the reference dataset CSV file")
    ap.add_argument("--current_data", type=Path, required=True, help="Path to the current dataset CSV file")
    ap.add_argument("--output_report", type=Path, default=Path("drift_reports/drift"), help="Path to save the drift report HTML file")
    args = ap.parse_args()

    ref = pd.read_csv(args.reference_data)
    cur = pd.read_csv(args.current_data)

    cols = ['predicted_class', 'confidence']
    ref = ref[cols].dropna()
    cur = cur[cols].dropna()

    out_dir = SCRIPT_DIR / args.output_report
    out_dir.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)


    out_html = out_dir / "predict_drift_report.html"
    report.save_html(str(out_html))
    print(f"Drift report saved to: {out_html}")

if __name__ == "__main__":
    main()
