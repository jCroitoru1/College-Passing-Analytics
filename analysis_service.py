from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"
VISUALS_DIR = OUTPUT_DIR / "visuals"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cfb_multi_model_pipeline import run_pipeline  # noqa: E402


MODEL_FILES = {
    "Elite QB": "elite_qb",
    "Winning Traits": "winning_traits",
    "Offense Output": "offense_output",
    "Offensive Scheme": "offense_scheme",
    "Breakout": "breakout",
    "Transfer": "transfer",
}

CORE_DATASETS = {
    "Model Metrics": OUTPUT_DIR / "model_metrics.csv",
    "Model Summary": OUTPUT_DIR / "model_summary.md",
    "Team Dataset": OUTPUT_DIR / "qb_team_dataset.csv",
    "Offense Dataset": OUTPUT_DIR / "team_offense_dataset.csv",
    "Player Trajectory Dataset": OUTPUT_DIR / "player_trajectory_dataset.csv",
    "2025 Candidates": OUTPUT_DIR / "trajectory_2025_candidates.csv",
}


def outputs_exist() -> bool:
    return CORE_DATASETS["Model Metrics"].exists()


def run_analysis() -> Path:
    run_pipeline(BASE_DIR, OUTPUT_DIR)
    return OUTPUT_DIR


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def get_model_metrics() -> pd.DataFrame:
    return read_csv(CORE_DATASETS["Model Metrics"])


def get_model_summary() -> str:
    return read_text(CORE_DATASETS["Model Summary"])


def get_team_dataset() -> pd.DataFrame:
    return read_csv(CORE_DATASETS["Team Dataset"])


def get_offense_dataset() -> pd.DataFrame:
    return read_csv(CORE_DATASETS["Offense Dataset"])


def get_player_dataset() -> pd.DataFrame:
    return read_csv(CORE_DATASETS["Player Trajectory Dataset"])


def get_candidate_dataset() -> pd.DataFrame:
    return read_csv(CORE_DATASETS["2025 Candidates"])


def get_model_artifacts(model_label: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    slug = MODEL_FILES[model_label]
    importance = read_csv(OUTPUT_DIR / f"{slug}_feature_importance.csv")
    predictions = read_csv(OUTPUT_DIR / f"{slug}_predictions.csv")
    report = read_text(OUTPUT_DIR / f"{slug}_report.txt")
    return importance, predictions, report


def list_visual_paths() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*.png")) + sorted(VISUALS_DIR.glob("*.png"))


def latest_output_timestamp() -> str:
    existing_files = [path for path in OUTPUT_DIR.rglob("*") if path.is_file()]
    if not existing_files:
        return "No generated outputs yet"
    latest = max(existing_files, key=lambda path: path.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
