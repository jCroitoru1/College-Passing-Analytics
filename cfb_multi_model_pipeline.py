from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


POWER_CONFERENCES = {"SEC", "Big 12", "Big Ten", "ACC", "Pac-12"}
OUTPUT_DIRNAME = "outputs"
VISUALS_DIRNAME = "visuals"
REQUIRED_SKILL_COLUMNS = {
    "passing": ["Cmp", "Att", "Cmp%", "Yds", "TD", "TD%", "Int", "Int%", "Y/A", "AY/A", "Y/C", "Y/G", "Rate", "Awards"],
    "rushing": ["Rushing_Att", "Rushing_Yds", "Rushing_TD", "Rushing_Y/A", "Awards"],
    "receiving": ["Receiving_Rec", "Receiving_Yds", "Receiving_TD", "Receiving_Yards_Per_Rec", "Awards"],
}
GENERATED_FILE_HINTS = {
    "combined_cleaned",
    "players_dataset",
    "cluster",
    "metrics",
    "feature_importance",
    "season_summary",
    "qb_team_dataset",
    "team_offense_dataset",
    "player_trajectory_dataset",
    "trajectory_2025_candidates",
    "model_summary",
}
SEASON_TOKENS = {
    2020: ("twenty_twenty", "twenty_20", "2020", "twenty_rushing"),
    2021: ("twenty_one", "2021"),
    2022: ("twenty_two", "twetnty_two", "twnty_two", "2022"),
    2023: ("twenty_three", "twentythree", "2023"),
    2024: ("twenty_four", "twetnty_four", "2024"),
    2025: ("twenty_five", "25_", "2025"),
}


@dataclass
class ModelArtifacts:
    metrics: dict[str, float | str | int]
    importances: pd.DataFrame
    predictions: pd.DataFrame | None = None
    report_text: str | None = None


def slugify(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_person(value: str) -> str:
    value = str(value or "").replace("*", "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_team(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("&", " and ")
    replacements = {
        "st.": "state",
        "st ": "state ",
        " ole miss ": " mississippi ",
        "utsa": "texas san antonio",
        "ul lafayette": "louisiana",
        "miami oh": "miami ohio",
        "miami (oh)": "miami ohio",
        "miami-fl": "miami",
    }
    value = f" {value} "
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_conference(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s*\([^)]*\)", "", text)
    lowered = text.lower()
    mapping = {
        "sec": "SEC",
        "big ten": "Big Ten",
        "big 10": "Big Ten",
        "b1g": "Big Ten",
        "big 12": "Big 12",
        "acc": "ACC",
        "pac-12": "Pac-12",
        "pac 12": "Pac-12",
        "pac12": "Pac-12",
        "american": "American",
        "aac": "American",
        "cusa": "CUSA",
        "c-usa": "CUSA",
        "mwc": "MWC",
        "mountain west": "MWC",
        "sun belt": "Sun Belt",
        "mac": "MAC",
        "ind": "Independent",
        "independent": "Independent",
    }
    for source, target in mapping.items():
        if lowered == source:
            return target
    return text if text else "Unknown"


def is_power_conference(value: str) -> int:
    return int(normalize_conference(value) in POWER_CONFERENCES)


def infer_season(path: Path) -> int | None:
    lowered = path.stem.lower()
    for season, tokens in SEASON_TOKENS.items():
        if any(token in lowered for token in tokens):
            return season
    return None


def infer_dataset_type(path: Path) -> str | None:
    lowered = path.stem.lower()
    if "passing" in lowered:
        return "passing"
    if "rush" in lowered:
        return "rushing"
    if "reciev" in lowered or "receiv" in lowered:
        return "receiving"
    if "win_loss" in lowered or "wins_loss" in lowered:
        return "wins"
    return None


def should_skip_source(path: Path) -> bool:
    if OUTPUT_DIRNAME in {part.lower() for part in path.parts}:
        return True
    lowered = path.stem.lower()
    return any(hint in lowered for hint in GENERATED_FILE_HINTS)


def discover_sources(base_dir: Path) -> list[tuple[Path, str, int]]:
    sources: list[tuple[Path, str, int]] = []
    for pattern in ("*.csv", "*.xlsx", "*.xls"):
        for path in sorted(base_dir.rglob(pattern)):
            if should_skip_source(path):
                continue
            data_type = infer_dataset_type(path)
            season = infer_season(path)
            if data_type and season:
                sources.append((path, data_type, season))
    return sources


def file_looks_like_excel(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            signature = handle.read(4)
    except OSError:
        return False
    return signature == b"PK\x03\x04"


def load_source_file(path: Path) -> list[pd.DataFrame]:
    if path.suffix.lower() == ".csv" and not file_looks_like_excel(path):
        return [pd.read_csv(path)]

    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        warnings.warn(
            f"Skipping {path.name} because reading Excel requires openpyxl or xlrd: {exc}",
            stacklevel=2,
        )
        return []

    frames = []
    for sheet_name in workbook.sheet_names:
        sheet_df = workbook.parse(sheet_name)
        if not sheet_df.empty:
            frames.append(sheet_df)
    return frames


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def make_unique_columns(columns: Iterable[object]) -> list[str]:
    counts: dict[str, int] = {}
    unique_columns: list[str] = []
    for raw_column in columns:
        base = str(raw_column).strip()
        if base not in counts:
            counts[base] = 0
            unique_columns.append(base)
            continue
        counts[base] += 1
        unique_columns.append(f"{base}__dup{counts[base]}")
    return unique_columns


def coalesce_text_columns(frame: pd.DataFrame, candidate_columns: Iterable[str], target_column: str) -> pd.DataFrame:
    matched = [column for column in candidate_columns if column in frame.columns]
    if not matched:
        frame[target_column] = ""
        return frame

    if target_column not in frame.columns:
        frame[target_column] = ""

    filled = frame[matched].fillna("").astype(str)
    frame[target_column] = filled.apply(
        lambda row: next((value.strip() for value in row if value and value.strip()), ""),
        axis=1,
    )
    return frame


def standardize_skill_columns(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.copy()
    standardized.columns = make_unique_columns(standardized.columns)

    rename_map = {
        "Passing_Completions": "Cmp",
        "Passing_Attempts": "Att",
        "Passing_Completion_pct": "Cmp%",
        "Passing_Yards": "Yds",
        "Passing_TD": "TD",
        "Passing_TD_pct": "TD%",
        "Passing_INT": "Int",
        "Passing_INT_pct": "Int%",
        "Yards_per_Attempt": "Y/A",
        "Adjusted_Yards_per_Attempt": "AY/A",
        "Yards_per_Completion": "Y/C",
        "Yards_per_Game": "Y/G",
        "Passer_Rating": "Rate",
    }
    standardized = standardized.rename(columns=rename_map)

    award_like_columns = [
        column
        for column in standardized.columns
        if re.search(r"(^awards?$|_awards?$)", column, flags=re.IGNORECASE)
    ]
    standardized = coalesce_text_columns(standardized, award_like_columns, "Awards")
    return standardized


def ensure_skill_columns(frame: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    ensured = frame.copy()
    for column in REQUIRED_SKILL_COLUMNS.get(dataset_type, []):
        if column not in ensured.columns:
            ensured[column] = np.nan if column != "Awards" else ""
    return ensured


def clean_skill_frame(df: pd.DataFrame, season: int, dataset_type: str, source_name: str) -> pd.DataFrame:
    frame = ensure_skill_columns(standardize_skill_columns(df.copy()), dataset_type)
    if "Rk" not in frame.columns:
        return pd.DataFrame()

    frame = frame[pd.to_numeric(frame["Rk"], errors="coerce").notna()].copy()
    if frame.empty:
        return frame

    text_columns = [column for column in ["Player", "Team", "Conf", "Awards"] if column in frame.columns]
    for column in text_columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    frame["Season"] = season
    frame["SourceType"] = dataset_type
    frame["SourceFile"] = source_name
    frame["Player"] = frame.get("Player", pd.Series(index=frame.index, dtype="object")).astype(str).str.replace("*", "", regex=False).str.strip()
    frame["Team"] = frame.get("Team", pd.Series(index=frame.index, dtype="object")).astype(str).str.strip()
    frame["Conf"] = frame.get("Conf", pd.Series(index=frame.index, dtype="object")).apply(normalize_conference)
    frame["PlayerKey"] = frame["Player"].apply(normalize_person)
    frame["TeamKey"] = frame["Team"].apply(normalize_team)

    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"Player", "Team", "Conf", "Awards", "SourceType", "SourceFile", "PlayerKey", "TeamKey"}
    ]
    frame = _coerce_numeric(frame, numeric_columns)
    frame["IsPowerConference"] = frame["Conf"].apply(is_power_conference)
    return frame


def clean_wins_frame(df: pd.DataFrame, season: int, source_name: str) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = make_unique_columns(frame.columns)
    if "Rk" in frame.columns:
        frame = frame[pd.to_numeric(frame["Rk"], errors="coerce").notna()].copy()
    if "School" not in frame.columns or frame.empty:
        return pd.DataFrame()

    frame["Season"] = season
    frame["SourceFile"] = source_name
    frame["School"] = frame["School"].fillna("").astype(str).str.strip()
    frame["TeamKey"] = frame["School"].apply(normalize_team)
    frame["Conf"] = frame.get("Conf", pd.Series(index=frame.index, dtype="object")).apply(normalize_conference)
    frame["IsPowerConference"] = frame["Conf"].apply(is_power_conference)
    numeric_columns = [column for column in frame.columns if column not in {"School", "Conf", "SourceFile", "TeamKey"}]
    frame = _coerce_numeric(frame, numeric_columns)
    return frame


def load_all_data(base_dir: Path) -> dict[str, pd.DataFrame]:
    sources = discover_sources(base_dir)
    buckets: dict[str, list[pd.DataFrame]] = {"passing": [], "rushing": [], "receiving": [], "wins": []}

    for path, dataset_type, season in sources:
        for raw_frame in load_source_file(path):
            if dataset_type == "wins":
                cleaned = clean_wins_frame(raw_frame, season=season, source_name=path.name)
            else:
                cleaned = clean_skill_frame(raw_frame, season=season, dataset_type=dataset_type, source_name=path.name)
            if not cleaned.empty:
                buckets[dataset_type].append(cleaned)

    combined: dict[str, pd.DataFrame] = {}
    for dataset_type, frames in buckets.items():
        if not frames:
            combined[dataset_type] = pd.DataFrame()
            continue
        dataset = pd.concat(frames, ignore_index=True)
        dedupe_columns = [column for column in ["Season", "PlayerKey", "TeamKey", "SourceType"] if column in dataset.columns]
        if dataset_type == "wins":
            dedupe_columns = [column for column in ["Season", "TeamKey"] if column in dataset.columns]
        dataset = dataset.drop_duplicates(subset=dedupe_columns, keep="last").reset_index(drop=True)
        combined[dataset_type] = dataset

    return combined


def ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    ensured = frame.copy()
    for column in required_columns:
        if column not in ensured.columns:
            ensured[column] = np.nan
    return ensured


def build_team_dataset(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    passing = ensure_skill_columns(data["passing"].copy(), "passing")
    rushing = ensure_skill_columns(data["rushing"].copy(), "rushing")
    receiving = ensure_skill_columns(data["receiving"].copy(), "receiving")
    wins = data["wins"].copy()

    primary_qb = (
        passing.sort_values(["Season", "TeamKey", "Att", "Yds"], ascending=[True, True, False, False])
        .drop_duplicates(["Season", "TeamKey"])
        .copy()
    )

    passing_team = (
        passing.groupby(["Season", "TeamKey"], as_index=False)
        .agg(
            Team=("Team", "first"),
            Conf=("Conf", "first"),
            team_pass_att=("Att", "sum"),
            team_pass_cmp=("Cmp", "sum"),
            team_pass_yds=("Yds", "sum"),
            team_pass_td=("TD", "sum"),
            team_int=("Int", "sum"),
            qb_count=("PlayerKey", "nunique"),
        )
    )
    passing_team["team_cmp_pct"] = np.where(
        passing_team["team_pass_att"] > 0,
        100 * passing_team["team_pass_cmp"] / passing_team["team_pass_att"],
        np.nan,
    )
    passing_team["team_pass_yds_per_att"] = np.where(
        passing_team["team_pass_att"] > 0,
        passing_team["team_pass_yds"] / passing_team["team_pass_att"],
        np.nan,
    )

    rushing_team = (
        rushing.groupby(["Season", "TeamKey"], as_index=False)
        .agg(
            team_rush_att=("Rushing_Att", "sum"),
            team_rush_yds=("Rushing_Yds", "sum"),
            team_rush_td=("Rushing_TD", "sum"),
        )
    )
    rushing_team["team_rush_yds_per_att"] = np.where(
        rushing_team["team_rush_att"] > 0,
        rushing_team["team_rush_yds"] / rushing_team["team_rush_att"],
        np.nan,
    )
    top_rusher = (
        rushing.sort_values(["Season", "TeamKey", "Rushing_Yds"], ascending=[True, True, False])
        .drop_duplicates(["Season", "TeamKey"])[["Season", "TeamKey", "Player", "Rushing_Yds", "Rushing_TD"]]
        .rename(
            columns={
                "Player": "TopRusher",
                "Rushing_Yds": "top_rusher_yds",
                "Rushing_TD": "top_rusher_td",
            }
        )
    )
    rushing_team = rushing_team.merge(top_rusher, on=["Season", "TeamKey"], how="left")

    receiving_team = (
        receiving.groupby(["Season", "TeamKey"], as_index=False)
        .agg(
            team_rec=("Receiving_Rec", "sum"),
            team_rec_yds=("Receiving_Yds", "sum"),
            team_rec_td=("Receiving_TD", "sum"),
        )
    )
    receiving_team["team_rec_yds_per_rec"] = np.where(
        receiving_team["team_rec"] > 0,
        receiving_team["team_rec_yds"] / receiving_team["team_rec"],
        np.nan,
    )
    top_receiver = (
        receiving.sort_values(["Season", "TeamKey", "Receiving_Yds"], ascending=[True, True, False])
        .drop_duplicates(["Season", "TeamKey"])[["Season", "TeamKey", "Player", "Receiving_Yds", "Receiving_TD", "Receiving_Rec"]]
        .rename(
            columns={
                "Player": "TopReceiver",
                "Receiving_Yds": "top_receiver_yds",
                "Receiving_TD": "top_receiver_td",
                "Receiving_Rec": "top_receiver_rec",
            }
        )
    )
    receiving_team = receiving_team.merge(top_receiver, on=["Season", "TeamKey"], how="left")

    qb_rushing = (
        rushing[["Season", "TeamKey", "PlayerKey", "Rushing_Att", "Rushing_Yds", "Rushing_TD", "Rushing_Y/A"]]
        .rename(
            columns={
                "Rushing_Att": "qb_rush_att",
                "Rushing_Yds": "qb_rush_yds",
                "Rushing_TD": "qb_rush_td",
                "Rushing_Y/A": "qb_rush_yds_per_att",
            }
        )
    )

    primary_qb = primary_qb.merge(qb_rushing, on=["Season", "TeamKey", "PlayerKey"], how="left")
    primary_qb["qb_rush_att"] = primary_qb["qb_rush_att"].fillna(0)
    primary_qb["qb_rush_yds"] = primary_qb["qb_rush_yds"].fillna(0)
    primary_qb["qb_rush_td"] = primary_qb["qb_rush_td"].fillna(0)
    primary_qb["qb_rush_yds_per_att"] = primary_qb["qb_rush_yds_per_att"].fillna(0)

    qb_team = primary_qb.rename(
        columns={
            "Player": "QB",
            "Cmp%": "qb_cmp_pct",
            "Yds": "qb_pass_yds",
            "TD": "qb_pass_td",
            "Int": "qb_int",
            "Int%": "qb_int_pct",
            "Y/A": "qb_yds_per_att",
            "AY/A": "qb_adj_yds_per_att",
            "Y/G": "qb_pass_yds_per_game",
            "Rate": "qb_rating",
            "Att": "qb_att",
            "Cmp": "qb_cmp",
            "G": "qb_games",
            "Awards": "qb_awards",
            "TD%": "qb_td_pct",
        }
    )

    wins_subset = ensure_columns(
        wins,
        [
            "Season",
            "TeamKey",
            "School",
            "Conf",
            "W",
            "L",
            "Pct",
            "Off",
            "Def",
            "SRS",
            "SOS",
            "Conference_W",
            "Conference_L",
            "Conference_Pct",
            "AP High",
            "AP Rank",
            "AP Post",
            "IsPowerConference",
        ],
    )[
        [
            "Season",
            "TeamKey",
            "School",
            "Conf",
            "W",
            "L",
            "Pct",
            "Off",
            "Def",
            "SRS",
            "SOS",
            "Conference_W",
            "Conference_L",
            "Conference_Pct",
            "AP High",
            "AP Rank",
            "AP Post",
            "IsPowerConference",
        ]
    ].copy()

    team_df = (
        passing_team.merge(rushing_team, on=["Season", "TeamKey"], how="outer")
        .merge(receiving_team, on=["Season", "TeamKey"], how="outer")
        .merge(
            qb_team[
                [
                    "Season",
                    "TeamKey",
                    "QB",
                    "PlayerKey",
                    "Conf",
                    "qb_games",
                    "qb_cmp",
                    "qb_att",
                    "qb_cmp_pct",
                    "qb_pass_yds",
                    "qb_pass_td",
                    "qb_td_pct",
                    "qb_int",
                    "qb_int_pct",
                    "qb_yds_per_att",
                    "qb_adj_yds_per_att",
                    "qb_pass_yds_per_game",
                    "qb_rating",
                    "qb_awards",
                    "qb_rush_att",
                    "qb_rush_yds",
                    "qb_rush_td",
                    "qb_rush_yds_per_att",
                ]
            ],
            on=["Season", "TeamKey"],
            how="left",
            suffixes=("", "_qb"),
        )
        .merge(wins_subset, on=["Season", "TeamKey"], how="left", suffixes=("", "_wins"))
    )

    team_df["Team"] = team_df["School"].fillna(team_df["Team"])
    team_df["Conf"] = team_df["Conf_wins"].fillna(team_df["Conf"])
    if "IsPowerConference_wins" in team_df.columns:
        team_df["IsPowerConference"] = team_df["IsPowerConference_wins"].fillna(team_df["IsPowerConference"])
    team_df = team_df.drop(columns=[column for column in ["School", "Conf_wins", "IsPowerConference_wins"] if column in team_df.columns])

    numeric_fill_zero = [
        "team_pass_att",
        "team_pass_cmp",
        "team_pass_yds",
        "team_pass_td",
        "team_int",
        "team_rush_att",
        "team_rush_yds",
        "team_rush_td",
        "team_rec",
        "team_rec_yds",
        "team_rec_td",
        "qb_rush_att",
        "qb_rush_yds",
        "qb_rush_td",
        "qb_att",
        "qb_cmp",
        "qb_pass_yds",
        "qb_pass_td",
        "qb_int",
        "top_receiver_yds",
        "top_rusher_yds",
    ]
    for column in numeric_fill_zero:
        if column in team_df.columns:
            team_df[column] = pd.to_numeric(team_df[column], errors="coerce").fillna(0)

    for column in ["Pct", "Off", "Def", "SRS", "SOS", "Conference_Pct", "AP High", "AP Rank", "AP Post"]:
        if column in team_df.columns:
            team_df[column] = pd.to_numeric(team_df[column], errors="coerce")

    total_plays = team_df["team_rush_att"] + team_df["team_pass_att"]
    team_df["run_rate"] = np.where(total_plays > 0, team_df["team_rush_att"] / total_plays, np.nan)
    team_df["pass_rate"] = np.where(total_plays > 0, team_df["team_pass_att"] / total_plays, np.nan)
    team_df["team_total_yds"] = team_df["team_pass_yds"] + team_df["team_rush_yds"]
    team_df["team_total_td"] = team_df["team_pass_td"] + team_df["team_rush_td"]
    team_df["team_yds_per_play"] = np.where(total_plays > 0, team_df["team_total_yds"] / total_plays, np.nan)
    team_df["qb_total_yds"] = team_df["qb_pass_yds"] + team_df["qb_rush_yds"]
    team_df["qb_total_td"] = team_df["qb_pass_td"] + team_df["qb_rush_td"]
    team_df["qb_td_to_int_ratio"] = np.where(team_df["qb_int"] > 0, team_df["qb_pass_td"] / team_df["qb_int"], team_df["qb_pass_td"])
    team_df["qb_rush_share"] = np.where(team_df["qb_total_yds"] > 0, team_df["qb_rush_yds"] / team_df["qb_total_yds"], 0)
    team_df["top_receiver_share"] = np.where(team_df["team_rec_yds"] > 0, team_df["top_receiver_yds"] / team_df["team_rec_yds"], np.nan)
    team_df["top_rusher_share"] = np.where(team_df["team_rush_yds"] > 0, team_df["top_rusher_yds"] / team_df["team_rush_yds"], np.nan)
    team_df["wins"] = team_df["W"].fillna(0)
    team_df["losses"] = team_df["L"].fillna(0)
    team_df["games_played"] = team_df["wins"] + team_df["losses"]
    team_df["points_per_win"] = np.where(team_df["wins"] > 0, team_df["Off"] / team_df["wins"], np.nan)
    team_df["power_status"] = team_df["Conf"].apply(is_power_conference)

    team_df["elite_score"] = (
        team_df["qb_pass_yds"].rank(pct=True).fillna(0)
        + team_df["qb_pass_td"].rank(pct=True).fillna(0)
        + team_df["qb_adj_yds_per_att"].rank(pct=True).fillna(0)
        + team_df["qb_rating"].rank(pct=True).fillna(0)
        + team_df["Pct"].rank(pct=True).fillna(0)
        + team_df["Off"].rank(pct=True).fillna(0)
    ) / 6
    award_mask = team_df["qb_awards"].fillna("").str.contains(r"H-|Maxwell|AA|Camp", case=False, regex=True)
    team_df["EliteQB"] = ((team_df["elite_score"] >= 0.8) | award_mask).astype(int)

    return team_df.sort_values(["Season", "Team"]).reset_index(drop=True)


def assign_offensive_scheme(team_df: pd.DataFrame) -> pd.Series:
    pass_rate_cut = team_df["pass_rate"].median()
    run_rate_cut = team_df["run_rate"].median()
    air_yards_cut = team_df["qb_yds_per_att"].median()
    rush_eff_cut = team_df["team_rush_yds_per_att"].median()
    comp_cut = team_df["qb_cmp_pct"].median()

    labels = []
    for row in team_df.itertuples(index=False):
        if (
            pd.notna(row.run_rate)
            and row.run_rate >= max(0.56, run_rate_cut)
            and row.team_rush_yds_per_att >= rush_eff_cut
        ):
            labels.append("Smashmouth/Triple Option")
        elif (
            pd.notna(row.pass_rate)
            and row.pass_rate >= max(0.55, pass_rate_cut)
            and row.qb_yds_per_att >= air_yards_cut
        ):
            labels.append("Air Raid")
        elif pd.notna(row.qb_cmp_pct) and row.qb_cmp_pct >= comp_cut:
            labels.append("West Coast")
        elif row.pass_rate >= row.run_rate:
            labels.append("Air Raid")
        else:
            labels.append("Smashmouth/Triple Option")

    return pd.Series(labels, index=team_df.index, name="OffensiveScheme")


def build_player_dataset(data: dict[str, pd.DataFrame], team_df: pd.DataFrame) -> pd.DataFrame:
    passing_source = ensure_skill_columns(data["passing"].copy(), "passing")
    rushing_source = ensure_skill_columns(data["rushing"].copy(), "rushing")
    receiving_source = ensure_skill_columns(data["receiving"].copy(), "receiving")

    passing = passing_source[
        ["Season", "Player", "PlayerKey", "Team", "TeamKey", "Conf", "Awards", "G", "Att", "Yds", "TD", "Int", "Cmp%", "Y/A", "AY/A", "Rate"]
    ].rename(
        columns={
            "Awards": "pass_awards",
            "G": "games",
            "Att": "pass_att",
            "Yds": "pass_yds",
            "TD": "pass_td",
            "Int": "pass_int",
            "Cmp%": "pass_cmp_pct",
            "Y/A": "pass_yds_per_att",
            "AY/A": "pass_adj_yds_per_att",
            "Rate": "pass_rating",
        }
    )
    rushing = rushing_source[
        ["Season", "Player", "PlayerKey", "Team", "TeamKey", "Conf", "Awards", "G", "Rushing_Att", "Rushing_Yds", "Rushing_TD", "Rushing_Y/A"]
    ].rename(
        columns={
            "Awards": "rush_awards",
            "G": "games_rush",
            "Rushing_Att": "rush_att",
            "Rushing_Yds": "rush_yds",
            "Rushing_TD": "rush_td",
            "Rushing_Y/A": "rush_yds_per_att",
        }
    )
    receiving = receiving_source[
        ["Season", "Player", "PlayerKey", "Team", "TeamKey", "Conf", "Awards", "G", "Receiving_Rec", "Receiving_Yds", "Receiving_TD", "Receiving_Yards_Per_Rec"]
    ].rename(
        columns={
            "Awards": "rec_awards",
            "G": "games_rec",
            "Receiving_Rec": "rec",
            "Receiving_Yds": "rec_yds",
            "Receiving_TD": "rec_td",
            "Receiving_Yards_Per_Rec": "rec_yds_per_rec",
        }
    )

    player_df = (
        passing.merge(rushing, on=["Season", "PlayerKey", "TeamKey"], how="outer", suffixes=("", "_rush"))
        .merge(receiving, on=["Season", "PlayerKey", "TeamKey"], how="outer", suffixes=("", "_rec"))
    )

    for column in ["Player", "Team", "Conf"]:
        alternatives = [name for name in player_df.columns if name.startswith(column)]
        player_df[column] = player_df[alternatives].bfill(axis=1).iloc[:, 0]

    player_df["games"] = player_df[["games", "games_rush", "games_rec"]].max(axis=1)
    player_df["awards"] = (
        player_df[["pass_awards", "rush_awards", "rec_awards"]]
        .fillna("")
        .agg(",".join, axis=1)
        .str.strip(",")
    )

    numeric_columns = [
        "pass_att",
        "pass_yds",
        "pass_td",
        "pass_int",
        "pass_cmp_pct",
        "pass_yds_per_att",
        "pass_adj_yds_per_att",
        "pass_rating",
        "rush_att",
        "rush_yds",
        "rush_td",
        "rush_yds_per_att",
        "rec",
        "rec_yds",
        "rec_td",
        "rec_yds_per_rec",
    ]
    for column in numeric_columns:
        if column in player_df.columns:
            player_df[column] = pd.to_numeric(player_df[column], errors="coerce").fillna(0)

    player_df["Conf"] = player_df["Conf"].apply(normalize_conference)
    player_df["IsPowerConference"] = player_df["Conf"].apply(is_power_conference)
    player_df["total_yds"] = player_df["pass_yds"] + player_df["rush_yds"] + player_df["rec_yds"]
    player_df["total_td"] = player_df["pass_td"] + player_df["rush_td"] + player_df["rec_td"]
    player_df["usage"] = player_df["pass_att"] + player_df["rush_att"] + player_df["rec"]
    player_df["explosive_index"] = (
        player_df["pass_yds_per_att"] * np.where(player_df["pass_att"] > 0, 1, 0)
        + player_df["rush_yds_per_att"] * np.where(player_df["rush_att"] > 0, 1, 0)
        + player_df["rec_yds_per_rec"] * np.where(player_df["rec"] > 0, 1, 0)
    )

    player_df["Role"] = np.select(
        [
            player_df["pass_att"] >= 75,
            (player_df["rush_att"] >= player_df["rec"]) & (player_df["rush_yds"] >= player_df["rec_yds"]),
        ],
        ["QB", "RB"],
        default="WR/TE",
    )

    team_context = team_df[["Season", "TeamKey", "Off", "Pct", "OffensiveScheme", "power_status"]].rename(
        columns={"Pct": "team_win_pct", "power_status": "team_is_power"}
    )
    player_df = player_df.merge(team_context, on=["Season", "TeamKey"], how="left")

    player_df = player_df.sort_values(["PlayerKey", "Season"]).reset_index(drop=True)
    grouped = player_df.groupby("PlayerKey", group_keys=False)

    player_df["prior_total_yds"] = grouped["total_yds"].shift(1)
    player_df["prior_total_td"] = grouped["total_td"].shift(1)
    player_df["prior_usage"] = grouped["usage"].shift(1)
    player_df["prior_team_key"] = grouped["TeamKey"].shift(1)
    player_df["next_total_yds"] = grouped["total_yds"].shift(-1)
    player_df["next_total_td"] = grouped["total_td"].shift(-1)
    player_df["next_usage"] = grouped["usage"].shift(-1)
    player_df["next_team_key"] = grouped["TeamKey"].shift(-1)
    player_df["next_conf"] = grouped["Conf"].shift(-1)
    player_df["next_awards"] = grouped["awards"].shift(-1)

    player_df["yds_growth"] = player_df["total_yds"] - player_df["prior_total_yds"].fillna(0)
    player_df["td_growth"] = player_df["total_td"] - player_df["prior_total_td"].fillna(0)
    player_df["usage_growth"] = player_df["usage"] - player_df["prior_usage"].fillna(0)
    player_df["changed_team_from_prior"] = (player_df["TeamKey"] != player_df["prior_team_key"]).astype(int)
    player_df["next_is_power"] = player_df["next_conf"].fillna("").apply(is_power_conference)
    player_df["BreakoutNextSeason"] = (
        (player_df["next_total_yds"] >= np.maximum(player_df["total_yds"] * 1.35, player_df["total_yds"] + 300))
        & (player_df["next_total_td"] >= np.maximum(player_df["total_td"] + 3, player_df["total_td"] * 1.2))
        & (player_df["next_usage"] >= np.maximum(player_df["usage"] * 1.1, player_df["usage"] + 10))
    ).astype(int)
    player_df["NonP5ToP5TransferNext"] = (
        (player_df["IsPowerConference"] == 0)
        & (player_df["next_is_power"] == 1)
        & player_df["next_team_key"].notna()
        & (player_df["next_team_key"] != player_df["TeamKey"])
    ).astype(int)
    player_df["NextSeasonAvailable"] = player_df["next_total_yds"].notna().astype(int)
    return player_df


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def feature_importance_frame(model: Pipeline, preprocessor_name: str, estimator_name: str) -> pd.DataFrame:
    feature_names = model.named_steps[preprocessor_name].get_feature_names_out()
    estimator = model.named_steps[estimator_name]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def safe_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = False,
):
    stratifier = None
    if stratify:
        counts = y.value_counts(dropna=False)
        if len(counts) > 1 and counts.min() >= 2:
            stratifier = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratifier)


def run_classifier(
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    categorical_features: list[str],
    positive_label: str | int | None = None,
) -> ModelArtifacts:
    model_df = df.dropna(subset=[target]).copy()
    if model_df[target].nunique() < 2:
        return ModelArtifacts(
            metrics={"model": target, "warning": "Target does not have at least two classes."},
            importances=pd.DataFrame(columns=["feature", "importance"]),
        )

    X = model_df[feature_columns]
    y = model_df[target]
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    X_train, X_test, y_train, y_test = safe_train_test(X, y, stratify=True)

    model = Pipeline(
        steps=[
            ("prep", make_preprocessor(numeric_features, categorical_features)),
            ("clf", RandomForestClassifier(n_estimators=450, min_samples_leaf=2, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "model": target,
        "rows": int(len(model_df)),
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "weighted_f1": round(f1_score(y_test, preds, average="weighted"), 4),
    }

    if positive_label is not None and hasattr(model.named_steps["clf"], "predict_proba") and len(y.unique()) == 2:
        probabilities = model.predict_proba(X_test)
        class_index = list(model.named_steps["clf"].classes_).index(positive_label)
        metrics["roc_auc"] = round(roc_auc_score(y_test, probabilities[:, class_index]), 4)

    labels = sorted(pd.Series(y).dropna().unique().tolist())
    report = classification_report(y_test, preds, zero_division=0)
    confusion = confusion_matrix(y_test, preds, labels=labels)
    report_text = report + "\nConfusion Matrix\n" + pd.DataFrame(confusion, index=labels, columns=labels).to_string()

    full_predictions = model_df[["Season"]].copy()
    for column in ["Team", "QB", "Player"]:
        if column in model_df.columns:
            full_predictions[column] = model_df[column]
    full_predictions["actual"] = y
    full_predictions["predicted"] = model.predict(X)
    if hasattr(model.named_steps["clf"], "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = model.named_steps["clf"].classes_
        for index, class_name in enumerate(classes):
            full_predictions[f"prob_{slugify(class_name)}"] = probabilities[:, index]

    return ModelArtifacts(
        metrics=metrics,
        importances=feature_importance_frame(model, "prep", "clf"),
        predictions=full_predictions,
        report_text=report_text,
    )


def run_regressor(
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    categorical_features: list[str],
) -> ModelArtifacts:
    model_df = df.dropna(subset=[target]).copy()
    X = model_df[feature_columns]
    y = model_df[target]
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    X_train, X_test, y_train, y_test = safe_train_test(X, y, stratify=False)

    model = Pipeline(
        steps=[
            ("prep", make_preprocessor(numeric_features, categorical_features)),
            ("reg", RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "model": target,
        "rows": int(len(model_df)),
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 4),
    }

    full_predictions = model_df[["Season"]].copy()
    for column in ["Team", "QB", "Player"]:
        if column in model_df.columns:
            full_predictions[column] = model_df[column]
    full_predictions["actual"] = y
    full_predictions["predicted"] = model.predict(X)

    return ModelArtifacts(
        metrics=metrics,
        importances=feature_importance_frame(model, "prep", "reg"),
        predictions=full_predictions,
    )


def create_visualizations(team_df: pd.DataFrame, player_df: pd.DataFrame, output_dir: Path) -> None:
    visuals_dir = output_dir / VISUALS_DIRNAME
    visuals_dir.mkdir(parents=True, exist_ok=True)

    corr_columns = [
        "qb_cmp_pct",
        "qb_yds_per_att",
        "qb_adj_yds_per_att",
        "qb_rating",
        "qb_rush_yds",
        "run_rate",
        "Off",
        "Pct",
    ]
    corr_frame = team_df[corr_columns].apply(pd.to_numeric, errors="coerce")
    corr = corr_frame.corr().fillna(0)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("QB Traits, Team Offense, and Winning Correlations")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(visuals_dir / "qb_traits_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    elite_colors = team_df["EliteQB"].map({0: "#87a8d0", 1: "#d1495b"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(team_df["qb_adj_yds_per_att"], team_df["Pct"], c=elite_colors, alpha=0.75)
    ax.set_xlabel("QB Adjusted Yards per Attempt")
    ax.set_ylabel("Team Win Percentage")
    ax.set_title("Elite QB Label vs Winning")
    fig.tight_layout()
    fig.savefig(visuals_dir / "elite_qb_vs_winning.png", dpi=180)
    plt.close(fig)

    scheme_palette = {
        "Air Raid": "#457b9d",
        "West Coast": "#2a9d8f",
        "Smashmouth/Triple Option": "#bc6c25",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for scheme, subset in team_df.groupby("OffensiveScheme"):
        ax.scatter(subset["run_rate"], subset["Off"], label=scheme, alpha=0.7, color=scheme_palette.get(scheme))
    ax.set_xlabel("Run Rate")
    ax.set_ylabel("Points Per Game")
    ax.set_title("Offensive Output by Inferred Scheme")
    ax.legend()
    fig.tight_layout()
    fig.savefig(visuals_dir / "scheme_vs_offense.png", dpi=180)
    plt.close(fig)

    latest = player_df[player_df["Season"] == player_df["Season"].max()].copy()
    candidate_columns = [column for column in ["breakout_probability", "transfer_probability"] if column in latest.columns]
    if len(candidate_columns) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        role_palette = {"QB": "#e76f51", "RB": "#e9c46a", "WR/TE": "#264653"}
        for role, subset in latest.groupby("Role"):
            ax.scatter(
                subset["breakout_probability"],
                subset["transfer_probability"],
                label=role,
                alpha=0.7,
                color=role_palette.get(role),
            )
        ax.set_xlabel("Breakout Probability")
        ax.set_ylabel("Non-P5 to P5 Transfer Probability")
        ax.set_title(f"{int(latest['Season'].max())} Player Trajectory Candidates")
        ax.legend()
        fig.tight_layout()
        fig.savefig(visuals_dir / "trajectory_candidates.png", dpi=180)
        plt.close(fig)


def write_markdown_summary(
    output_dir: Path,
    *,
    team_df: pd.DataFrame,
    player_df: pd.DataFrame,
    model_metrics: list[dict[str, float | str | int]],
) -> None:
    latest_season = int(team_df["Season"].max())
    elite_rate = round(team_df["EliteQB"].mean() * 100, 1)
    scheme_mix = team_df["OffensiveScheme"].value_counts(normalize=True).mul(100).round(1).to_dict()
    latest_candidates = player_df[player_df["Season"] == latest_season].copy()

    lines = [
        "# CFB Multi-Model Summary",
        "",
        "## Data coverage",
        f"- Team seasons modeled: {len(team_df)}",
        f"- Player seasons modeled: {len(player_df)}",
        f"- Elite QB label rate: {elite_rate}%",
        f"- Latest scoring season used for candidate outputs: {latest_season}",
        "",
        "## Inferred scheme mix",
    ]
    for scheme, pct in scheme_mix.items():
        lines.append(f"- {scheme}: {pct}%")

    lines.extend(["", "## Model metrics"])
    for metric_row in model_metrics:
        lines.append(f"- {json.dumps(metric_row, default=str)}")

    if {"breakout_probability", "transfer_probability"}.issubset(latest_candidates.columns):
        top_breakout = latest_candidates.sort_values("breakout_probability", ascending=False).head(10)
        top_transfer = latest_candidates.sort_values("transfer_probability", ascending=False).head(10)

        lines.extend(["", "## Top breakout candidates"])
        for row in top_breakout.itertuples(index=False):
            lines.append(
                f"- {row.Player} ({row.Team}, {row.Role}) breakout probability {row.breakout_probability:.3f}"
            )

        lines.extend(["", "## Top non-P5 to P5 transfer candidates"])
        for row in top_transfer.itertuples(index=False):
            lines.append(
                f"- {row.Player} ({row.Team}, {row.Conf}) transfer probability {row.transfer_probability:.3f}"
            )

    (output_dir / "model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def score_binary_candidates(
    training_df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    categorical_features: list[str],
    score_column: str,
) -> pd.Series:
    model_df = training_df.dropna(subset=[target]).copy()
    if model_df[target].nunique() < 2:
        return pd.Series(np.nan, index=scoring_df.index, name=score_column)

    X_train = model_df[feature_columns]
    y_train = model_df[target]
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    model = Pipeline(
        steps=[
            ("prep", make_preprocessor(numeric_features, categorical_features)),
            ("clf", RandomForestClassifier(n_estimators=450, min_samples_leaf=2, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(scoring_df[feature_columns])
    class_index = list(model.named_steps["clf"].classes_).index(1)
    return pd.Series(probabilities[:, class_index], index=scoring_df.index, name=score_column)


def run_pipeline(base_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data(base_dir)
    if data["passing"].empty or data["wins"].empty:
        raise RuntimeError("Passing and win/loss data are required to build the requested models.")

    team_df = build_team_dataset(data)
    team_df["OffensiveScheme"] = assign_offensive_scheme(team_df)
    player_df = build_player_dataset(data, team_df)

    qb_features = [
        "Season",
        "Conf",
        "power_status",
        "qb_games",
        "qb_att",
        "qb_cmp_pct",
        "qb_td_pct",
        "qb_int_pct",
        "qb_yds_per_att",
        "qb_adj_yds_per_att",
        "qb_rating",
        "qb_rush_att",
        "qb_rush_yds",
        "qb_rush_td",
        "top_receiver_yds",
        "top_rusher_yds",
        "run_rate",
        "team_yds_per_play",
    ]
    winning_trait_features = [
        "Season",
        "Conf",
        "power_status",
        "qb_att",
        "qb_cmp_pct",
        "qb_td_pct",
        "qb_int_pct",
        "qb_yds_per_att",
        "qb_adj_yds_per_att",
        "qb_rating",
        "qb_rush_yds",
        "top_receiver_yds",
        "team_rush_yds",
        "run_rate",
        "Off",
    ]
    offense_features = [
        "Season",
        "Conf",
        "power_status",
        "qb_cmp_pct",
        "qb_yds_per_att",
        "qb_adj_yds_per_att",
        "qb_rating",
        "qb_rush_yds",
        "team_pass_att",
        "team_pass_yds",
        "team_rush_att",
        "team_rush_yds",
        "team_rush_yds_per_att",
        "top_receiver_yds",
        "top_rusher_yds",
        "run_rate",
        "pass_rate",
        "team_yds_per_play",
    ]
    scheme_features = [
        "Season",
        "Conf",
        "power_status",
        "qb_cmp_pct",
        "qb_yds_per_att",
        "qb_adj_yds_per_att",
        "qb_rating",
        "team_pass_att",
        "team_pass_yds",
        "team_rush_att",
        "team_rush_yds",
        "team_rush_yds_per_att",
        "run_rate",
        "pass_rate",
        "top_receiver_share",
        "top_rusher_share",
        "team_yds_per_play",
    ]
    trajectory_features = [
        "Season",
        "Role",
        "Conf",
        "IsPowerConference",
        "games",
        "pass_att",
        "pass_yds",
        "pass_td",
        "pass_int",
        "pass_cmp_pct",
        "pass_yds_per_att",
        "pass_adj_yds_per_att",
        "pass_rating",
        "rush_att",
        "rush_yds",
        "rush_td",
        "rush_yds_per_att",
        "rec",
        "rec_yds",
        "rec_td",
        "rec_yds_per_rec",
        "total_yds",
        "total_td",
        "usage",
        "explosive_index",
        "prior_total_yds",
        "prior_total_td",
        "prior_usage",
        "yds_growth",
        "td_growth",
        "usage_growth",
        "changed_team_from_prior",
        "Off",
        "team_win_pct",
        "team_is_power",
        "OffensiveScheme",
    ]
    categorical_team = ["Conf"]
    categorical_scheme = ["Conf"]
    categorical_trajectory = ["Role", "Conf", "OffensiveScheme"]

    elite_qb_results = run_classifier(
        team_df,
        target="EliteQB",
        feature_columns=qb_features,
        categorical_features=categorical_team,
        positive_label=1,
    )
    win_pct_results = run_regressor(
        team_df.dropna(subset=["Pct"]),
        target="Pct",
        feature_columns=winning_trait_features,
        categorical_features=categorical_team,
    )
    offense_results = run_regressor(
        team_df.dropna(subset=["Off"]),
        target="Off",
        feature_columns=offense_features,
        categorical_features=categorical_team,
    )
    scheme_results = run_classifier(
        team_df,
        target="OffensiveScheme",
        feature_columns=scheme_features,
        categorical_features=categorical_scheme,
    )

    trajectory_training = player_df[player_df["NextSeasonAvailable"] == 1].copy()
    breakout_results = run_classifier(
        trajectory_training,
        target="BreakoutNextSeason",
        feature_columns=trajectory_features,
        categorical_features=categorical_trajectory,
        positive_label=1,
    )
    transfer_results = run_classifier(
        trajectory_training,
        target="NonP5ToP5TransferNext",
        feature_columns=trajectory_features,
        categorical_features=categorical_trajectory,
        positive_label=1,
    )

    latest_player_mask = player_df["Season"] == player_df["Season"].max()
    player_df["breakout_probability"] = np.nan
    player_df["transfer_probability"] = np.nan
    player_df.loc[latest_player_mask, "breakout_probability"] = score_binary_candidates(
        trajectory_training,
        player_df.loc[latest_player_mask].copy(),
        target="BreakoutNextSeason",
        feature_columns=trajectory_features,
        categorical_features=categorical_trajectory,
        score_column="breakout_probability",
    )
    player_df.loc[latest_player_mask, "transfer_probability"] = score_binary_candidates(
        trajectory_training,
        player_df.loc[latest_player_mask].copy(),
        target="NonP5ToP5TransferNext",
        feature_columns=trajectory_features,
        categorical_features=categorical_trajectory,
        score_column="transfer_probability",
    )

    create_visualizations(team_df, player_df, output_dir)

    team_df.to_csv(output_dir / "qb_team_dataset.csv", index=False)
    team_df.to_csv(output_dir / "team_offense_dataset.csv", index=False)
    player_df.to_csv(output_dir / "player_trajectory_dataset.csv", index=False)

    latest_candidates = player_df[player_df["Season"] == player_df["Season"].max()].copy()
    latest_candidates = latest_candidates.sort_values(
        ["breakout_probability", "transfer_probability"],
        ascending=[False, False],
        na_position="last",
    )
    latest_candidates.to_csv(output_dir / "trajectory_2025_candidates.csv", index=False)

    model_metrics = [
        elite_qb_results.metrics,
        win_pct_results.metrics,
        offense_results.metrics,
        scheme_results.metrics,
        breakout_results.metrics,
        transfer_results.metrics,
    ]
    pd.DataFrame(model_metrics).to_csv(output_dir / "model_metrics.csv", index=False)

    artifact_map = {
        "elite_qb": elite_qb_results,
        "winning_traits": win_pct_results,
        "offense_output": offense_results,
        "offense_scheme": scheme_results,
        "breakout": breakout_results,
        "transfer": transfer_results,
    }
    for name, artifacts in artifact_map.items():
        artifacts.importances.to_csv(output_dir / f"{name}_feature_importance.csv", index=False)
        if artifacts.predictions is not None:
            artifacts.predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        if artifacts.report_text:
            (output_dir / f"{name}_report.txt").write_text(artifacts.report_text, encoding="utf-8")

    write_markdown_summary(output_dir, team_df=team_df, player_df=player_df, model_metrics=model_metrics)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / OUTPUT_DIRNAME
    run_pipeline(base_dir, output_dir)
