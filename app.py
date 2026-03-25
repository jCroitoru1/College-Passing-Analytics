from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis_service import (
    CORE_DATASETS,
    MODEL_FILES,
    get_candidate_dataset,
    get_model_artifacts,
    get_model_metrics,
    get_model_summary,
    get_offense_dataset,
    get_player_dataset,
    get_team_dataset,
    latest_output_timestamp,
    list_visual_paths,
    outputs_exist,
    run_analysis,
)


st.set_page_config(
    page_title="CFB Analysis Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(245, 176, 65, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(52, 152, 219, 0.16), transparent 28%),
            linear-gradient(180deg, #f6f1e8 0%, #fcfaf6 55%, #eef4f8 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(22, 44, 63, 0.10);
        border-radius: 18px;
        background: rgba(255, 252, 246, 0.85);
        box-shadow: 0 10px 30px rgba(22, 44, 63, 0.07);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        color: #102a43;
        font-size: 2.3rem;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        color: #334e68;
        font-size: 1rem;
    }
    .metric-card {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(16, 42, 67, 0.08);
        background: rgba(255, 255, 255, 0.75);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> pd.DataFrame:
    return get_model_metrics()


@st.cache_data(show_spinner=False)
def load_model_summary() -> str:
    return get_model_summary()


@st.cache_data(show_spinner=False)
def load_team_dataset() -> pd.DataFrame:
    return get_team_dataset()


@st.cache_data(show_spinner=False)
def load_offense_dataset() -> pd.DataFrame:
    return get_offense_dataset()


@st.cache_data(show_spinner=False)
def load_player_dataset() -> pd.DataFrame:
    return get_player_dataset()


@st.cache_data(show_spinner=False)
def load_candidate_dataset() -> pd.DataFrame:
    return get_candidate_dataset()


@st.cache_data(show_spinner=False)
def load_model_artifacts(model_label: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    return get_model_artifacts(model_label)


@st.cache_data(show_spinner=False)
def load_visual_paths() -> list[str]:
    return [str(path) for path in list_visual_paths()]


def clear_caches() -> None:
    load_model_metrics.clear()
    load_model_summary.clear()
    load_team_dataset.clear()
    load_offense_dataset.clear()
    load_player_dataset.clear()
    load_candidate_dataset.clear()
    load_model_artifacts.clear()
    load_visual_paths.clear()


def format_metric_value(value: float | int | str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, str):
        return value
    if abs(float(value)) <= 1:
        return f"{float(value):.3f}"
    return f"{float(value):,.3f}"


def render_overview(metrics_df: pd.DataFrame, summary_text: str, team_df: pd.DataFrame, player_df: pd.DataFrame) -> None:
    st.subheader("Overview")
    latest_season = int(player_df["Season"].max()) if not player_df.empty and "Season" in player_df.columns else "n/a"
    total_team_rows = int(len(team_df)) if not team_df.empty else 0
    total_player_rows = int(len(player_df)) if not player_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", len(metrics_df))
    col2.metric("Team Seasons", f"{total_team_rows:,}")
    col3.metric("Player Seasons", f"{total_player_rows:,}")
    col4.metric("Latest Season", latest_season)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### Summary")
        st.markdown(summary_text or "No summary generated yet.")
    with right:
        st.markdown("##### Data Products")
        dataset_rows = []
        for label, path in CORE_DATASETS.items():
            dataset_rows.append(
                {
                    "Artifact": label,
                    "Exists": path.exists(),
                    "Path": str(path),
                }
            )
        st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True, hide_index=True)


def render_models(metrics_df: pd.DataFrame) -> None:
    st.subheader("Models")
    if metrics_df.empty:
        st.info("Run the analysis to populate model metrics.")
        return

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    selected_model = st.selectbox("Inspect model artifacts", list(MODEL_FILES))
    importance_df, predictions_df, report_text = load_model_artifacts(selected_model)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### Feature Importance")
        if importance_df.empty:
            st.info("No feature importance file found for this model.")
        else:
            top_n = st.slider("Top features", min_value=5, max_value=min(30, len(importance_df)), value=min(15, len(importance_df)))
            chart_df = importance_df.head(top_n).sort_values("importance")
            st.bar_chart(chart_df.set_index("feature")["importance"])
            st.dataframe(importance_df.head(top_n), use_container_width=True, hide_index=True)

    with right:
        st.markdown("##### Model Outputs")
        if report_text:
            st.code(report_text)
        elif not predictions_df.empty:
            st.dataframe(predictions_df.head(25), use_container_width=True, hide_index=True)
        else:
            st.info("No report or prediction sample is available for this model.")


def render_candidates(candidates_df: pd.DataFrame) -> None:
    st.subheader("Candidates")
    if candidates_df.empty:
        st.info("No candidate table found yet.")
        return

    role_options = ["All"] + sorted(candidates_df["Role"].dropna().astype(str).unique().tolist()) if "Role" in candidates_df.columns else ["All"]
    team_options = ["All"] + sorted(candidates_df["Team"].dropna().astype(str).unique().tolist()) if "Team" in candidates_df.columns else ["All"]

    col1, col2, col3 = st.columns(3)
    selected_role = col1.selectbox("Role", role_options)
    selected_team = col2.selectbox("Team", team_options)
    sort_column = col3.selectbox(
        "Sort by",
        [column for column in ["breakout_probability", "transfer_probability", "total_yds", "total_td", "usage"] if column in candidates_df.columns],
    )

    filtered_df = candidates_df.copy()
    if selected_role != "All":
        filtered_df = filtered_df[filtered_df["Role"].astype(str) == selected_role]
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["Team"].astype(str) == selected_team]

    filtered_df = filtered_df.sort_values(sort_column, ascending=False, na_position="last")
    st.dataframe(filtered_df.head(100), use_container_width=True, hide_index=True)


def render_data_explorer(team_df: pd.DataFrame, offense_df: pd.DataFrame, player_df: pd.DataFrame) -> None:
    st.subheader("Data Explorer")
    dataset_name = st.radio(
        "Dataset",
        ["Team", "Offense", "Player Trajectory"],
        horizontal=True,
    )
    dataset_map = {
        "Team": team_df,
        "Offense": offense_df,
        "Player Trajectory": player_df,
    }
    selected_df = dataset_map[dataset_name]
    if selected_df.empty:
        st.info(f"No {dataset_name.lower()} dataset is available yet.")
        return

    searchable_columns = [col for col in ["Player", "Team", "Conf", "Season", "Role", "OffensiveScheme"] if col in selected_df.columns]
    query = st.text_input("Quick filter")

    working_df = selected_df.copy()
    if query and searchable_columns:
        mask = pd.Series(False, index=working_df.index)
        lowered = query.lower()
        for column in searchable_columns:
            mask = mask | working_df[column].astype(str).str.lower().str.contains(lowered, na=False)
        working_df = working_df[mask]

    st.caption(f"{len(working_df):,} rows")
    st.dataframe(working_df.head(500), use_container_width=True, hide_index=True)


def render_visuals() -> None:
    st.subheader("Visuals")
    visual_paths = load_visual_paths()
    if not visual_paths:
        st.info("No generated visuals found yet.")
        return

    columns = st.columns(2)
    for index, visual_path in enumerate(visual_paths):
        with columns[index % 2]:
            st.image(visual_path, caption=visual_path.split("\\")[-1], use_container_width=True)


st.markdown(
    """
    <div class="hero">
        <h1>College Football Analysis Lab</h1>
        <p>Explore Potential Changes and how College Offenses Work.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    st.write("Use the existing analysis pipeline as the app backend.")
    st.caption(f"Last output update: {latest_output_timestamp()}")
    run_clicked = st.button("Run / Refresh Analysis", use_container_width=True, type="primary")

    if run_clicked:
        with st.spinner("Running pipeline and rebuilding outputs..."):
            run_analysis()
        clear_caches()
        st.success("Analysis complete. Outputs refreshed.")
        st.rerun()

    st.divider()
    st.write("App location")
    st.code("streamlit/app.py")


if not outputs_exist():
    st.warning("No generated outputs were found. Run the analysis from the sidebar to build the app data.")
    st.stop()


metrics_df = load_model_metrics()
summary_text = load_model_summary()
team_df = load_team_dataset()
offense_df = load_offense_dataset()
player_df = load_player_dataset()
candidates_df = load_candidate_dataset()

overview_tab, models_tab, candidates_tab, data_tab, visuals_tab = st.tabs(
    ["Overview", "Models", "Candidates", "Data Explorer", "Visuals"]
)

with overview_tab:
    render_overview(metrics_df, summary_text, team_df, player_df)

with models_tab:
    render_models(metrics_df)

with candidates_tab:
    render_candidates(candidates_df)

with data_tab:
    render_data_explorer(team_df, offense_df, player_df)

with visuals_tab:
    render_visuals()
