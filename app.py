from __future__ import annotations

import pandas as pd
import plotly.express as px
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
            radial-gradient(circle at top left, rgba(232, 180, 94, 0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(49, 130, 189, 0.18), transparent 26%),
            linear-gradient(180deg, #f7f4ed 0%, #fbfaf6 48%, #eef5f8 100%);
    }
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.03em;
        color: #102a43;
    }
    .hero {
        padding: 1.6rem;
        border: 1px solid rgba(16, 42, 67, 0.10);
        border-radius: 22px;
        background: rgba(255, 252, 246, 0.88);
        box-shadow: 0 14px 38px rgba(16, 42, 67, 0.08);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .hero p {
        margin: 0.65rem 0 0 0;
        color: #334e68;
        font-size: 1.02rem;
        line-height: 1.5;
    }
    .info-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(16, 42, 67, 0.08);
        background: rgba(255, 255, 255, 0.78);
        min-height: 170px;
    }
    .info-card h4 {
        margin-top: 0;
        margin-bottom: 0.55rem;
        color: #12324a;
    }
    .info-card p {
        margin: 0;
        color: #486581;
        line-height: 1.5;
    }
    .callout {
        padding: 1rem 1.15rem;
        border-left: 5px solid #d97706;
        border-radius: 0 14px 14px 0;
        background: rgba(255, 247, 237, 0.95);
        color: #7c2d12;
        margin-bottom: 1rem;
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


def clear_caches() -> None:
    load_model_metrics.clear()
    load_model_summary.clear()
    load_team_dataset.clear()
    load_offense_dataset.clear()
    load_player_dataset.clear()
    load_candidate_dataset.clear()
    load_model_artifacts.clear()


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    working_df = df.copy()
    for column in columns:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
    return working_df


def format_metric_value(value: float | int | str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, str):
        return value
    numeric_value = float(value)
    if abs(numeric_value) <= 1:
        return f"{numeric_value:.3f}"
    if abs(numeric_value) < 100:
        return f"{numeric_value:.1f}"
    return f"{numeric_value:,.0f}"


def describe_scheme(scheme_name: str) -> str:
    descriptions = {
        "Air Raid": "Pass-heavy offenses that stretch defenses horizontally and attack with volume through the air.",
        "West Coast": "Balanced attacks built around efficient short and intermediate throws, timing, and rhythm.",
        "Smashmouth/Triple Option": "Run-centered systems that lean on rushing volume, physicality, and ball control.",
    }
    return descriptions.get(scheme_name, "This label groups teams with similar offensive tendencies.")


def build_metrics_glossary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Term": "Breakout probability", "Meaning": "How likely a player is to take a meaningful step forward next season based on past production and usage."},
            {"Term": "Transfer probability", "Meaning": "How likely a player is to move from a non-power program into a power conference situation next."},
            {"Term": "Elite QB", "Meaning": "A model label that estimates whether a quarterback profile looks like the kind tied to high-end team results."},
            {"Term": "Offensive scheme", "Meaning": "A simplified style label for how a team tends to generate offense."},
            {"Term": "Usage", "Meaning": "How much of the offense runs through a player, using touches, targets, or pass attempts."},
            {"Term": "Explosive index", "Meaning": "A quick signal for how much yardage a player creates per opportunity."},
        ]
    )


def render_intro(metrics_df: pd.DataFrame, team_df: pd.DataFrame, player_df: pd.DataFrame) -> None:
    latest_season = int(player_df["Season"].max()) if not player_df.empty and "Season" in player_df.columns else "n/a"
    team_rows = len(team_df) if not team_df.empty else 0
    player_rows = len(player_df) if not player_df.empty else 0

    st.markdown(
        """
        <div class="hero">
            <h1>College Football Analysis Lab</h1>
            <p>This app is built to answer a simple question: what types of players, quarterbacks, and offensive styles tend to lead to better outcomes, and who looks most interesting next? The page is organized for someone brand new to the project, so you can start with the big picture and then drill into teams, players, and models only if you want more detail.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("Models Included", len(metrics_df))
    b.metric("Team Seasons Studied", f"{team_rows:,}")
    c.metric("Player Seasons Studied", f"{player_rows:,}")
    d.metric("Latest Season In View", latest_season)

    st.markdown('<div class="callout"><strong>What is the end goal?</strong> We are not trying to predict every box score. We are using historical college football data to spot patterns in winning offenses, quarterback play, player development, and possible future breakouts or transfers.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h4>1. Understand The Landscape</h4>
                <p>Start by seeing how teams score, how often they pass or run, and what offensive styles show up most often.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h4>2. Explore Player Signals</h4>
                <p>Then look at which player traits and usage patterns connect to growth, breakout seasons, or transfer movement.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h4>3. Use The Models Carefully</h4>
                <p>The models are best used as decision support. They surface useful clues, not guaranteed outcomes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_start_here(metrics_df: pd.DataFrame, summary_text: str, team_df: pd.DataFrame, candidate_df: pd.DataFrame) -> None:
    st.subheader("Start Here")
    st.write(
        "If you only visit one section, use this one. It explains what the project is doing, what the major terms mean, and gives you a quick sense of what the data says right now."
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("##### What The Project Is Measuring")
        st.dataframe(build_metrics_glossary(), use_container_width=True, hide_index=True)
    with right:
        st.markdown("##### Auto-Generated Summary")
        st.markdown(summary_text or "No summary has been generated yet.")

    if team_df.empty:
        return

    chart_df = coerce_numeric(team_df, ["Off", "Pct", "pass_rate", "run_rate"])
    if "OffensiveScheme" in chart_df.columns:
        scheme_counts = (
            chart_df["OffensiveScheme"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("OffensiveScheme")
            .reset_index(name="Teams")
        )
        fig = px.treemap(
            scheme_counts,
            path=["OffensiveScheme"],
            values="Teams",
            color="Teams",
            color_continuous_scale="YlGnBu",
            title="How The Dataset Groups Offensive Styles",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        scheme_choice = st.selectbox(
            "Pick an offensive style to understand it",
            scheme_counts["OffensiveScheme"].tolist(),
            key="scheme_explainer",
        )
        st.info(describe_scheme(scheme_choice))

    if not candidate_df.empty and {"Player", "Team", "Role", "breakout_probability"}.issubset(candidate_df.columns):
        preview = candidate_df.sort_values("breakout_probability", ascending=False).head(10)[
            ["Player", "Team", "Role", "breakout_probability", "transfer_probability"]
        ]
        st.markdown("##### Early Names To Know")
        st.dataframe(preview, use_container_width=True, hide_index=True)

    st.markdown("##### Core Output Files")
    dataset_rows = [{"Artifact": label, "Path": str(path), "Exists": path.exists()} for label, path in CORE_DATASETS.items()]
    st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True, hide_index=True)


def render_team_lab(team_df: pd.DataFrame) -> None:
    st.subheader("Team Offense Explorer")
    st.write(
        "Use this section to compare entire teams. Hover over points for details, zoom into a cluster, and use the filters to narrow down a conference, season, or offensive style."
    )
    if team_df.empty:
        st.info("No team dataset is available yet.")
        return

    team_df = coerce_numeric(team_df, ["Season", "Off", "Pct", "pass_rate", "run_rate", "elite_score", "team_total_yds", "team_total_td"])

    col1, col2, col3 = st.columns(3)
    seasons = sorted(team_df["Season"].dropna().astype(int).unique().tolist()) if "Season" in team_df.columns else []
    selected_seasons = col1.multiselect("Season", seasons, default=seasons[-3:] if len(seasons) >= 3 else seasons)
    conferences = sorted(team_df["Conf"].dropna().astype(str).unique().tolist()) if "Conf" in team_df.columns else []
    selected_conf = col2.selectbox("Conference", ["All"] + conferences)
    schemes = sorted(team_df["OffensiveScheme"].dropna().astype(str).unique().tolist()) if "OffensiveScheme" in team_df.columns else []
    selected_scheme = col3.selectbox("Offensive Style", ["All"] + schemes)

    filtered_df = team_df.copy()
    if selected_seasons:
        filtered_df = filtered_df[filtered_df["Season"].isin(selected_seasons)]
    if selected_conf != "All":
        filtered_df = filtered_df[filtered_df["Conf"].astype(str) == selected_conf]
    if selected_scheme != "All":
        filtered_df = filtered_df[filtered_df["OffensiveScheme"].astype(str) == selected_scheme]

    if filtered_df.empty:
        st.warning("The current filters returned no teams.")
        return

    fig = px.scatter(
        filtered_df,
        x="pass_rate",
        y="Off",
        size="team_total_yds",
        color="OffensiveScheme" if "OffensiveScheme" in filtered_df.columns else None,
        hover_name="Team",
        hover_data=["Season", "Conf", "Pct", "elite_score", "team_total_td"],
        title="How Passing Tendencies Connect To Offensive Output",
        labels={
            "pass_rate": "Pass Rate",
            "Off": "Points Per Game",
            "team_total_yds": "Total Yards",
        },
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.82))
    fig.update_layout(legend_title_text="Offensive Style")
    st.plotly_chart(fig, use_container_width=True)

    comparison = filtered_df.sort_values(["Off", "Pct"], ascending=False).head(25)[
        [column for column in ["Season", "Team", "Conf", "OffensiveScheme", "Off", "Pct", "pass_rate", "run_rate", "elite_score"] if column in filtered_df.columns]
    ]
    st.markdown("##### Best Matching Teams In The Current View")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if {"OffensiveScheme", "Pct"}.issubset(filtered_df.columns):
        summary = (
            filtered_df.groupby("OffensiveScheme", dropna=False)
            .agg(teams=("Team", "count"), avg_win_pct=("Pct", "mean"), avg_points=("Off", "mean"))
            .reset_index()
            .sort_values("avg_win_pct", ascending=False)
        )
        scheme_fig = px.bar(
            summary,
            x="OffensiveScheme",
            y="avg_win_pct",
            color="avg_points",
            text="teams",
            color_continuous_scale="Sunset",
            title="Which Offensive Styles Tend To Win More In This Slice?",
            labels={"avg_win_pct": "Average Win Percentage", "avg_points": "Average Points Per Game"},
        )
        scheme_fig.update_traces(textposition="outside")
        scheme_fig.update_layout(xaxis_title="", yaxis_tickformat=".0%")
        st.plotly_chart(scheme_fig, use_container_width=True)


def render_player_lab(player_df: pd.DataFrame) -> None:
    st.subheader("Player Growth Explorer")
    st.write(
        "This section focuses on individuals. The goal is to see how usage, explosiveness, and role relate to next-season growth."
    )
    if player_df.empty:
        st.info("No player trajectory dataset is available yet.")
        return

    player_df = coerce_numeric(
        player_df,
        [
            "Season",
            "usage",
            "explosive_index",
            "total_yds",
            "total_td",
            "breakout_probability",
            "transfer_probability",
            "usage_growth",
            "yds_growth",
            "td_growth",
            "team_win_pct",
        ],
    )

    col1, col2, col3 = st.columns(3)
    roles = sorted(player_df["Role"].dropna().astype(str).unique().tolist()) if "Role" in player_df.columns else []
    selected_roles = col1.multiselect("Role", roles, default=roles[:3] if len(roles) >= 3 else roles)
    selected_min_usage = col2.slider("Minimum usage", min_value=0.0, max_value=float(player_df["usage"].fillna(0).max()), value=0.0)
    selected_metric = col3.selectbox(
        "Color points by",
        [column for column in ["breakout_probability", "transfer_probability", "team_win_pct", "yds_growth"] if column in player_df.columns],
    )

    filtered_df = player_df.copy()
    if selected_roles:
        filtered_df = filtered_df[filtered_df["Role"].astype(str).isin(selected_roles)]
    filtered_df = filtered_df[filtered_df["usage"].fillna(0) >= selected_min_usage]

    if filtered_df.empty:
        st.warning("The current player filters returned no rows.")
        return

    fig = px.scatter(
        filtered_df,
        x="usage",
        y="explosive_index",
        size="total_yds",
        color=selected_metric,
        hover_name="Player",
        hover_data=["Team", "Season", "Role", "total_td", "yds_growth", "td_growth"],
        color_continuous_scale="Viridis",
        title="How Usage And Explosiveness Relate To Future Signals",
        labels={"usage": "Current Usage", "explosive_index": "Explosive Index", "total_yds": "Total Yards"},
    )
    fig.update_traces(marker=dict(opacity=0.80, line=dict(width=0.8, color="white")))
    st.plotly_chart(fig, use_container_width=True)

    if {"Role", "breakout_probability"}.issubset(filtered_df.columns):
        distribution_fig = px.box(
            filtered_df,
            x="Role",
            y="breakout_probability",
            color="Role",
            points="all",
            title="Breakout Scores By Position Group",
            labels={"breakout_probability": "Breakout Probability", "Role": ""},
        )
        distribution_fig.update_layout(showlegend=False)
        st.plotly_chart(distribution_fig, use_container_width=True)

    st.markdown("##### Players Standing Out In The Current View")
    leaders = filtered_df.sort_values("breakout_probability", ascending=False).head(50)[
        [column for column in ["Player", "Team", "Role", "usage", "explosive_index", "breakout_probability", "transfer_probability"] if column in filtered_df.columns]
    ]
    st.dataframe(leaders, use_container_width=True, hide_index=True)


def render_candidate_lab(candidates_df: pd.DataFrame) -> None:
    st.subheader("2025 Candidate Board")
    st.write(
        "Think of this as a scouting board created from the model outputs. Use the controls to focus on a role, a team, or only the players clearing a probability threshold."
    )
    if candidates_df.empty:
        st.info("No candidate table is available yet.")
        return

    candidates_df = coerce_numeric(
        candidates_df,
        ["Season", "usage", "explosive_index", "total_yds", "total_td", "breakout_probability", "transfer_probability", "team_win_pct"],
    )

    col1, col2, col3, col4 = st.columns(4)
    role_options = sorted(candidates_df["Role"].dropna().astype(str).unique().tolist()) if "Role" in candidates_df.columns else []
    selected_role = col1.selectbox("Role", ["All"] + role_options)
    team_options = sorted(candidates_df["Team"].dropna().astype(str).unique().tolist()) if "Team" in candidates_df.columns else []
    selected_team = col2.selectbox("Team", ["All"] + team_options)
    min_breakout = col3.slider("Minimum breakout probability", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
    min_transfer = col4.slider("Minimum transfer probability", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    filtered_df = candidates_df.copy()
    if selected_role != "All":
        filtered_df = filtered_df[filtered_df["Role"].astype(str) == selected_role]
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["Team"].astype(str) == selected_team]
    filtered_df = filtered_df[
        (filtered_df["breakout_probability"].fillna(0) >= min_breakout)
        & (filtered_df["transfer_probability"].fillna(0) >= min_transfer)
    ]

    if filtered_df.empty:
        st.warning("No candidates matched the current thresholds.")
        return

    top_n = st.slider("How many players to plot", min_value=10, max_value=min(100, len(filtered_df)), value=min(30, len(filtered_df)))
    ranked_df = filtered_df.sort_values(["breakout_probability", "transfer_probability"], ascending=False).head(top_n)

    fig = px.scatter(
        ranked_df,
        x="breakout_probability",
        y="transfer_probability",
        size="usage",
        color="Role" if "Role" in ranked_df.columns else None,
        hover_name="Player",
        hover_data=["Team", "Conf", "total_yds", "total_td", "explosive_index"],
        title="Candidate Board: Breakout Risk vs Transfer Risk",
        labels={
            "breakout_probability": "Breakout Probability",
            "transfer_probability": "Transfer Probability",
            "usage": "Usage",
        },
    )
    fig.update_traces(marker=dict(opacity=0.84, line=dict(width=1, color="white")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Current Candidate Table")
    st.dataframe(
        ranked_df[
            [column for column in ["Player", "Team", "Conf", "Role", "usage", "explosive_index", "breakout_probability", "transfer_probability"] if column in ranked_df.columns]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_model_lab(metrics_df: pd.DataFrame) -> None:
    st.subheader("Model Lab")
    st.write(
        "This is the more technical section. It is here for users who want to inspect how each model performed and which inputs mattered most."
    )
    if metrics_df.empty:
        st.info("Run the analysis to populate model metrics.")
        return

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    selected_model = st.selectbox("Choose a model to inspect", list(MODEL_FILES))
    importance_df, predictions_df, report_text = load_model_artifacts(selected_model)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### What Drove The Model")
        if importance_df.empty:
            st.info("No feature importance file was found for this model.")
        else:
            importance_df = coerce_numeric(importance_df, ["importance"])
            top_n = st.slider("Top factors", min_value=5, max_value=min(25, len(importance_df)), value=min(12, len(importance_df)), key="model_top_n")
            chart_df = importance_df.head(top_n).sort_values("importance")
            fig = px.bar(
                chart_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Cividis",
                title="Most Influential Inputs",
                labels={"importance": "Importance", "feature": ""},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(chart_df.sort_values("importance", ascending=False), use_container_width=True, hide_index=True)

    with right:
        st.markdown("##### Plain-Language Reading")
        if report_text:
            st.code(report_text)
        elif not predictions_df.empty:
            st.dataframe(predictions_df.head(25), use_container_width=True, hide_index=True)
        else:
            st.info("No report or sample predictions are available for this model.")


def render_raw_data(team_df: pd.DataFrame, offense_df: pd.DataFrame, player_df: pd.DataFrame, candidates_df: pd.DataFrame) -> None:
    st.subheader("Raw Data Explorer")
    st.write("For power users, this section exposes the source tables behind the app. Use it to search, filter, and inspect the raw rows.")

    dataset_name = st.radio(
        "Choose a dataset",
        ["Team", "Offense", "Player Trajectory", "Candidates"],
        horizontal=True,
    )
    dataset_map = {
        "Team": team_df,
        "Offense": offense_df,
        "Player Trajectory": player_df,
        "Candidates": candidates_df,
    }
    selected_df = dataset_map[dataset_name]
    if selected_df.empty:
        st.info(f"No {dataset_name.lower()} dataset is available yet.")
        return

    query = st.text_input("Search within the visible dataset")
    working_df = selected_df.copy()
    if query:
        mask = pd.Series(False, index=working_df.index)
        lowered = query.lower()
        for column in working_df.columns:
            mask = mask | working_df[column].astype(str).str.lower().str.contains(lowered, na=False)
        working_df = working_df[mask]

    st.caption(f"{len(working_df):,} rows shown before the 500-row display cap.")
    st.dataframe(working_df.head(500), use_container_width=True, hide_index=True)


render_intro(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

with st.sidebar:
    st.header("Analysis Controls")
    st.write("Refresh the data products whenever you want to rerun the pipeline.")
    st.caption(f"Last output update: {latest_output_timestamp()}")
    run_clicked = st.button("Run / Refresh Analysis", use_container_width=True, type="primary")

    if run_clicked:
        with st.spinner("Running the pipeline and rebuilding outputs..."):
            run_analysis()
        clear_caches()
        st.success("Analysis complete. Outputs refreshed.")
        st.rerun()

    st.divider()
    st.markdown("##### How To Use This App")
    st.caption("Start in `Start Here`, then move into teams, players, and candidates. Use `Model Lab` only when you want the technical details.")
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

render_intro(metrics_df, team_df, player_df)

start_tab, teams_tab, players_tab, candidates_tab, models_tab, data_tab = st.tabs(
    ["Start Here", "Teams", "Players", "Candidates", "Model Lab", "Raw Data"]
)

with start_tab:
    render_start_here(metrics_df, summary_text, team_df, candidates_df)

with teams_tab:
    render_team_lab(team_df)

with players_tab:
    render_player_lab(player_df)

with candidates_tab:
    render_candidate_lab(candidates_df)

with models_tab:
    render_model_lab(metrics_df)

with data_tab:
    render_raw_data(team_df, offense_df, player_df, candidates_df)
