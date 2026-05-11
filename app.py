import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import math

st.set_page_config(
    page_title="Sports League Standings Manager",
    page_icon="🏆",
    layout="wide",
)

# -------------------------------
# --------- THEME CSS -----------
# -------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #101a2e;
        --panel-2: #0f172a;
        --accent: #22c55e;
        --accent-2: #3b82f6;
        --muted: #94a3b8;
        --text: #e5e7eb;
        --text-2: #cbd5e1;
        --chip: #1f2937;
        --warn: #f59e0b;
        --danger: #ef4444;
    }
    .main, .block-container { background: var(--bg) !important; }
    .stApp { background: var(--bg) !important; }
    .css-18ni7ap, .css-1dp5vir { background: var(--panel) !important; }
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, span { color: var(--text) !important; }
    .metric-card {
        background: linear-gradient(145deg, var(--panel), var(--panel-2));
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid #1f2a44;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .chip {
        display:inline-block; padding:6px 10px; border-radius:999px; background:var(--chip); color:var(--text-2);
        border:1px solid #2a3a5e; font-size:12px; margin-right:6px;
    }
    .title {
        font-size:28px; font-weight:800; letter-spacing:0.3px;
        background: linear-gradient(90deg, #fff, #93c5fd);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: var(--muted); font-size:14px; }
    .divider { height:1px; background: #1f2a44; margin:10px 0 20px 0; }
    .kpi-number { font-size:28px; font-weight:700; color: #fff; }
    .kpi-label { color: var(--muted); font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# ----- DATA & UTILITIES --------
# -------------------------------

REQUIRED_COLS = ["Home_Team", "Away_Team", "Home_Score", "Away_Score"]

def load_results(uploaded_file, default_path: Path):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if default_path.exists():
            df = pd.read_csv(default_path)
        else:
            st.warning("Upload a CSV or place your file next to the app and set the name in the sidebar.")
            return pd.DataFrame(columns=REQUIRED_COLS)
    return df

def validate_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return False
    for sc in ["Home_Score", "Away_Score"]:
        if not pd.api.types.is_numeric_dtype(df[sc]):
            try:
                df[sc] = pd.to_numeric(df[sc])
            except Exception as e:
                st.error(f"Column {sc} must be numeric. Error: {e}")
                return False
    # Normalize strings
    df["Home_Team"] = df["Home_Team"].astype(str).str.strip()
    df["Away_Team"] = df["Away_Team"].astype(str).str.strip()
    return True

def compute_standings(df: pd.DataFrame, win_points=3, tie_points=1, loss_points=0):
    if df.empty:
        return pd.DataFrame(columns=[
            "Team","Games_Played","Wins","Losses","Ties",
            "Goals_For","Goals_Against","Goal_Difference","Points"
        ])

    def outcomes(row):
        hs, as_ = int(row["Home_Score"]), int(row["Away_Score"])
        if hs > as_:
            return pd.Series({"Home_W":1,"Home_L":0,"Home_T":0,"Away_W":0,"Away_L":1,"Away_T":0})
        elif hs < as_:
            return pd.Series({"Home_W":0,"Home_L":1,"Home_T":0,"Away_W":1,"Away_L":0,"Away_T":0})
        else:
            return pd.Series({"Home_W":0,"Home_L":0,"Home_T":1,"Away_W":0,"Away_L":0,"Away_T":1})

    tmp = df.copy()
    tmp = pd.concat([tmp, tmp.apply(outcomes, axis=1)], axis=1)

    home_stats = (
        tmp.groupby("Home_Team")
          .agg(Games_Played_home=("Home_Team","count"),
               Wins_home=("Home_W","sum"),
               Losses_home=("Home_L","sum"),
               Ties_home=("Home_T","sum"),
               Goals_For_home=("Home_Score","sum"),
               Goals_Against_home=("Away_Score","sum"))
          .reset_index().rename(columns={"Home_Team":"Team"})
    )
    away_stats = (
        tmp.groupby("Away_Team")
          .agg(Games_Played_away=("Away_Team","count"),
               Wins_away=("Away_W","sum"),
               Losses_away=("Away_L","sum"),
               Ties_away=("Away_T","sum"),
               Goals_For_away=("Away_Score","sum"),
               Goals_Against_away=("Home_Score","sum"))
          .reset_index().rename(columns={"Away_Team":"Team"})
    )

    standings = pd.merge(home_stats, away_stats, on="Team", how="outer").fillna(0)
    standings["Games_Played"] = standings["Games_Played_home"] + standings["Games_Played_away"]
    standings["Wins"] = standings["Wins_home"] + standings["Wins_away"]
    standings["Losses"] = standings["Losses_home"] + standings["Losses_away"]
    standings["Ties"] = standings["Ties_home"] + standings["Ties_away"]
    standings["Goals_For"] = standings["Goals_For_home"] + standings["Goals_For_away"]
    standings["Goals_Against"] = standings["Goals_Against_home"] + standings["Goals_Against_away"]
    standings["Goal_Difference"] = standings["Goals_For"] - standings["Goals_Against"]
    standings["Points"] = standings["Wins"] * win_points + standings["Ties"] * tie_points + standings["Losses"] * loss_points

    order = ["Team","Games_Played","Wins","Losses","Ties","Goals_For","Goals_Against","Goal_Difference","Points",
             "Wins_home","Wins_away","Goals_For_home","Goals_For_away"]
    standings = standings[order].sort_values(by=["Points","Goal_Difference","Goals_For"], ascending=[False, False, False]).reset_index(drop=True)
    return standings

def last_n_form(df: pd.DataFrame, team: str, n=5):
    if df.empty:
        return ""
    rel = df[(df["Home_Team"] == team) | (df["Away_Team"] == team)].copy()
    if rel.empty:
        return ""
    # sort by a best-guess "date" if exists; else by index
    if "Date" in rel.columns:
        rel = rel.sort_values("Date")
    # compute results
    def r(row):
        home = row["Home_Team"] == team
        gf = int(row["Home_Score"]) if home else int(row["Away_Score"])
        ga = int(row["Away_Score"]) if home else int(row["Home_Score"])
        if gf > ga: return "W"
        if gf < ga: return "L"
        return "D"
    seq = rel.apply(r, axis=1).tolist()[-n:]
    return "-".join(seq) if seq else ""

def head_to_head(df: pd.DataFrame, team: str):
    df = df[(df["Home_Team"] == team) | (df["Away_Team"] == team)].copy()
    if df.empty:
        return pd.DataFrame(columns=["Opponent","Games","Wins","Losses","Ties","Goals_For","Goals_Against","Points"])

    def per_row(r):
        home = r["Home_Team"] == team
        opp = r["Away_Team"] if home else r["Home_Team"]
        gf = int(r["Home_Score"]) if home else int(r["Away_Score"])
        ga = int(r["Away_Score"]) if home else int(r["Home_Score"])
        if gf > ga:
            w,l,t,pts = 1,0,0,3
        elif gf < ga:
            w,l,t,pts = 0,1,0,0
        else:
            w,l,t,pts = 0,0,1,1
        return pd.Series({"Opponent":opp,"Games":1,"Wins":w,"Losses":l,"Ties":t,"Goals_For":gf,"Goals_Against":ga,"Points":pts})
    perf = df.apply(per_row, axis=1)
    return (perf.groupby("Opponent")
                 .sum(numeric_only=True)
                 .reset_index()
                 .sort_values(by=["Points","Goals_For"], ascending=[False, False]))

def build_match_browser(df: pd.DataFrame, team_filter=None):
    if df.empty:
        return df
    tmp = df.copy()
    tmp["Outcome"] = np.where(tmp["Home_Score"] > tmp["Away_Score"], "Home Win",
                       np.where(tmp["Home_Score"] < tmp["Away_Score"], "Away Win", "Draw"))
    if team_filter and team_filter != "All Teams":
        tmp = tmp[(tmp["Home_Team"] == team_filter) | (tmp["Away_Team"] == team_filter)]
    return tmp

# --------- Prediction Utilities (Poisson model) ---------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def team_rates(df: pd.DataFrame):
    # League totals
    lg_home_avg = df["Home_Score"].mean() if len(df) else 1.2
    lg_away_avg = df["Away_Score"].mean() if len(df) else 1.0

    # per-team splits
    home = df.groupby("Home_Team").agg(
        GF_home=("Home_Score", "mean"),
        GA_home=("Away_Score", "mean"),
        GP_home=("Home_Score", "count"),
    )
    away = df.groupby("Away_Team").agg(
        GF_away=("Away_Score", "mean"),
        GA_away=("Home_Score", "mean"),
        GP_away=("Away_Score", "count"),
    )
    teams_all = sorted(set(df["Home_Team"]).union(set(df["Away_Team"])))
    stats = {}
    for t in teams_all:
        gf_h = home.loc[t, "GF_home"] if t in home.index else np.nan
        ga_h = home.loc[t, "GA_home"] if t in home.index else np.nan
        gp_h = home.loc[t, "GP_home"] if t in home.index else 0
        gf_a = away.loc[t, "GF_away"] if t in away.index else np.nan
        ga_a = away.loc[t, "GA_away"] if t in away.index else np.nan
        gp_a = away.loc[t, "GP_away"] if t in away.index else 0

        # Fallback to overall if missing split
        played = gp_h + gp_a if (gp_h + gp_a) > 0 else 1
        gf_all = ( (gf_h if not np.isnan(gf_h) else 0) * gp_h + (gf_a if not np.isnan(gf_a) else 0) * gp_a ) / max(1, played)
        ga_all = ( (ga_h if not np.isnan(ga_h) else 0) * gp_h + (ga_a if not np.isnan(ga_a) else 0) * gp_a ) / max(1, played)

        stats[t] = dict(
            GF_home=gf_h if not np.isnan(gf_h) else gf_all,
            GA_home=ga_h if not np.isnan(ga_h) else ga_all,
            GF_away=gf_a if not np.isnan(gf_a) else gf_all,
            GA_away=ga_a if not np.isnan(ga_a) else ga_all,
            GF_all=gf_all, GA_all=ga_all,
        )
    return stats, lg_home_avg, lg_away_avg

def expected_goals(stats, lg_home_avg, lg_away_avg, home_team, away_team, home_adv_scale=1.0):
    hs = stats[home_team]
    as_ = stats[away_team]

    # Attack/defense multipliers (simple)
    home_attack = hs["GF_home"] / max(0.01, lg_home_avg)
    away_defense = as_["GA_away"] / max(0.01, lg_home_avg)

    away_attack = as_["GF_away"] / max(0.01, lg_away_avg)
    home_defense = hs["GA_home"] / max(0.01, lg_away_avg)

    lam_home = lg_home_avg * home_attack * (1.0 / max(0.01, away_defense)) * home_adv_scale
    lam_away = lg_away_avg * away_attack * (1.0 / max(0.01, home_defense))

    # Guard rails
    lam_home = max(0.05, min(lam_home, 4.5))
    lam_away = max(0.05, min(lam_away, 4.5))
    return float(lam_home), float(lam_away)

def score_matrix(lh, la, max_goals=6):
    i = np.arange(0, max_goals+1)
    j = np.arange(0, max_goals+1)
    home_p = np.array([poisson_pmf(k, lh) for k in i])
    away_p = np.array([poisson_pmf(k, la) for k in j])
    mat = np.outer(home_p, away_p)  # (i x j)
    return mat

def outcome_probs(mat):
    # P(home win), P(draw), P(away win)
    hp = np.tril(mat, -1).sum()  # incorrect: tril is below diagonal (home<away) - fix below
    # Correct compute by iterating indices
    n = mat.shape[0]
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for h in range(n):
        for a in range(n):
            if h > a:
                p_home += mat[h, a]
            elif h == a:
                p_draw += mat[h, a]
            else:
                p_away += mat[h, a]
    return p_home, p_draw, p_away

# -------------------------------
# -------- SIDEBAR UI -----------
# -------------------------------

st.sidebar.markdown("<div class='title'>🏆 League Standings Manager</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtitle'>Upload your results CSV or use default filename.</div>", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Upload CSV (Home_Team, Away_Team, Home_Score, Away_Score)", type=["csv"])

default_name = st.sidebar.text_input("Default CSV filename (in same folder):", "lebanon_div1_full_double_round_robin.csv")

st.sidebar.markdown("---")
st.sidebar.subheader("Points System")
win_pts = st.sidebar.number_input("Win points", min_value=0, max_value=10, value=3, step=1)
tie_pts = st.sidebar.number_input("Tie points", min_value=0, max_value=10, value=1, step=1)
loss_pts = st.sidebar.number_input("Loss points", min_value=0, max_value=10, value=0, step=1)

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Top N chart", min_value=3, max_value=12, value=5, step=1)

# -------------------------------
# --------- DATA LOAD -----------
# -------------------------------
df = load_results(uploaded, Path(default_name))
if not df.empty and validate_df(df):
    teams = sorted(set(df["Home_Team"]).union(set(df["Away_Team"])))
else:
    teams = []

# -------------------------------
# --------- HEADER --------------
# -------------------------------
st.markdown("<div class='title'>Sports League Standings Manager</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Standings • Head-to-Head • Browser • Insights • <b>Predictions</b></div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# -------------------------------
# --------- KPI ROW -------------
# -------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Total Matches</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-number'>{len(df)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Teams</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-number'>{len(teams)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Win Points</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-number'>{win_pts}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Tie / Loss Points</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-number'>{tie_pts} / {loss_pts}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if df.empty or not validate_df(df):
    st.info("Upload a CSV to start exploring.")
    st.stop()

# -------------------------------
# --------- COMPUTE -------------
# -------------------------------
standings = compute_standings(df, win_points=win_pts, tie_points=tie_pts, loss_points=loss_pts)

# -------------------------------
# --------- TABS ----------------
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏁 Overview", "📊 Standings", "🤝 Head-to-Head", "🗂 Match Browser", "🧠 Insights", "🔮 Prediction"])

with tab1:
    c1, c2 = st.columns([1,1])
    with c1:
        # Top N bar chart
        top_df = standings.nlargest(top_n, "Points")[["Team","Points","Goal_Difference"]]
        fig = px.bar(top_df, x="Team", y="Points", hover_data=["Goal_Difference"], title=f"Top {top_n} Teams by Points")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Goal Difference vs Points scatter
        fig2 = px.scatter(standings, x="Goal_Difference", y="Points", color="Points", hover_name="Team",
                          size="Goals_For", title="Points vs Goal Difference (bubble size = Goals For)")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("### League Table")
    st.dataframe(standings, use_container_width=True, height=520)
    st.download_button("⬇️ Download Standings (CSV)", data=standings.to_csv(index=False), file_name="standings_out.csv", mime="text/csv")

with tab3:
    st.markdown("### Head-to-Head Explorer")
    team_sel = st.selectbox("Select a team", ["Select a team"] + teams)
    if team_sel != "Select a team":
        h2h = head_to_head(df, team_sel)
        # add form string
        form = last_n_form(df, team_sel, n=5)
        st.markdown(f"**Recent form (last 5):** {form if form else 'N/A'}")
        # table + bar
        c1, c2 = st.columns([1,1])
        with c1:
            st.dataframe(h2h, use_container_width=True, height=520)
        with c2:
            if not h2h.empty:
                fig3 = px.bar(h2h, x="Opponent", y="Points", title=f"Points vs Opponents — {team_sel}")
                fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown("### Match Browser")
    team_filter = st.selectbox("Filter by team", ["All Teams"] + teams, key="browser")
    browser_df = build_match_browser(df, None if team_filter == "All Teams" else team_filter)
    st.dataframe(browser_df, use_container_width=True, height=520)
    st.download_button("⬇️ Download Filtered Matches (CSV)",
                       data=browser_df.to_csv(index=False), file_name="matches_filtered.csv", mime="text/csv")

with tab5:
    st.markdown("### Insights & Extras")
    # Home vs Away comparison
    st.markdown("**Home vs Away Goals (Top 10 by Goals For)**")
    comp = standings[["Team","Goals_For_home","Goals_For_away"]].copy()
    comp = comp.sort_values("Goals_For_home", ascending=False).head(10)
    comp_long = comp.melt(id_vars="Team", value_vars=["Goals_For_home","Goals_For_away"],
                          var_name="Venue", value_name="Goals")
    fig4 = px.bar(comp_long, x="Team", y="Goals", color="Venue", barmode="group", title="Home vs Away Goals")
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

    # Simple balance gauge: Points vs GD
    st.markdown("**Consistency Gauge**")
    if not standings.empty:
        # Normalize to 0-100 for a simple gauge feel
        pts_norm = 100 * (standings["Points"] - standings["Points"].min()) / max(1, (standings["Points"].max() - standings["Points"].min()))
        gd_norm = 100 * (standings["Goal_Difference"] - standings["Goal_Difference"].min()) / max(1, (standings["Goal_Difference"].max() - standings["Goal_Difference"].min()))
        consistency = (pts_norm + gd_norm) / 2
        gauge = pd.DataFrame({"Team": standings["Team"], "Consistency": consistency.round(1)}).sort_values("Consistency", ascending=False)
        fig5 = px.bar(gauge.head(10), x="Team", y="Consistency", title="Top 10 Consistent Teams (scaled 0–100)")
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)

with tab6:
    st.markdown("### Match Prediction (Poisson Model)")
    if not teams:
        st.info("Load a CSV to enable predictions.")
    else:
        colA, colB = st.columns(2)
        with colA:
            home_team = st.selectbox("Home Team", teams, index=0, key="home_team_sel")
        with colB:
            away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0, key="away_team_sel")

        if home_team == away_team:
            st.warning("Select two different teams.")
        else:
            max_goals = st.slider("Max goals per side (for probability table)", 3, 10, 6, 1)
            home_adv = st.slider("Home advantage scale", 0.6, 1.6, 1.1, 0.05)

            # Build rates
            stats, lg_home_avg, lg_away_avg = team_rates(df)
            lam_home, lam_away = expected_goals(stats, lg_home_avg, lg_away_avg, home_team, away_team, home_adv_scale=home_adv)

            # Score matrix & outcome probabilities
            mat = score_matrix(lam_home, lam_away, max_goals=max_goals)
            p_home, p_draw, p_away = outcome_probs(mat)
            # Most likely scoreline
            idx = np.unravel_index(np.argmax(mat), mat.shape)
            likely_score = (int(idx[0]), int(idx[1]))
            likely_prob = float(mat[idx])

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Expected Goals — Home</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-number'>{lam_home:.2f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Expected Goals — Away</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-number'>{lam_away:.2f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Win / Draw / Win</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-number'>{p_home*100:.1f}% / {p_draw*100:.1f}% / {p_away*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-label'>Most Likely Score</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-number'>{likely_score[0]}–{likely_score[1]}  ({likely_prob*100:.1f}%)</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Heatmap of scoreline probabilities
            heat_df = pd.DataFrame(mat, index=[str(i) for i in range(max_goals+1)], columns=[str(j) for j in range(max_goals+1)])
            figh = px.imshow(heat_df, text_auto=True, aspect="auto",
                             labels=dict(x=f"{away_team} goals", y=f"{home_team} goals", color="Probability"),
                             title="Scoreline Probability Heatmap")
            figh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figh, use_container_width=True)

            # Top 5 most likely scorelines
            flat = []
            for h in range(mat.shape[0]):
                for a in range(mat.shape[1]):
                    flat.append(((h, a), mat[h, a]))
            top_scores = sorted(flat, key=lambda x: x[1], reverse=True)[:5]
            top_df = pd.DataFrame([{"Scoreline": f"{home_team} {h}–{a} {away_team}", "Probability": f"{p*100:.2f}%"} for (h, a), p in top_scores])
            st.markdown("**Top 5 most likely scorelines**")
            st.table(top_df)

            st.caption("Model: Poisson goals with team attack/defense vs league averages and adjustable home advantage. This is a simple statistical model — not betting advice.")

st.markdown("<br><div class='subtitle'>Built with ❤️ using Streamlit + Plotly. Customize points, upload new CSVs, predict matches, and export results.</div>", unsafe_allow_html=True)
