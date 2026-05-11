#!/usr/bin/env python3
"""
Sports League Standings Manager
--------------------------------
Usage Examples:
  python standings_manager.py --csv results.csv --show
  python standings_manager.py --csv results.csv --team "Barcelona"
  python standings_manager.py --csv results.csv --export standings_out.csv --plot top5.png

CSV Columns (required, case-sensitive):
  Home_Team, Away_Team, Home_Score, Away_Score

Notes:
- Points system is Win=3, Tie=1, Loss=0 by default (change with flags).
- Outputs a sorted standings table (Points desc, then Goal_Difference).
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = ["Home_Team", "Away_Team", "Home_Score", "Away_Score"]


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    # Basic type/validity checks
    for sc in ["Home_Score", "Away_Score"]:
        if not pd.api.types.is_numeric_dtype(df[sc]):
            try:
                df[sc] = pd.to_numeric(df[sc])
            except Exception as e:
                raise ValueError(f"Column {sc} must be numeric. Error: {e}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace and unify team strings
    df["Home_Team"] = df["Home_Team"].astype(str).str.strip()
    df["Away_Team"] = df["Away_Team"].astype(str).str.strip()
    return df


def compute_match_points(home_score: int, away_score: int) -> Tuple[int, int]:
    """Return (home_points, away_points) with Win=3, Tie=1, Loss=0."""
    if home_score > away_score:
        return 3, 0
    if home_score < away_score:
        return 0, 3
    return 1, 1


def build_standings(
    df: pd.DataFrame,
    win_points: int = 3,
    tie_points: int = 1,
    loss_points: int = 0,
) -> pd.DataFrame:
    """
    Build a standings table with:
    Team, Games_Played, Wins, Losses, Ties, Goals_For, Goals_Against, Goal_Difference, Points
    """
    # Compute per-row outcomes for home and away
    def outcome_cols(row):
        hs, as_ = int(row["Home_Score"]), int(row["Away_Score"])
        # Points (default 3/1/0) but we compute as W/L/T first then map to points
        if hs > as_:
            h_w, h_l, h_t = 1, 0, 0
            a_w, a_l, a_t = 0, 1, 0
        elif hs < as_:
            h_w, h_l, h_t = 0, 1, 0
            a_w, a_l, a_t = 1, 0, 0
        else:
            h_w, h_l, h_t = 0, 0, 1
            a_w, a_l, a_t = 0, 0, 1

        return pd.Series({
            "Home_W": h_w, "Home_L": h_l, "Home_T": h_t,
            "Away_W": a_w, "Away_L": a_l, "Away_T": a_t,
        })

    df = df.copy()
    df = normalize(df)
    validate_input(df)
    outcomes = df.apply(outcome_cols, axis=1)
    df = pd.concat([df, outcomes], axis=1)

    # Home team contributions
    home_stats = (
        df.groupby("Home_Team")
          .agg(
              Games_Played_home=("Home_Team", "count"),
              Wins_home=("Home_W", "sum"),
              Losses_home=("Home_L", "sum"),
              Ties_home=("Home_T", "sum"),
              Goals_For_home=("Home_Score", "sum"),
              Goals_Against_home=("Away_Score", "sum"),
          )
          .reset_index()
          .rename(columns={"Home_Team": "Team"})
    )

    # Away team contributions
    away_stats = (
        df.groupby("Away_Team")
          .agg(
              Games_Played_away=("Away_Team", "count"),
              Wins_away=("Away_W", "sum"),
              Losses_away=("Away_L", "sum"),
              Ties_away=("Away_T", "sum"),
              Goals_For_away=("Away_Score", "sum"),
              Goals_Against_away=("Home_Score", "sum"),
          )
          .reset_index()
          .rename(columns={"Away_Team": "Team"})
    )

    # Merge and fill missing teams
    standings = pd.merge(home_stats, away_stats, on="Team", how="outer").fillna(0)

    # Sum home + away
    standings["Games_Played"] = standings["Games_Played_home"] + standings["Games_Played_away"]
    standings["Wins"] = standings["Wins_home"] + standings["Wins_away"]
    standings["Losses"] = standings["Losses_home"] + standings["Losses_away"]
    standings["Ties"] = standings["Ties_home"] + standings["Ties_away"]
    standings["Goals_For"] = standings["Goals_For_home"] + standings["Goals_For_away"]
    standings["Goals_Against"] = standings["Goals_Against_home"] + standings["Goals_Against_away"]
    standings["Goal_Difference"] = standings["Goals_For"] - standings["Goals_Against"]

    # Points mapping
    standings["Points"] = standings["Wins"] * win_points + standings["Ties"] * tie_points + standings["Losses"] * loss_points

    # Keep only required columns in nice order
    standings = standings[[
        "Team", "Games_Played", "Wins", "Losses", "Ties",
        "Goals_For", "Goals_Against", "Goal_Difference", "Points"
    ]].sort_values(by=["Points", "Goal_Difference", "Goals_For"], ascending=[False, False, False]).reset_index(drop=True)

    return standings


def head_to_head(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    For a selected team, return record vs every other team:
    Opponent, Games, Wins, Losses, Ties, Goals_For, Goals_Against, Points
    """
    df = df.copy()
    df = normalize(df)
    validate_input(df)
    team = team.strip()

    # Select rows where team participated
    rel = df[(df["Home_Team"] == team) | (df["Away_Team"] == team)]
    if rel.empty:
        return pd.DataFrame(columns=["Opponent", "Games", "Wins", "Losses", "Ties", "Goals_For", "Goals_Against", "Points"])

    def per_row(r):
        home = r["Home_Team"] == team
        opp = r["Away_Team"] if home else r["Home_Team"]
        gf = int(r["Home_Score"]) if home else int(r["Away_Score"])
        ga = int(r["Away_Score"]) if home else int(r["Home_Score"])
        if gf > ga:
            w, l, t, pts = 1, 0, 0, 3
        elif gf < ga:
            w, l, t, pts = 0, 1, 0, 0
        else:
            w, l, t, pts = 0, 0, 1, 1
        return pd.Series({"Opponent": opp, "Games": 1, "Wins": w, "Losses": l, "Ties": t, "Goals_For": gf, "Goals_Against": ga, "Points": pts})

    perf = rel.apply(per_row, axis=1)
    summary = (perf.groupby("Opponent")
                    .sum(numeric_only=True)
                    .reset_index()
                    .sort_values(by=["Points", "Goals_For"], ascending=[False, False]))
    return summary


def plot_top_n(standings: pd.DataFrame, top_n: int = 5, out_file: Path = None) -> None:
    top = standings.nlargest(top_n, "Points")
    plt.figure(figsize=(8, 5))
    plt.bar(top["Team"], top["Points"])
    plt.title(f"Top {top_n} Teams by Points")
    plt.xlabel("Team")
    plt.ylabel("Points")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        print(f"Saved bar chart to: {out_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Sports League Standings Manager")
    parser.add_argument("--csv", required=True, help="Path to results CSV (with columns Home_Team, Away_Team, Home_Score, Away_Score)")
    parser.add_argument("--show", action="store_true", help="Print standings to console")
    parser.add_argument("--export", help="Export standings to CSV")
    parser.add_argument("--plot", help="Save a bar chart of top N teams to this file (e.g., top5.png)")
    parser.add_argument("--top", type=int, default=5, help="How many top teams to visualize (default 5)")
    parser.add_argument("--team", help="Show head-to-head summary for this team")
    parser.add_argument("--win_points", type=int, default=3, help="Points for a win (default 3)")
    parser.add_argument("--tie_points", type=int, default=1, help="Points for a tie (default 1)")
    parser.add_argument("--loss_points", type=int, default=0, help="Points for a loss (default 0)")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    standings = build_standings(df, win_points=args.win_points, tie_points=args.tie_points, loss_points=args.loss_points)

    if args.show or (not args.export and not args.team and not args.plot):
        # Default: print standings if no other action specified
        print(standings.to_string(index=False))

    if args.export:
        out = Path(args.export)
        standings.to_csv(out, index=False)
        print(f"Standings exported to: {out}")

    if args.team:
        h2h = head_to_head(df, args.team)
        if h2h.empty:
            print(f'No matches found for team "{args.team}".')
        else:
            print(f'\nHead-to-Head for "{args.team}":')
            print(h2h.to_string(index=False))

    if args.plot:
        plot_top_n(standings, top_n=args.top, out_file=Path(args.plot))


if __name__ == "__main__":
    main()
