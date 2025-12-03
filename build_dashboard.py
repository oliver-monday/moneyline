#!/usr/bin/env python3
"""
Generates a lightly-styled HTML dashboard summarizing today's NBA matchups.
Uses nba_master.csv.

Features:
- Highlighted current-state rows (favorite/dog, home/away, home/away favorite/dog)
- Moneyline buckets
- ROI stats
- Opponent-strength stats (vs strong/weak opps)
- Recent form (Last 5 / Last 10 / Streak)
- League overview block at bottom
- Collapsible per-game sections with a summary line
"""

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

MASTER_PATH = Path("nba_master.csv")

ML_BUCKETS = [
    (-10000, -300, "Big favorite (≤ -300)"),
    (-299, -151, "Medium favorite (-299 to -151)"),
    (-150, -101, "Small favorite (-150 to -101)"),
    (-100, 100, "Coinflip (-100 to +100)"),
    (101, 200, "Small dog (+101 to +200)"),
    (201, 10000, "Big dog (≥ +201)"),
]

TEAM_TRICODES = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# ------------ helpers --------------------------------------------------


def pick_bucket(odds):
    if pd.isna(odds):
        return None
    for lo, hi, label in ML_BUCKETS:
        if lo <= odds <= hi:
            return label
    return None


def american_to_prob(odds):
    if pd.isna(odds):
        return None
    o = float(odds)
    if o < 0:
        return (-o) / ((-o) + 100)
    if o > 0:
        return 100 / (o + 100)
    return None


def fmt_odds(odds):
    if pd.isna(odds):
        return "—"
    return f"{int(round(odds)):+d}"


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.1f}%"


def maybe_hl_block(lines, highlight):
    """
    Given a list of already-built HTML lines, wrap them in <span class='hl'>
    if highlight is True, otherwise return as-is.
    """
    if not highlight:
        return lines
    return ["<span class='hl'>", *lines, "</span>"]


def label(text: str) -> str:
    """Label line (no value on same line for nested style)."""
    return text


def value(text: str, indent: int = 4) -> str:
    """Value line, indented."""
    return " " * indent + text


def load_master():
    df = pd.read_csv(MASTER_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    for col in [
        "home_ml",
        "away_ml",
        "home_score",
        "away_score",
        "home_spread",
        "away_spread",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def calc_return(row):
    """
    Unit ROI for a single game:
      - stake 1 unit on this team’s ML whenever we have odds + result
      - if win: profit based on American odds
      - if loss: -1 unit
    """
    if (not row["has_result"]) or (not row["has_ml"]):
        return 0.0
    odds = row["ml"]
    if pd.isna(odds):
        return 0.0
    if row["is_win"]:
        if odds > 0:
            return odds / 100.0
        else:
            return 100.0 / (-odds)
    else:
        return -1.0


# ------------ team results --------------------------------------------


def build_team_results(master):
    rows = []
    for _, r in master.iterrows():
        gid = r["game_id"]
        date = r["game_date"]

        rows.append(
            {
                "game_id": gid,
                "game_date": date,
                "team": r["home_team_name"],
                "opponent": r["away_team_name"],
                "is_home": True,
                "ml": r["home_ml"],
                "spread": r["home_spread"],
                "pf": r["home_score"],
                "pa": r["away_score"],
            }
        )
        rows.append(
            {
                "game_id": gid,
                "game_date": date,
                "team": r["away_team_name"],
                "opponent": r["home_team_name"],
                "is_home": False,
                "ml": r["away_ml"],
                "spread": r["away_spread"],
                "pf": r["away_score"],
                "pa": r["home_score"],
            }
        )

    df = pd.DataFrame(rows)

    df["has_result"] = (~df["pf"].isna()) & (~df["pa"].isna())
    df["is_win"] = df["has_result"] & (df["pf"] > df["pa"])
    df["has_ml"] = ~df["ml"].isna()
    df["is_fav"] = df["has_ml"] & (df["ml"] < 0)
    df["is_dog"] = df["has_ml"] & (df["ml"] > 0)

    # State flags
    df["is_home_fav"] = df["is_home"] & df["is_fav"]
    df["is_home_dog"] = df["is_home"] & df["is_dog"]
    df["is_away_fav"] = (~df["is_home"]) & df["is_fav"]
    df["is_away_dog"] = (~df["is_home"]) & df["is_dog"]

    df["ml_bucket"] = df["ml"].apply(pick_bucket)

    # ROI per row
    df["roi"] = df.apply(calc_return, axis=1)

    return df


def summarize_team(df):
    played = df[df["has_result"] & df["has_ml"]]
    total = len(played)
    wins = int(played["is_win"].sum())
    losses = total - wins

    def pct(w, l):
        return w / (w + l) if w + l > 0 else np.nan

    fav = played[played["is_fav"]]
    dog = played[played["is_dog"]]
    home = played[played["is_home"]]
    away = played[~played["is_home"]]

    # Derived granular states
    home_fav = played[played["is_home_fav"]]
    home_dog = played[played["is_home_dog"]]
    away_fav = played[played["is_away_fav"]]
    away_dog = played[played["is_away_dog"]]

    # ROI aggregates
    total_roi = played["roi"].sum()
    roi_pct = total_roi / total if total > 0 else np.nan

    fav_roi = fav["roi"].sum()
    fav_roi_pct = fav_roi / len(fav) if len(fav) > 0 else np.nan

    dog_roi = dog["roi"].sum()
    dog_roi_pct = dog_roi / len(dog) if len(dog) > 0 else np.nan

    return {
        "ml_record": f"{wins}-{losses}",
        "ml_win_pct": pct(wins, losses),
        "fav_record": f"{int(fav['is_win'].sum())}-{len(fav) - fav['is_win'].sum()}",
        "fav_win_pct": pct(fav["is_win"].sum(), len(fav) - fav["is_win"].sum()),
        "dog_record": f"{int(dog['is_win'].sum())}-{len(dog) - dog['is_win'].sum()}",
        "dog_win_pct": pct(dog["is_win"].sum(), len(dog) - dog["is_win"].sum()),
        "home_record": f"{int(home['is_win'].sum())}-{len(home) - home['is_win'].sum()}",
        "home_win_pct": pct(home["is_win"].sum(), len(home) - home["is_win"].sum()),
        "away_record": f"{int(away['is_win'].sum())}-{len(away) - away['is_win'].sum()}",
        "away_win_pct": pct(away["is_win"].sum(), len(away) - away["is_win"].sum()),
        "home_fav_record": f"{int(home_fav['is_win'].sum())}-{len(home_fav) - home_fav['is_win'].sum()}",
        "home_fav_win_pct": pct(
            home_fav["is_win"].sum(), len(home_fav) - home_fav["is_win"].sum()
        ),
        "home_dog_record": f"{int(home_dog['is_win'].sum())}-{len(home_dog) - home_dog['is_win'].sum()}",
        "home_dog_win_pct": pct(
            home_dog["is_win"].sum(), len(home_dog) - home_dog["is_win"].sum()
        ),
        "away_fav_record": f"{int(away_fav['is_win'].sum())}-{len(away_fav) - away_fav['is_win'].sum()}",
        "away_fav_win_pct": pct(
            away_fav["is_win"].sum(), len(away_fav) - away_fav["is_win"].sum()
        ),
        "away_dog_record": f"{int(away_dog['is_win'].sum())}-{len(away_dog) - away_dog['is_win'].sum()}",
        "away_dog_win_pct": pct(
            away_dog["is_win"].sum(), len(away_dog) - away_dog["is_win"].sum()
        ),
        "roi_units": total_roi,
        "roi_pct": roi_pct,
        "fav_roi_units": fav_roi,
        "fav_roi_pct": fav_roi_pct,
        "dog_roi_units": dog_roi,
        "dog_roi_pct": dog_roi_pct,
    }


def summarize_bucket(df, bucket):
    b = df[(df["ml_bucket"] == bucket) & df["has_result"] & df["has_ml"]]
    if b.empty:
        return {"bucket_record": "—", "bucket_win_pct": np.nan, "n": 0}

    wins = int(b["is_win"].sum())
    losses = len(b) - wins
    return {
        "bucket_record": f"{wins}-{losses}",
        "bucket_win_pct": wins / (wins + losses),
        "n": len(b),
    }


# ------------ opponent-adjusted & form ---------------------------------


def league_overview(team_results: pd.DataFrame) -> pd.DataFrame:
    """Per-team league table for use in overview + opponent strength."""
    hist = team_results[team_results["has_result"] & team_results["has_ml"]]

    rows = []
    for team, df in hist.groupby("team"):
        w_ = df["is_win"].sum()
        l_ = len(df) - w_
        ml_pct = w_ / (w_ + l_) if (w_ + l_) > 0 else np.nan

        fav = df[df["is_fav"]]
        w_f = fav["is_win"].sum()
        home = df[df["is_home"]]
        w_h = home["is_win"].sum()
        away = df[~df["is_home"]]
        w_a = away["is_win"].sum()
        dog = df[df["is_dog"]]
        w_d = dog["is_win"].sum()

        rows.append(
            {
                "team": team,
                "ml_pct": ml_pct,
                "home_pct": w_h / len(home) if len(home) > 0 else np.nan,
                "away_pct": w_a / len(away) if len(away) > 0 else np.nan,
                "fav_pct": w_f / len(fav) if len(fav) > 0 else np.nan,
                "dog_pct": w_d / len(dog) if len(dog) > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def opponent_adjusted_stats(team_df: pd.DataFrame, league_tbl: pd.DataFrame):
    """
    Record vs strong and weak opponents based on opponents' ML win%.
    Robust to missing columns / small samples.
    """
    if league_tbl.empty or "ml_pct" not in league_tbl.columns:
        return {
            "vs_strong_record": "—",
            "vs_strong_pct": np.nan,
            "vs_strong_n": 0,
            "vs_weak_record": "—",
            "vs_weak_pct": np.nan,
            "vs_weak_n": 0,
        }

    played = team_df[team_df["has_result"] & team_df["has_ml"]]
    if played.empty:
        return {
            "vs_strong_record": "—",
            "vs_strong_pct": np.nan,
            "vs_strong_n": 0,
            "vs_weak_record": "—",
            "vs_weak_pct": np.nan,
            "vs_weak_n": 0,
        }

    merged = played.merge(
        league_tbl[["team", "ml_pct"]],
        left_on="opponent",
        right_on="team",
        how="left",
        suffixes=("", "_opp"),
    )

    # Handle both cases: with or without suffix
    if "ml_pct_opp" in merged.columns:
        col = "ml_pct_opp"
    else:
        col = "ml_pct"

    merged = merged.dropna(subset=[col])
    if merged.empty:
        return {
            "vs_strong_record": "—",
            "vs_strong_pct": np.nan,
            "vs_strong_n": 0,
            "vs_weak_record": "—",
            "vs_weak_pct": np.nan,
            "vs_weak_n": 0,
        }

    base = league_tbl["ml_pct"].dropna()
    if base.empty:
        return {
            "vs_strong_record": "—",
            "vs_strong_pct": np.nan,
            "vs_strong_n": 0,
            "vs_weak_record": "—",
            "vs_weak_pct": np.nan,
            "vs_weak_n": 0,
        }

    top_cut = base.quantile(0.80)
    bot_cut = base.quantile(0.20)

    strong = merged[merged[col] >= top_cut]
    weak = merged[merged[col] <= bot_cut]

    def rec(df_):
        if df_.empty:
            return "—", np.nan, 0
        w_ = df_["is_win"].sum()
        l_ = len(df_) - w_
        return f"{int(w_)}-{int(l_)}", w_ / (w_ + l_), len(df_)

    s_rec, s_pct, s_n = rec(strong)
    w_rec, w_pct, w_n = rec(weak)

    return {
        "vs_strong_record": s_rec,
        "vs_strong_pct": s_pct,
        "vs_strong_n": s_n,
        "vs_weak_record": w_rec,
        "vs_weak_pct": w_pct,
        "vs_weak_n": w_n,
    }


def recent_form(team_df: pd.DataFrame):
    """Last 5 / 10 and current streak based on ML games with results."""
    played = team_df[team_df["has_result"] & team_df["has_ml"]].sort_values(
        "game_date"
    )
    if played.empty:
        return {"last5_pct": np.nan, "last10_pct": np.nan, "streak": 0}

    def window(df_, n):
        sub = df_.tail(n)
        if sub.empty:
            return np.nan
        return sub["is_win"].mean()

    last5_pct = window(played, 5)
    last10_pct = window(played, 10)

    # Streak from most recent backwards
    streak = 0
    for _, row in played.sort_values("game_date", ascending=False).iterrows():
        if row["is_win"]:
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break

    return {"last5_pct": last5_pct, "last10_pct": last10_pct, "streak": streak}


def fmt_streak(streak):
    if streak == 0:
        return "—"
    if streak > 0:
        return f"W{streak}"
    return f"L{abs(streak)}"


# ------------ HTML rendering ------------------------------------------


def build_html(slate, team_results, league_tbl, outfile):

    CSS = """
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; color: #222; font-size: 14px; }
        .game-card { border: 1px solid #ccc; padding: 15px; margin-top: 25px;
                     border-radius: 6px; background: #fafafa; }
        .subheader { font-weight: bold; margin-top: 10px; }
        pre { background: #fff; border: 1px solid #ddd; padding: 10px; border-radius: 4px;
              font-family: Menlo, Monaco, Consolas, "Courier New", monospace; font-size: 14px; }
        .hl { background-color: #fff3b0; padding: 2px 4px; border-radius: 4px; display: inline-block; }
        .league-block { margin-top: 30px; }
        details { margin-top: 8px; }
        summary { cursor: pointer; font-weight: bold; }
        .summary-line { margin-top: 4px; font-weight: 500; }
    </style>
    """

    lines = []
    w = lines.append

    w("<html><head>")
    w(CSS)
    w("</head><body>")

    today = slate["game_date"].iloc[0]
    w("<h1>NBA Moneyline Dashboard</h1>")
    w(f"<h2>{today}</h2>")

    # ---------- Per-game cards ----------
    for _, g in slate.iterrows():
        home = g["home_team_name"]
        away = g["away_team_name"]
        home_ml = g["home_ml"]
        away_ml = g["away_ml"]

        home_prob = american_to_prob(home_ml)
        away_prob = american_to_prob(away_ml)

        hist_home = team_results[
            (team_results["team"] == home)
            & (team_results["game_date"] < g["game_date"])
        ]
        hist_away = team_results[
            (team_results["team"] == away)
            & (team_results["game_date"] < g["game_date"])
        ]

        home_sum = summarize_team(hist_home)
        away_sum = summarize_team(hist_away)

        bucket_home = pick_bucket(home_ml)
        bucket_away = pick_bucket(away_ml)

        bucket_home_stats = summarize_bucket(hist_home, bucket_home)
        bucket_away_stats = summarize_bucket(hist_away, bucket_away)

        oppH = opponent_adjusted_stats(hist_home, league_tbl)
        oppA = opponent_adjusted_stats(hist_away, league_tbl)

        formH = recent_form(hist_home)
        formA = recent_form(hist_away)

        home_is_fav = home_ml < 0
        home_is_dog = home_ml > 0
        away_is_fav = away_ml < 0
        away_is_dog = away_ml > 0

        # Summary line: AWAY first, then HOME
        away_tri = TEAM_TRICODES.get(away, away)
        home_tri = TEAM_TRICODES.get(home, home)

        summary_line = (
            f"{away_tri} {fmt_odds(away_ml)} ({fmt_pct(away_prob)}) | "
            f"{home_tri} {fmt_odds(home_ml)} ({fmt_pct(home_prob)})"
        )

        w("<div class='game-card'>")
        w(f"<h3>{away} @ {home}</h3>")
        w(f"<div class='summary-line'>{summary_line}</div>")

        # Collapsible details for the rest
        w("<details>")
        w("<summary>Click to expand</summary>")

        # ---------------- HOME TEAM ----------------
        w(f"<div class='subheader'>{home.upper()} (HOME)</div>")
        w("<pre>")

        block_lines = maybe_hl_block(
            [
                label("ML record:"),
                value(f"{home_sum['ml_record']} ({fmt_pct(home_sum['ml_win_pct'])})"),
            ],
            highlight=False,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As favorite:"),
                value(f"{home_sum['fav_record']} ({fmt_pct(home_sum['fav_win_pct'])})"),
            ],
            highlight=home_is_fav,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As underdog:"),
                value(f"{home_sum['dog_record']} ({fmt_pct(home_sum['dog_win_pct'])})"),
            ],
            highlight=home_is_dog,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("Home:"),
                value(
                    f"{home_sum['home_record']} ({fmt_pct(home_sum['home_win_pct'])})"
                ),
            ],
            highlight=True,  # always highlight home context line
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("Away:"),
                value(
                    f"{home_sum['away_record']} ({fmt_pct(home_sum['away_win_pct'])})"
                ),
            ],
            highlight=False,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As home favorite:"),
                value(
                    f"{home_sum['home_fav_record']} "
                    f"({fmt_pct(home_sum['home_fav_win_pct'])})"
                ),
            ],
            highlight=home_is_fav,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As home underdog:"),
                value(
                    f"{home_sum['home_dog_record']} "
                    f"({fmt_pct(home_sum['home_dog_win_pct'])})"
                ),
            ],
            highlight=home_is_dog,
        )
        for line in block_lines:
            w(line)

        if bucket_home:
            b = bucket_home_stats
            block_lines = [
                label(f"{bucket_home}:"),
                value(
                    f"{b['bucket_record']} "
                    f"({fmt_pct(b['bucket_win_pct'])}, n={b['n']})"
                ),
            ]
            for line in block_lines:
                w(line)

        block_lines = [
            label("ROI all ML bets:"),
            value(
                f"{home_sum['roi_units']:+0.1f}u "
                f"({fmt_pct(home_sum['roi_pct'])})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("ROI as favorite:"),
            value(
                f"{home_sum['fav_roi_units']:+0.1f}u "
                f"({fmt_pct(home_sum['fav_roi_pct'])})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("ROI as underdog:"),
            value(
                f"{home_sum['dog_roi_units']:+0.1f}u "
                f"({fmt_pct(home_sum['dog_roi_pct'])})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("Vs strong opps:"),
            value(
                f"{oppH['vs_strong_record']} "
                f"({fmt_pct(oppH['vs_strong_pct'])}, n={oppH['vs_strong_n']})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("Vs weak opps:"),
            value(
                f"{oppH['vs_weak_record']} "
                f"({fmt_pct(oppH['vs_weak_pct'])}, n={oppH['vs_weak_n']})"
            ),
        ]
        for line in block_lines:
            w(line)

        w("Recent form:")
        w(value(f"Last 5:   {fmt_pct(formH['last5_pct'])}"))
        w(value(f"Last 10:  {fmt_pct(formH['last10_pct'])}"))
        w(value(f"Streak:   {fmt_streak(formH['streak'])}"))

        w("</pre>")

        # ---------------- AWAY TEAM ----------------
        w(f"<div class='subheader'>{away.upper()} (AWAY)</div>")
        w("<pre>")

        block_lines = maybe_hl_block(
            [
                label("ML record:"),
                value(f"{away_sum['ml_record']} ({fmt_pct(away_sum['ml_win_pct'])})"),
            ],
            highlight=False,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As favorite:"),
                value(f"{away_sum['fav_record']} ({fmt_pct(away_sum['fav_win_pct'])})"),
            ],
            highlight=away_is_fav,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As underdog:"),
                value(f"{away_sum['dog_record']} ({fmt_pct(away_sum['dog_win_pct'])})"),
            ],
            highlight=away_is_dog,
        )
        for line in block_lines:
            w(line)

        block_lines = [
            label("Home:"),
            value(
                f"{away_sum['home_record']} "
                f"({fmt_pct(away_sum['home_win_pct'])})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("Away:"),
                value(
                    f"{away_sum['away_record']} "
                    f"({fmt_pct(away_sum['away_win_pct'])})"
                ),
            ],
            highlight=True,  # always highlight away context for away team
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As away favorite:"),
                value(
                    f"{away_sum['away_fav_record']} "
                    f"({fmt_pct(away_sum['away_fav_win_pct'])})"
                ),
            ],
            highlight=away_is_fav,
        )
        for line in block_lines:
            w(line)

        block_lines = maybe_hl_block(
            [
                label("As away underdog:"),
                value(
                    f"{away_sum['away_dog_record']} "
                    f"({fmt_pct(away_sum['away_dog_win_pct'])})"
                ),
            ],
            highlight=away_is_dog,
        )
        for line in block_lines:
            w(line)

        if bucket_away:
            b = bucket_away_stats
            block_lines = [
                label(f"{bucket_away}:"),
                value(
                    f"{b['bucket_record']} "
                    f"({fmt_pct(b['bucket_win_pct'])}, n={b['n']})"
                ),
            ]
            for line in block_lines:
                w(line)

        block_lines = [
            label("ROI all ML bets:"),
            value(
                f"{away_sum['roi_units']:+0.1f}u "
                f"({fmt_pct(away_sum['roi_pct'])})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("ROI as favorite:"),
            value(
                f"{away_sum['fav_roi_units']:+0.1f}u "
                f"({fmt_pct(away_sum['fav_roi_pct'])})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("ROI as underdog:"),
            value(
                f"{away_sum['dog_roi_units']:+0.1f}u "
                f"({fmt_pct(away_sum['dog_roi_pct'])})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("Vs strong opps:"),
            value(
                f"{oppA['vs_strong_record']} "
                f"({fmt_pct(oppA['vs_strong_pct'])}, n={oppA['vs_strong_n']})"
            ),
        ]
        for line in block_lines:
            w(line)

        block_lines = [
            label("Vs weak opps:"),
            value(
                f"{oppA['vs_weak_record']} "
                f"({fmt_pct(oppA['vs_weak_pct'])}, n={oppA['vs_weak_n']})"
            ),
        ]
        for line in block_lines:
            w(line)

        w("Recent form:")
        w(value(f"Last 5:   {fmt_pct(formA['last5_pct'])}"))
        w(value(f"Last 10:  {fmt_pct(formA['last10_pct'])}"))
        w(value(f"Streak:   {fmt_streak(formA['streak'])}"))

        w("</pre>")
        w("</details>")
        w("</div>")  # end game-card

    # --------- League overview at bottom ---------------------------
    if not league_tbl.empty:
        w("<div class='league-block'>")
        w("<h2>League Overview</h2>")

        lv = league_tbl
        overall = lv.sort_values("ml_pct", ascending=False).head(5)
        home_best = lv.sort_values("home_pct", ascending=False).head(5)
        away_best = lv.sort_values("away_pct", ascending=False).head(5)
        fav_best = lv.sort_values("fav_pct", ascending=False).head(5)
        dog_best = lv.sort_values("dog_pct", ascending=False).head(5)

        def block(title, df, col):
            w(f"<h3>{title}</h3>")
            w("<pre>")
            for i, row in enumerate(df.itertuples(index=False), start=1):
                w(f"{i}. {row.team:22s} {fmt_pct(getattr(row, col))}")
            w("</pre>")

        block("Best overall (ML win%)", overall, "ml_pct")
        block("Best home teams", home_best, "home_pct")
        block("Best away teams", away_best, "away_pct")
        block("Best favorites", fav_best, "fav_pct")
        block("Best underdogs", dog_best, "dog_pct")

        w("</div>")

    w("</body></html>")

    with open(outfile, "w") as f:
        f.write("\n".join(lines))


# ---------- main -------------------------------------------------------


def main():
    master = load_master()

    today = dt.date.today()
    if today not in master["game_date"].unique():
        today = master["game_date"].max()

    slate = master[master["game_date"] == today]

    if slate.empty:
        print("No games for", today)
        return

    team_results = build_team_results(master)
    league_tbl = league_overview(team_results)
    outfile = f"dashboard_{today}.html"

    build_html(slate, team_results, league_tbl, outfile)
    print("Dashboard written →", outfile)


if __name__ == "__main__":
    main()