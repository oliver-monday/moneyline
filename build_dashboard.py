#!/usr/bin/env python3
"""
Generates a lightly-styled HTML dashboard summarizing today's NBA matchups.
Uses nba_master.csv.

Upgrade B + Highlight Patch + League / Trends / ROI:
- Cleaner styling
- Adds "as home favorite / home underdog / away favorite / away underdog"
- Highlights the rows that correspond to the team’s current state.
- Adds opponent-adjusted stats, recent form, ROI, and a league overview section.
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
    return f"{x*100:0.1f}%"


def maybe_hl(text, cond):
    return f"<span class='hl'>{text}</span>" if cond else text


def load_master():
    df = pd.read_csv(MASTER_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    for col in ["home_ml", "away_ml", "home_score", "away_score",
                "home_spread", "away_spread"]:
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

        rows.append({
            "game_id": gid,
            "game_date": date,
            "team": r["home_team_name"],
            "opponent": r["away_team_name"],
            "is_home": True,
            "ml": r["home_ml"],
            "spread": r["home_spread"],
            "pf": r["home_score"],
            "pa": r["away_score"],
        })
        rows.append({
            "game_id": gid,
            "game_date": date,
            "team": r["away_team_name"],
            "opponent": r["home_team_name"],
            "is_home": False,
            "ml": r["away_ml"],
            "spread": r["away_spread"],
            "pf": r["away_score"],
            "pa": r["home_score"],
        })

    df = pd.DataFrame(rows)

    df["has_result"] = (~df["pf"].isna()) & (~df["pa"].isna())
    df["is_win"] = df["has_result"] & (df["pf"] > df["pa"])
    df["has_ml"] = ~df["ml"].isna()
    df["is_fav"] = df["has_ml"] & (df["ml"] < 0)
    df["is_dog"] = df["has_ml"] & (df["ml"] > 0)

    # New state flags for highlighting:
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

        "fav_record": f"{int(fav['is_win'].sum())}-{len(fav)-fav['is_win'].sum()}",
        "fav_win_pct": pct(fav["is_win"].sum(), len(fav)-fav["is_win"].sum()),

        "dog_record": f"{int(dog['is_win'].sum())}-{len(dog)-dog['is_win'].sum()}",
        "dog_win_pct": pct(dog["is_win"].sum(), len(dog)-dog["is_win"].sum()),

        "home_record": f"{int(home['is_win'].sum())}-{len(home)-home['is_win'].sum()}",
        "home_win_pct": pct(home["is_win"].sum(), len(home)-home["is_win"].sum()),

        "away_record": f"{int(away['is_win'].sum())}-{len(away)-away['is_win'].sum()}",
        "away_win_pct": pct(away["is_win"].sum(), len(away)-away["is_win"].sum()),

        # New granular records:
        "home_fav_record": f"{int(home_fav['is_win'].sum())}-{len(home_fav)-home_fav['is_win'].sum()}",
        "home_fav_win_pct": pct(home_fav["is_win"].sum(), len(home_fav)-home_fav["is_win"].sum()),

        "home_dog_record": f"{int(home_dog['is_win'].sum())}-{len(home_dog)-home_dog['is_win'].sum()}",
        "home_dog_win_pct": pct(home_dog["is_win"].sum(), len(home_dog)-home_dog["is_win"].sum()),

        "away_fav_record": f"{int(away_fav['is_win'].sum())}-{len(away_fav)-away_fav['is_win'].sum()}",
        "away_fav_win_pct": pct(away_fav["is_win"].sum(), len(away_fav)-away_fav["is_win"].sum()),

        "away_dog_record": f"{int(away_dog['is_win'].sum())}-{len(away_dog)-away_dog['is_win'].sum()}",
        "away_dog_win_pct": pct(away_dog["is_win"].sum(), len(away_dog)-away_dog["is_win"].sum()),

        # ROI
        "roi_units": total_roi,
        "roi_pct": roi_pct,
        "fav_roi_units": fav_roi,
        "fav_roi_pct": fav_roi_pct,
        "dog_roi_units": dog_roi,
        "dog_roi_pct": dog_roi_pct,
    }


def summarize_bucket(df, bucket):
    b = df[
        (df["ml_bucket"] == bucket)
        & df["has_result"]
        & df["has_ml"]
    ]
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

def league_overview(team_results):
    """Per-team league table for use in overview + opponent strength."""
    hist = team_results[team_results["has_result"] & team_results["has_ml"]]

    rows = []
    for team, df in hist.groupby("team"):
        w = df["is_win"].sum()
        l = len(df) - w
        ml_pct = w / (w + l) if (w + l) > 0 else np.nan

        fav = df[df["is_fav"]]
        w_f = fav["is_win"].sum()
        home = df[df["is_home"]]
        w_h = home["is_win"].sum()
        away = df[~df["is_home"]]
        w_a = away["is_win"].sum()
        dog = df[df["is_dog"]]
        w_d = dog["is_win"].sum()

        rows.append({
            "team": team,
            "ml_pct": ml_pct,
            "home_pct": w_h / len(home) if len(home) > 0 else np.nan,
            "away_pct": w_a / len(away) if len(away) > 0 else np.nan,
            "fav_pct": w_f / len(fav) if len(fav) > 0 else np.nan,
            "dog_pct": w_d / len(dog) if len(dog) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def opponent_adjusted_stats(team_df, league_tbl):
    """
    Record vs strong and weak opponents based on opponents' ML win%.
    Uses league_tbl["ml_pct"] and classifies:
      - strong: opponents in top 25% of ML win%
      - weak:   opponents in bottom 25%
    If anything is missing / too small-sample, returns blanks.
    """

    # Defensive guards: no league table or no ml_pct info
    if league_tbl.empty or "ml_pct" not in league_tbl.columns:
        return {
            "vs_strong_record": "—", "vs_strong_pct": np.nan, "vs_strong_n": 0,
            "vs_weak_record": "—",   "vs_weak_pct": np.nan,   "vs_weak_n": 0,
        }

    played = team_df[team_df["has_result"] & team_df["has_ml"]]
    if played.empty:
        return {
            "vs_strong_record": "—", "vs_strong_pct": np.nan, "vs_strong_n": 0,
            "vs_weak_record": "—",   "vs_weak_pct": np.nan,   "vs_weak_n": 0,
        }

    # Attach opponent ML% explicitly as ml_pct_opp
    opp_tbl = league_tbl[["team", "ml_pct"]].rename(
        columns={"team": "opponent", "ml_pct": "ml_pct_opp"}
    )
    merged = played.merge(opp_tbl, on="opponent", how="left")

    merged = merged.dropna(subset=["ml_pct_opp"])
    if merged.empty:
        return {
            "vs_strong_record": "—", "vs_strong_pct": np.nan, "vs_strong_n": 0,
            "vs_weak_record": "—",   "vs_weak_pct": np.nan,   "vs_weak_n": 0,
        }

    base = league_tbl["ml_pct"].dropna()
    if base.empty:
        return {
            "vs_strong_record": "—", "vs_strong_pct": np.nan, "vs_strong_n": 0,
            "vs_weak_record": "—",   "vs_weak_pct": np.nan,   "vs_weak_n": 0,
        }

    # 80th / 20th percentile cutoffs
    top_cut = base.quantile(0.80)
    bot_cut = base.quantile(0.20)

    strong = merged[merged["ml_pct_opp"] >= top_cut]
    weak = merged[merged["ml_pct_opp"] <= bot_cut]

    def rec(df):
        if df.empty:
            return "—", np.nan, 0
        w = df["is_win"].sum()
        l = len(df) - w
        return f"{int(w)}-{int(l)}", w / (w + l), len(df)

    s_rec, s_pct, s_n = rec(strong)
    w_rec, w_pct, w_n = rec(weak)

    return {
        "vs_strong_record": s_rec, "vs_strong_pct": s_pct, "vs_strong_n": s_n,
        "vs_weak_record":   w_rec, "vs_weak_pct":   w_pct, "vs_weak_n":   w_n,
    }


def recent_form(team_df):
    """Last 5 / 10 and current streak based on ML games with results."""
    played = team_df[team_df["has_result"] & team_df["has_ml"]].sort_values("game_date")
    if played.empty:
        return {
            "last5": (np.nan, 0),
            "last10": (np.nan, 0),
            "streak": 0,
        }

    def window(df, n):
        sub = df.tail(n)
        if sub.empty:
            return np.nan, 0
        pct = sub["is_win"].mean()
        return pct, len(sub)

    last5_pct, last5_n = window(played, 5)
    last10_pct, last10_n = window(played, 10)

    # Streak from most recent backwards
    streak = 0
    for _, row in played[::-1].iterrows():
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

    return {
        "last5": (last5_pct, last5_n),
        "last10": (last10_pct, last10_n),
        "streak": streak,
    }


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
        body { font-family: Arial; margin: 30px; color: #222; }
        .game-card { border: 1px solid #ccc; padding: 15px; margin-top: 25px;
                     border-radius: 6px; background: #fafafa; }
        .subheader { font-weight: bold; margin-top: 10px; }
        pre { background: #fff; border: 1px solid #ddd; padding: 10px; border-radius: 4px; }
        .hl { background-color: #fff3b0; padding: 2px 4px; border-radius: 4px; }
        .league-block { margin-top: 30px; }
    </style>
    """

    lines = []
    w = lines.append

    w("<html><head>")
    w(CSS)
    w("</head><body>")

    today = slate["game_date"].iloc[0]
    w(f"<h1>NBA Moneyline Dashboard</h1>")
    w(f"<h2>{today}</h2>")

    # Per-game cards
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

        # Opponent-adjusted + form
        home_opp = opponent_adjusted_stats(hist_home, league_tbl)
        away_opp = opponent_adjusted_stats(hist_away, league_tbl)

        home_form = recent_form(hist_home)
        away_form = recent_form(hist_away)

        # Which highlight applies?
        home_is_fav = (home_ml < 0)
        home_is_dog = (home_ml > 0)
        away_is_fav = (away_ml < 0)
        away_is_dog = (away_ml > 0)

        w("<div class='game-card'>")
        w(f"<h3>{away} @ {home}</h3>")

        w("<div class='subheader'>Current Odds</div>")
        w("<pre>")
        w(f"{away}: {fmt_odds(away_ml)}   | implied: {fmt_pct(away_prob)}")
        w(f"{home}: {fmt_odds(home_ml)}   | implied: {fmt_pct(home_prob)}")
        w("</pre>")

        w("<div class='subheader'>Venue</div>")
        w(f"{g.get('venue_city','')}, {g.get('venue_state','')}")
        w("<br><br>")

        # ---------------- HOME TEAM ----------------
        w(f"<div class='subheader'>{home.upper()} (HOME)</div>")
        w("<pre>")

        w(f"ML record:      {home_sum['ml_record']} ({fmt_pct(home_sum['ml_win_pct'])})")

        w(maybe_hl(
            f"As favorite:     {home_sum['fav_record']} ({fmt_pct(home_sum['fav_win_pct'])})",
            home_is_fav
        ))

        w(maybe_hl(
            f"As underdog:     {home_sum['dog_record']} ({fmt_pct(home_sum['dog_win_pct'])})",
            home_is_dog
        ))

        # Always highlight home line
        w(maybe_hl(
            f"Home:            {home_sum['home_record']} ({fmt_pct(home_sum['home_win_pct'])})",
            True
        ))

        w(f"Away:            {home_sum['away_record']} ({fmt_pct(home_sum['away_win_pct'])})")

        # intersection states
        w(maybe_hl(
            f"As home favorite:{home_sum['home_fav_record']} ({fmt_pct(home_sum['home_fav_win_pct'])})",
            home_is_fav
        ))
        w(maybe_hl(
            f"As home underdog:{home_sum['home_dog_record']} ({fmt_pct(home_sum['home_dog_win_pct'])})",
            home_is_dog
        ))

        if bucket_home:
            b = bucket_home_stats
            w(f"Bucket {bucket_home}: {b['bucket_record']} ({fmt_pct(b['bucket_win_pct'])}, n={b['n']})")

        # ROI + opponent-adjusted + form
        w(f"ROI all ML bets: {home_sum['roi_units']:+0.1f}u ({fmt_pct(home_sum['roi_pct'])})")
        w(f"ROI as favorite: {home_sum['fav_roi_units']:+0.1f}u ({fmt_pct(home_sum['fav_roi_pct'])})")
        w(f"ROI as underdog: {home_sum['dog_roi_units']:+0.1f}u ({fmt_pct(home_sum['dog_roi_pct'])})")

        w(f"Vs strong opps:  {home_opp['vs_strong_record']} ({fmt_pct(home_opp['vs_strong_pct'])}, n={home_opp['vs_strong_n']})")
        w(f"Vs weak opps:    {home_opp['vs_weak_record']} ({fmt_pct(home_opp['vs_weak_pct'])}, n={home_opp['vs_weak_n']})")

        last5_pct, last5_n = home_form["last5"]
        last10_pct, last10_n = home_form["last10"]
        w(f"Recent form:     Last 5: {fmt_pct(last5_pct)} (n={last5_n}),  "
          f"Last 10: {fmt_pct(last10_pct)} (n={last10_n}),  Streak: {fmt_streak(home_form['streak'])}")

        w("</pre>")

        # ---------------- AWAY TEAM ----------------
        w(f"<div class='subheader'>{away.upper()} (AWAY)</div>")
        w("<pre>")

        w(f"ML record:      {away_sum['ml_record']} ({fmt_pct(away_sum['ml_win_pct'])})")

        w(maybe_hl(
            f"As favorite:     {away_sum['fav_record']} ({fmt_pct(away_sum['fav_win_pct'])})",
            away_is_fav
        ))

        w(maybe_hl(
            f"As underdog:     {away_sum['dog_record']} ({fmt_pct(away_sum['dog_win_pct'])})",
            away_is_dog
        ))

        w(f"Home:            {away_sum['home_record']} ({fmt_pct(away_sum['home_win_pct'])})")

        w(maybe_hl(
            f"Away:            {away_sum['away_record']} ({fmt_pct(away_sum['away_win_pct'])})",
            True
        ))

        w(maybe_hl(
            f"As away favorite:{away_sum['away_fav_record']} ({fmt_pct(away_sum['away_fav_win_pct'])})",
            away_is_fav
        ))
        w(maybe_hl(
            f"As away underdog:{away_sum['away_dog_record']} ({fmt_pct(away_sum['away_dog_win_pct'])})",
            away_is_dog
        ))

        if bucket_away:
            b = bucket_away_stats
            w(f"Bucket {bucket_away}: {b['bucket_record']} ({fmt_pct(b['bucket_win_pct'])}, n={b['n']})")

        w(f"ROI all ML bets: {away_sum['roi_units']:+0.1f}u ({fmt_pct(away_sum['roi_pct'])})")
        w(f"ROI as favorite: {away_sum['fav_roi_units']:+0.1f}u ({fmt_pct(away_sum['fav_roi_pct'])})")
        w(f"ROI as underdog: {away_sum['dog_roi_units']:+0.1f}u ({fmt_pct(away_sum['dog_roi_pct'])})")

        w(f"Vs strong opps:  {away_opp['vs_strong_record']} ({fmt_pct(away_opp['vs_strong_pct'])}, n={away_opp['vs_strong_n']})")
        w(f"Vs weak opps:    {away_opp['vs_weak_record']} ({fmt_pct(away_opp['vs_weak_pct'])}, n={away_opp['vs_weak_n']})")

        last5_pct, last5_n = away_form["last5"]
        last10_pct, last10_n = away_form["last10"]
        w(f"Recent form:     Last 5: {fmt_pct(last5_pct)} (n={last5_n}),  "
          f"Last 10: {fmt_pct(last10_pct)} (n={last10_n}),  Streak: {fmt_streak(away_form['streak'])}")

        w("</pre>")
        w("</div>")  # end game

    # --------- League overview at bottom (C) ---------------------------
    if not league_tbl.empty:
        w("<div class='league-block'>")
        w("<h2>League Overview</h2>")

        overall = league_tbl.sort_values("ml_pct", ascending=False).head(5)
        home_best = league_tbl.sort_values("home_pct", ascending=False).head(5)
        away_best = league_tbl.sort_values("away_pct", ascending=False).head(5)
        fav_best = league_tbl.sort_values("fav_pct", ascending=False).head(5)
        dog_best = league_tbl.sort_values("dog_pct", ascending=False).head(5)

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