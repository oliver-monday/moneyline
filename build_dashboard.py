#!/usr/bin/env python3
"""
NBA Moneyline Dashboard Generator
Uses nba_master.csv to build styled HTML dashboard with:
- Highlighted matchup state
- Indented stat formatting
- Condensed recent form
- ROI, opponent-adjusted stats, league overview
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
    (-100, 100,  "Coinflip (-100 to +100)"),
    (101, 200,  "Small dog (+101 to +200)"),
    (201, 10000, "Big dog (≥ +201)"),
]

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

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
    return 100 / (o + 100)

def fmt_odds(v):
    if pd.isna(v):
        return "—"
    return f"{int(round(v)):+d}"

def fmt_pct(v):
    if v is None or pd.isna(v):
        return "—"
    return f"{v*100:0.1f}%"

def maybe_hl(text, cond):
    return f"<span class='hl'>{text}</span>" if cond else text

def load_master():
    df = pd.read_csv(MASTER_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    for col in ["home_ml", "away_ml", "home_score", "away_score",
                "home_spread", "away_spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ROI for single game
def calc_return(row):
    if not (row["has_result"] and row["has_ml"]):
        return 0.0
    odds = row["ml"]
    if pd.isna(odds):
        return 0.0
    return (odds / 100.0) if row["is_win"] and odds > 0 else \
           (100.0 / -odds) if row["is_win"] else -1.0

# ----------------------------------------------------------------------
# TEAM RESULTS TABLE
# ----------------------------------------------------------------------

def build_team_results(master):
    rows = []
    for _, r in master.iterrows():
        gid = r["game_id"]
        d = r["game_date"]

        rows.append({
            "game_id": gid, "game_date": d,
            "team": r["home_team_name"], "opponent": r["away_team_name"],
            "is_home": True,
            "ml": r["home_ml"], "spread": r["home_spread"],
            "pf": r["home_score"], "pa": r["away_score"],
        })
        rows.append({
            "game_id": gid, "game_date": d,
            "team": r["away_team_name"], "opponent": r["home_team_name"],
            "is_home": False,
            "ml": r["away_ml"], "spread": r["away_spread"],
            "pf": r["away_score"], "pa": r["home_score"],
        })

    df = pd.DataFrame(rows)

    df["has_result"] = (~df["pf"].isna()) & (~df["pa"].isna())
    df["is_win"] = df["has_result"] & (df["pf"] > df["pa"])
    df["has_ml"] = ~df["ml"].isna()

    df["is_fav"] = df["has_ml"] & (df["ml"] < 0)
    df["is_dog"] = df["has_ml"] & (df["ml"] > 0)

    df["is_home_fav"] = df["is_home"] & df["is_fav"]
    df["is_home_dog"] = df["is_home"] & df["is_dog"]
    df["is_away_fav"] = (~df["is_home"]) & df["is_fav"]
    df["is_away_dog"] = (~df["is_home"]) & df["is_dog"]

    df["ml_bucket"] = df["ml"].apply(pick_bucket)
    df["roi"] = df.apply(calc_return, axis=1)

    return df

# ----------------------------------------------------------------------
# SUMMARIZATION FUNCTIONS
# ----------------------------------------------------------------------

def summarize_team(df):
    played = df[df["has_result"] & df["has_ml"]]
    total = len(played)
    wins = int(played["is_win"].sum())

    pct = lambda w, l: w/(w+l) if (w+l)>0 else np.nan

    fav  = played[played["is_fav"]]
    dog  = played[played["is_dog"]]
    home = played[played["is_home"]]
    away = played[~played["is_home"]]

    home_fav = played[played["is_home_fav"]]
    home_dog = played[played["is_home_dog"]]
    away_fav = played[played["is_away_fav"]]
    away_dog = played[played["is_away_dog"]]

    roi_all = played["roi"].sum()
    roi_pct = roi_all / total if total > 0 else np.nan

    def roi(df):
        return df["roi"].sum(), (df["roi"].sum() / len(df)) if len(df)>0 else np.nan

    fav_r, fav_rp = roi(fav)
    dog_r, dog_rp = roi(dog)

    return {
        "ml_record": f"{wins}-{total-wins}", "ml_pct": pct(wins, total-wins),

        "fav_record": f"{int(fav['is_win'].sum())}-{len(fav)-fav['is_win'].sum()}",
        "fav_pct": pct(fav["is_win"].sum(), len(fav)-fav["is_win"].sum()),

        "dog_record": f"{int(dog['is_win'].sum())}-{len(dog)-dog['is_win'].sum()}",
        "dog_pct": pct(dog["is_win"].sum(), len(dog)-dog["is_win"].sum()),

        "home_record": f"{int(home['is_win'].sum())}-{len(home)-home['is_win'].sum()}",
        "home_pct": pct(home["is_win"].sum(), len(home)-home["is_win"].sum()),

        "away_record": f"{int(away['is_win'].sum())}-{len(away)-away['is_win'].sum()}",
        "away_pct": pct(away["is_win"].sum(), len(away)-away['is_win'].sum()),

        "home_fav_record": f"{int(home_fav['is_win'].sum())}-{len(home_fav)-home_fav['is_win'].sum()}",
        "home_fav_pct": pct(home_fav["is_win"].sum(), len(home_fav)-home_fav["is_win"].sum()),

        "home_dog_record": f"{int(home_dog['is_win'].sum())}-{len(home_dog)-home_dog['is_win'].sum()}",
        "home_dog_pct": pct(home_dog["is_win"].sum(), len(home_dog)-home_dog["is_win"].sum()),

        "away_fav_record": f"{int(away_fav['is_win'].sum())}-{len(away_fav)-away_fav['is_win'].sum()}",
        "away_fav_pct": pct(away_fav["is_win"].sum(), len(away_fav)-away_fav["is_win"].sum()),

        "away_dog_record": f"{int(away_dog['is_win'].sum())}-{len(away_dog)-away_dog['is_win'].sum()}",
        "away_dog_pct": pct(away_dog["is_win"].sum(), len(away_dog)-away_dog["is_win"].sum()),

        "roi_units": roi_all, "roi_pct": roi_pct,
        "fav_roi_units": fav_r, "fav_roi_pct": fav_rp,
        "dog_roi_units": dog_r, "dog_roi_pct": dog_rp,
    }


def summarize_bucket(df, bucket):
    b = df[(df["ml_bucket"] == bucket) & df["has_result"] & df["has_ml"]]
    if b.empty:
        return {"bucket_record": "—", "bucket_pct": np.nan, "n": 0}

    wins = int(b["is_win"].sum())
    return {
        "bucket_record": f"{wins}-{len(b)-wins}",
        "bucket_pct": wins / len(b),
        "n": len(b),
    }


# ----------------------------------------------------------------------
# OPPONENT ADJUSTED + RECENT FORM
# ----------------------------------------------------------------------

def league_overview(team_results):
    hist = team_results[team_results["has_result"] & team_results["has_ml"]]

    rows = []
    for team, df in hist.groupby("team"):
        w = df["is_win"].sum()
        l = len(df) - w

        pct = lambda w,l: w/(w+l) if (w+l)>0 else np.nan

        fav = df[df["is_fav"]]
        dog = df[df["is_dog"]]
        home = df[df["is_home"]]
        away = df[~df["is_home"]]

        rows.append({
            "team": team,
            "ml_pct": pct(w,l),
            "home_pct": pct(home["is_win"].sum(), len(home)),
            "away_pct": pct(away["is_win"].sum(), len(away)),
            "fav_pct": pct(fav["is_win"].sum(), len(fav)),
            "dog_pct": pct(dog["is_win"].sum(), len(dog)),
        })

    return pd.DataFrame(rows)


def opponent_adjusted_stats(team_df, league_tbl):
    played = team_df[team_df["has_result"] & team_df["has_ml"]]
    if played.empty or league_tbl.empty:
        return {"strong_record":"—","strong_pct":np.nan,"strong_n":0,
                "weak_record":"—","weak_pct":np.nan,"weak_n":0}

    merged = played.merge(
        league_tbl[["team","ml_pct"]],
        left_on="opponent",
        right_on="team",
        how="left",
        suffixes=("","_opp")
    )

    merged = merged.dropna(subset=["ml_pct_opp"])
    if merged.empty:
        return {"strong_record":"—","strong_pct":np.nan,"strong_n":0,
                "weak_record":"—","weak_pct":np.nan,"weak_n":0}

    vals = league_tbl["ml_pct"].dropna()
    top = vals.quantile(0.80)
    bot = vals.quantile(0.20)

    strong = merged[merged["ml_pct_opp"] >= top]
    weak   = merged[merged["ml_pct_opp"] <= bot]

    def rec(df):
        if df.empty: return "—",np.nan,0
        w = df["is_win"].sum()
        return f"{int(w)}-{len(df)-int(w)}", w/len(df), len(df)

    srec, sp, sn = rec(strong)
    wrec, wp, wn = rec(weak)

    return {"strong_record":srec,"strong_pct":sp,"strong_n":sn,
            "weak_record":wrec,"weak_pct":wp,"weak_n":wn}


# CLEAN FIXED VERSION OF STREAK COMPUTATION
def recent_form(df):
    df = df[df["has_result"] & df["has_ml"]].sort_values("game_date")
    if df.empty:
        return {"last5":np.nan, "last10":np.nan, "streak":0}

    last5 = df.tail(5)["is_win"].mean()
    last10 = df.tail(10)["is_win"].mean()

    streak = 0
    last_results = list(df["is_win"])[::-1]

    for result in last_results:
        if result:   # win
            if streak >= 0:
                streak += 1
            else:
                break
        else:        # loss
            if streak <= 0:
                streak -= 1
            else:
                break

    return {"last5":last5, "last10":last10, "streak":streak}


def fmt_streak(n):
    return "—" if n == 0 else f"W{n}" if n > 0 else f"L{abs(n)}"


# ----------------------------------------------------------------------
# HTML BUILDING
# ----------------------------------------------------------------------

def build_html(slate, team_results, league_tbl, outfile):

    CSS = """
    <style>
        body { font-family: Arial; margin: 30px; color: #222; }
        .game-card { border: 1px solid #ccc; padding: 18px;
                     margin-top: 28px; border-radius: 6px;
                     background: #fafafa; }
        .subheader { font-weight: bold; margin-top: 12px; }
        pre {
            background: #fff; border: 1px solid #ddd;
            padding: 12px; border-radius: 5px;
            line-height: 1.35em;
            font-family: "Courier New", monospace;
        }
        .hl { background-color: #fff3b0; padding: 2px 4px; border-radius: 4px; }
    </style>
    """

    out = []; w = out.append

    w("<html><head>")
    w(CSS)
    w("</head><body>")

    date = slate["game_date"].iloc[0]
    w(f"<h1>NBA Moneyline Dashboard</h1>")
    w(f"<h2>{date}</h2>")

    # ----------------------------------------------------------
    # GAME CARDS
    # ----------------------------------------------------------
    for _, g in slate.iterrows():

        home, away = g["home_team_name"], g["away_team_name"]
        home_ml, away_ml = g["home_ml"], g["away_ml"]
        home_prob, away_prob = american_to_prob(home_ml), american_to_prob(away_ml)

        hist_home = team_results[(team_results["team"]==home) & 
                                 (team_results["game_date"] < g["game_date"])]

        hist_away = team_results[(team_results["team"]==away) &
                                 (team_results["game_date"] < g["game_date"])]

        H = summarize_team(hist_home)
        A = summarize_team(hist_away)

        bH = pick_bucket(home_ml)
        bA = pick_bucket(away_ml)
        bHs = summarize_bucket(hist_home, bH)
        bAs = summarize_bucket(hist_away, bA)

        oppH = opponent_adjusted_stats(hist_home, league_tbl)
        oppA = opponent_adjusted_stats(hist_away, league_tbl)

        formH = recent_form(hist_home)
        formA = recent_form(hist_away)

        home_fav, home_dog = (home_ml < 0), (home_ml > 0)
        away_fav, away_dog = (away_ml < 0), (away_ml > 0)

        w("<div class='game-card'>")
        w(f"<h3>{away} @ {home}</h3>")

        # ODDS
        w("<div class='subheader'>Current Odds</div>")
        w("<pre>")
        w(f"{away:25s} {fmt_odds(away_ml):>6s} | implied: {fmt_pct(away_prob)}")
        w(f"{home:25s} {fmt_odds(home_ml):>6s} | implied: {fmt_pct(home_prob)}")
        w("</pre>")

        # VENUE
        w("<div class='subheader'>Venue</div>")
        w(f"{g['venue_city']}, {g['venue_state']}<br><br>")

        # ---------------- HOME TEAM ----------------
        w(f"<div class='subheader'>{home.upper()} (HOME)</div>")
        w("<pre>")

        w(f"ML record:          {H['ml_record']:10s} ({fmt_pct(H['ml_pct'])})")

        w(maybe_hl(
            f"As favorite:        {H['fav_record']:10s} ({fmt_pct(H['fav_pct'])})",
            home_fav
        ))
        w(maybe_hl(
            f"As underdog:        {H['dog_record']:10s} ({fmt_pct(H['dog_pct'])})",
            home_dog
        ))

        w(maybe_hl(
            f"Home:               {H['home_record']:10s} ({fmt_pct(H['home_pct'])})",
            True
        ))
        w(f"Away:               {H['away_record']:10s} ({fmt_pct(H['away_pct'])})")

        w(maybe_hl(
            f"As home favorite:   {H['home_fav_record']:10s} ({fmt_pct(H['home_fav_pct'])})",
            home_fav
        ))
        w(maybe_hl(
            f"As home underdog:   {H['home_dog_record']:10s} ({fmt_pct(H['home_dog_pct'])})",
            home_dog
        ))

        if bH:
            w(f"Bucket {bH:13s}: {bHs['bucket_record']} ({fmt_pct(bHs['bucket_pct'])}, n={bHs['n']})")

        w(f"ROI all ML bets:    {H['roi_units']:+0.1f}u ({fmt_pct(H['roi_pct'])})")
        w(f"ROI as favorite:    {H['fav_roi_units']:+0.1f}u ({fmt_pct(H['fav_roi_pct'])})")
        w(f"ROI as underdog:    {H['dog_roi_units']:+0.1f}u ({fmt_pct(H['dog_roi_pct'])})")

        w(f"Vs strong opps:     {oppH['strong_record']:10s} ({fmt_pct(oppH['strong_pct'])}, n={oppH['strong_n']})")
        w(f"Vs weak opps:       {oppH['weak_record']:10s} ({fmt_pct(oppH['weak_pct'])}, n={oppH['weak_n']})")

        w("Recent form:")
        w(f"    Last 5:         {fmt_pct(formH['last5'])}")
        w(f"    Last 10:        {fmt_pct(formH['last10'])}")
        w(f"    Streak:         {fmt_streak(formH['streak'])}")

        w("</pre>")

        # ---------------- AWAY TEAM ----------------
        w(f"<div class='subheader'>{away.upper()} (AWAY)</div>")
        w("<pre>")

        w(f"ML record:          {A['ml_record']:10s} ({fmt_pct(A['ml_pct'])})")

        w(maybe_hl(
            f"As favorite:        {A['fav_record']:10s} ({fmt_pct(A['fav_pct'])})",
            away_fav
        ))
        w(maybe_hl(
            f"As underdog:        {A['dog_record']:10s} ({fmt_pct(A['dog_pct'])})",
            away_dog
        ))

        w(f"Home:               {A['home_record']:10s} ({fmt_pct(A['home_pct'])})")
        w(maybe_hl(
            f"Away:               {A['away_record']:10s} ({fmt_pct(A['away_pct'])})",
            True
        ))

        w(maybe_hl(
            f"As away favorite:   {A['away_fav_record']:10s} ({fmt_pct(A['away_fav_pct'])})",
            away_fav
        ))
        w(maybe_hl(
            f"As away underdog:   {A['away_dog_record']:10s} ({fmt_pct(A['away_dog_pct'])})",
            away_dog
        ))

        if bA:
            w(f"Bucket {bA:13s}: {bAs['bucket_record']} ({fmt_pct(bAs['bucket_pct'])}, n={bAs['n']})")

        w(f"ROI all ML bets:    {A['roi_units']:+0.1f}u ({fmt_pct(A['roi_pct'])})")
        w(f"ROI as favorite:    {A['fav_roi_units']:+0.1f}u ({fmt_pct(A['fav_roi_pct'])})")
        w(f"ROI as underdog:    {A['dog_roi_units']:+0.1f}u ({fmt_pct(A['dog_roi_pct'])})")

        w(f"Vs strong opps:     {oppA['strong_record']:10s} ({fmt_pct(oppA['strong_pct'])}, n={oppA['strong_n']})")
        w(f"Vs weak opps:       {oppA['weak_record']:10s} ({fmt_pct(oppA['weak_pct'])}, n={oppA['weak_n']})")

        w("Recent form:")
        w(f"    Last 5:         {fmt_pct(formA['last5'])}")
        w(f"    Last 10:        {fmt_pct(formA['last10'])}")
        w(f"    Streak:         {fmt_streak(formA['streak'])}")

        w("</pre>")
        w("</div>")

    # ----------------------------------------------------------
    # LEAGUE OVERVIEW
    # ----------------------------------------------------------
    w("<h2>League Overview</h2>")

    def block(title, df, col):
        w(f"<h3>{title}</h3>")
        w("<pre>")
        for i, row in enumerate(df.itertuples(index=False), start=1):
            w(f"{i}. {row.team:22s} {fmt_pct(getattr(row,col))}")
        w("</pre>")

    table = league_tbl

    block("Best overall (ML win%)",
          table.sort_values("ml_pct", ascending=False).head(5), "ml_pct")

    block("Best home teams",
          table.sort_values("home_pct", ascending=False).head(5), "home_pct")

    block("Best away teams",
          table.sort_values("away_pct", ascending=False).head(5), "away_pct")

    block("Best favorites",
          table.sort_values("fav_pct", ascending=False).head(5), "fav_pct")

    block("Best underdogs",
          table.sort_values("dog_pct", ascending=False).head(5), "dog_pct")

    w("</body></html>")

    with open(outfile, "w") as f:
        f.write("\n".join(out))


# ----------------------------------------------------------------------

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