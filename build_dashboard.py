#!/usr/bin/env python3
"""
NBA Moneyline Dashboard (Nested Formatting + Collapsible Matchups)
"""

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

MASTER_PATH = Path("nba_master.csv")

ML_BUCKETS = [
    (-10000, -300, "Big favorite (≤ -300)"),
    (-299, -151,  "Medium favorite (-299 to -151)"),
    (-150, -101,  "Small favorite (-150 to -101)"),
    (-100, 100,   "Coinflip (-100 to +100)"),
    (101, 200,    "Small dog (+101 to +200)"),
    (201, 10000,  "Big dog (≥ +201)"),
]


# ========================= HELPERS =========================

def pick_bucket(odds):
    if pd.isna(odds):
        return None
    for lo, hi, label in ML_BUCKETS:
        if lo <= odds <= hi:
            return label
    return None


def american_to_prob(odds):
    if pd.isna(odds): return None
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def fmt_odds(odds):
    if pd.isna(odds): return "—"
    return f"{int(round(odds)):+d}"


def fmt_pct(x):
    if x is None or pd.isna(x): return "—"
    return f"{x*100:0.1f}%"


def fmt_streak(s):
    if s == 0: return "—"
    return f"W{s}" if s > 0 else f"L{abs(s)}"


def hl_block(content, cond):
    """Wrap whole block if highlighted."""
    if cond:
        return f"<span class='hl'>{content}</span>"
    return content


def nest(label, value):
    """Label on one line, indented value on next."""
    return f"{label}\n    {value}"


# ========================= INGEST =========================

def load_master():
    df = pd.read_csv(MASTER_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    numeric_cols = ["home_ml","away_ml","home_score","away_score",
                    "home_spread","away_spread"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def calc_return(row):
    if (not row["has_result"]) or (not row["has_ml"]):
        return 0.0
    odds = row["ml"]
    if row["is_win"]:
        return odds/100 if odds > 0 else 100/(-odds)
    return -1.0


# ========================= TEAM RESULTS =========================

def build_team_results(master):
    rows = []
    for _, r in master.iterrows():
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "team": r["home_team_name"],
            "abbr": r["home_team_abbrev"],
            "opponent": r["away_team_name"],
            "is_home": True,
            "ml": r["home_ml"],
            "pf": r["home_score"],
            "pa": r["away_score"]
        })
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "team": r["away_team_name"],
            "abbr": r["away_team_abbrev"],
            "opponent": r["home_team_name"],
            "is_home": False,
            "ml": r["away_ml"],
            "pf": r["away_score"],
            "pa": r["home_score"]
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


def summarize_team(df):
    played = df[df["has_result"] & df["has_ml"]]
    wins = int(played["is_win"].sum())
    losses = len(played) - wins

    def pct(w, l): return w/(w+l) if (w+l)>0 else np.nan

    def rec(sub):
        w = sub["is_win"].sum()
        l = len(sub) - w
        return f"{int(w)}-{int(l)}", pct(w, l)

    fav = played[played["is_fav"]]
    dog = played[played["is_dog"]]
    home = played[played["is_home"]]
    away = played[~played["is_home"]]

    hfav = played[played["is_home_fav"]]
    hdog = played[played["is_home_dog"]]
    afav = played[played["is_away_fav"]]
    adog = played[played["is_away_dog"]]

    out = {
        "ml_record": f"{wins}-{losses}",
        "ml_pct": pct(wins, losses),
        "fav_record": rec(fav)[0], "fav_pct": rec(fav)[1],
        "dog_record": rec(dog)[0], "dog_pct": rec(dog)[1],
        "home_record": rec(home)[0], "home_pct": rec(home)[1],
        "away_record": rec(away)[0], "away_pct": rec(away)[1],
        "home_fav_record": rec(hfav)[0], "home_fav_pct": rec(hfav)[1],
        "home_dog_record": rec(hdog)[0], "home_dog_pct": rec(hdog)[1],
        "away_fav_record": rec(afav)[0], "away_fav_pct": rec(afav)[1],
        "away_dog_record": rec(adog)[0], "away_dog_pct": rec(adog)[1],
        "roi_units": played["roi"].sum(),
        "roi_pct": played["roi"].mean() if len(played)>0 else np.nan
    }
    return out


def summarize_bucket(df, bucket):
    b = df[(df["ml_bucket"] == bucket) & df["has_result"] & df["has_ml"]]
    if b.empty:
        return ("—", np.nan, 0)
    w = b["is_win"].sum()
    l = len(b) - w
    return (f"{int(w)}-{int(l)}", w/(w+l), len(b))


# ========================= OPPONENT ADJ + FORM =========================

def league_overview(team_results):
    hist = team_results[team_results["has_result"] & team_results["has_ml"]]
    rows = []
    for team, df in hist.groupby("team"):
        w = df["is_win"].sum()
        l = len(df) - w
        pct = w/(w+l) if (w+l)>0 else np.nan

        fav = df[df["is_fav"]]
        dog = df[df["is_dog"]]
        home = df[df["is_home"]]
        away = df[~df["is_home"]]

        rows.append({
            "team": team,
            "ml_pct": pct,
            "home_pct": home["is_win"].mean() if len(home)>0 else np.nan,
            "away_pct": away["is_win"].mean() if len(away)>0 else np.nan,
            "fav_pct": fav["is_win"].mean() if len(fav)>0 else np.nan,
            "dog_pct": dog["is_win"].mean() if len(dog)>0 else np.nan
        })
    return pd.DataFrame(rows)


def opponent_adjusted_stats(team_df, league_tbl):
    played = team_df[team_df["has_result"] & team_df["has_ml"]]
    if played.empty or league_tbl.empty:
        return {"vs_strong": ("—", np.nan, 0),
                "vs_weak": ("—", np.nan, 0)}

    merged = played.merge(
        league_tbl[["team","ml_pct"]],
        left_on="opponent", right_on="team", how="left"
    ).dropna(subset=["ml_pct"])

    if merged.empty:
        return {"vs_strong": ("—", np.nan, 0),
                "vs_weak": ("—", np.nan, 0)}

    cuts = league_tbl["ml_pct"].dropna()
    top, bot = cuts.quantile(0.80), cuts.quantile(0.20)

    strong = merged[merged["ml_pct"] >= top]
    weak = merged[merged["ml_pct"] <= bot]

    def rec(df):
        if df.empty: return ("—", np.nan, 0)
        w = df["is_win"].sum()
        l = len(df) - w
        return (f"{int(w)}-{int(l)}", w/(w+l), len(df))

    return {"vs_strong": rec(strong), "vs_weak": rec(weak)}


def recent_form(team_df):
    df = team_df[team_df["has_result"] & team_df["has_ml"]].sort_values("game_date")
    if df.empty:
        return {"last5": np.nan, "last10": np.nan, "streak": 0}
    last5 = df["is_win"].tail(5).mean()
    last10 = df["is_win"].tail(10).mean()

    streak = 0
    for _, row in df.sort_values("game_date", ascending=False).iterrows():
        if row["is_win"]:
            if streak >= 0: streak += 1
            else: break
        else:
            if streak <= 0: streak -= 1
            else: break

    return {"last5": last5, "last10": last10, "streak": streak}


# ========================= HTML =========================

def build_html(slate, team_results, league_tbl, outfile):

    CSS = """
    <style>
        body { font-family: Arial; margin: 30px; color: #222; line-height: 1.45; }
        .game-card { 
            border: 1px solid #ccc; 
            padding: 15px; 
            margin-top: 25px;
            border-radius: 6px; 
            background: #fafafa;
        }
        .summary-line {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        details summary {
            cursor: pointer;
            font-weight: bold;
            margin-bottom: 8px;
        }
        pre {
            background: #fff;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 4px;
            white-space: pre-wrap;
        }
        .hl { 
            background-color: #fff3b0; 
            padding: 2px 4px; 
            border-radius: 4px; 
            display: inline-block;
        }
        .league-block { margin-top: 30px; }
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

    # ----- Matchups -----
    for _, g in slate.iterrows():

        home, away = g["home_team_name"], g["away_team_name"]
        home_ml, away_ml = g["home_ml"], g["away_ml"]

        home_abbr = team_results[team_results["team"]==home]["abbr"].iloc[0]
        away_abbr = team_results[team_results["team"]==away]["abbr"].iloc[0]

        home_prob = fmt_pct(american_to_prob(home_ml))
        away_prob = fmt_pct(american_to_prob(away_ml))

        histH = team_results[(team_results["team"]==home) & 
                             (team_results["game_date"]<g["game_date"])]
        histA = team_results[(team_results["team"]==away) & 
                             (team_results["game_date"]<g["game_date"])]

        sumH, sumA = summarize_team(histH), summarize_team(histA)

        bH, bA = pick_bucket(home_ml), pick_bucket(away_ml)
        bucketH = summarize_bucket(histH, bH) if bH else None
        bucketA = summarize_bucket(histA, bA) if bA else None

        oppH = opponent_adjusted_stats(histH, league_tbl)
        oppA = opponent_adjusted_stats(histA, league_tbl)

        formH = recent_form(histH)
        formA = recent_form(histA)

        home_is_fav, away_is_fav = home_ml < 0, away_ml < 0
        home_is_dog, away_is_dog = home_ml > 0, away_ml > 0

        # ----- CARD -----
        w("<div class='game-card'>")

        # Title
        w(f"<h3>{away} @ {home}</h3>")

        # Summary line
        summary = (
            f"{away_abbr} {fmt_odds(away_ml)} ({away_prob})"
            f" | {home_abbr} {fmt_odds(home_ml)} ({home_prob})"
        )
        w(f"<div class='summary-line'>{summary}</div>")

        # Collapsible
        w("<details>")
        w("<summary>Click to expand</summary>")

        # ----- Odds -----
        w("<div class='subheader'>Current Odds</div>")
        w("<pre>")
        w(f"{away}: {fmt_odds(away_ml)}   | implied: {away_prob}")
        w(f"{home}: {fmt_odds(home_ml)}   | implied: {home_prob}")
        w("</pre>")

        # ----- Venue -----
        w("<div class='subheader'>Venue</div>")
        w(f"{g.get('venue_city','')}, {g.get('venue_state','')}")
        w("<br><br>")

        # =================== HOME TEAM BLOCK ===================
        block = []

        block.append(nest("ML record:", f"{sumH['ml_record']} ({fmt_pct(sumH['ml_pct'])})"))

        block.append(hl_block(
            nest("As favorite:", f"{sumH['fav_record']} ({fmt_pct(sumH['fav_pct'])})"),
            home_is_fav
        ))
        block.append(hl_block(
            nest("As underdog:", f"{sumH['dog_record']} ({fmt_pct(sumH['dog_pct'])})"),
            home_is_dog
        ))

        block.append(hl_block(
            nest("Home:", f"{sumH['home_record']} ({fmt_pct(sumH['home_pct'])})"),
            True
        ))
        block.append(nest("Away:", f"{sumH['away_record']} ({fmt_pct(sumH['away_pct'])})"))

        block.append(hl_block(
            nest("As home favorite:", 
                 f"{sumH['home_fav_record']} ({fmt_pct(sumH['home_fav_pct'])})"),
            home_is_fav
        ))
        block.append(hl_block(
            nest("As home underdog:", 
                 f"{sumH['home_dog_record']} ({fmt_pct(sumH['home_dog_pct'])})"),
            home_is_dog
        ))

        if bucketH:
            rec, pct, n = bucketH
            block.append(nest(f"{bH}:", f"{rec} ({fmt_pct(pct)}, n={n})"))

        block.append(nest("ROI all ML bets:", 
                          f"{sumH['roi_units']:+0.1f}u ({fmt_pct(sumH['roi_pct'])})"))

        vsS = oppH["vs_strong"]
        vsW = oppH["vs_weak"]
        block.append(nest("Vs strong opps:", f"{vsS[0]} ({fmt_pct(vsS[1])}, n={vsS[2]})"))
        block.append(nest("Vs weak opps:", f"{vsW[0]} ({fmt_pct(vsW[1])}, n={vsW[2]})"))

        block.append("Recent form:")
        block.append(f"    Last 5:  {fmt_pct(formH['last5'])}")
        block.append(f"    Last 10: {fmt_pct(formH['last10'])}")
        block.append(f"    Streak:  {fmt_streak(formH['streak'])}")

        w(f"<div class='subheader'>{home.upper()} (HOME)</div>")
        w("<pre>")
        for line in block:
            w(line)
        w("</pre>")

        # =================== AWAY TEAM BLOCK ===================
        block = []
        block.append(nest("ML record:", f"{sumA['ml_record']} ({fmt_pct(sumA['ml_pct'])})"))

        block.append(hl_block(
            nest("As favorite:", f"{sumA['fav_record']} ({fmt_pct(sumA['fav_pct'])})"),
            away_is_fav
        ))
        block.append(hl_block(
            nest("As underdog:", f"{sumA['dog_record']} ({fmt_pct(sumA['dog_pct'])})"),
            away_is_dog
        ))

        block.append(nest("Home:", f"{sumA['home_record']} ({fmt_pct(sumA['home_pct'])})"))
        block.append(hl_block(
            nest("Away:", f"{sumA['away_record']} ({fmt_pct(sumA['away_pct'])})"),
            True
        ))

        block.append(hl_block(
            nest("As away favorite:",
                 f"{sumA['away_fav_record']} ({fmt_pct(sumA['away_fav_pct'])})"),
            away_is_fav
        ))
        block.append(hl_block(
            nest("As away underdog:",
                 f"{sumA['away_dog_record']} ({fmt_pct(sumA['away_dog_pct'])})"),
            away_is_dog
        ))

        if bucketA:
            rec, pct, n = bucketA
            block.append(nest(f"{bA}:", f"{rec} ({fmt_pct(pct)}, n={n})"))

        block.append(nest("ROI all ML bets:", 
                          f"{sumA['roi_units']:+0.1f}u ({fmt_pct(sumA['roi_pct'])})"))

        vsS = oppA["vs_strong"]
        vsW = oppA["vs_weak"]
        block.append(nest("Vs strong opps:", f"{vsS[0]} ({fmt_pct(vsS[1])}, n={vsS[2]})"))
        block.append(nest("Vs weak opps:", f"{vsW[0]} ({fmt_pct(vsW[1])}, n={vsW[2]})"))

        block.append("Recent form:")
        block.append(f"    Last 5:  {fmt_pct(formA['last5'])}")
        block.append(f"    Last 10: {fmt_pct(formA['last10'])}")
        block.append(f"    Streak:  {fmt_streak(formA['streak'])}")

        w(f"<div class='subheader'>{away.upper()} (AWAY)</div>")
        w("<pre>")
        for line in block:
            w(line)
        w("</pre>")

        w("</details>")
        w("</div>")  # end card

    # ---------- LEAGUE OVERVIEW ----------
    if not league_tbl.empty:
        w("<div class='league-block'>")
        w("<h2>League Overview</h2>")

        def section(title, df, col):
            w(f"<h3>{title}</h3><pre>")
            for i, row in enumerate(df.itertuples(index=False), start=1):
                w(f"{i}. {row.team:22s} {fmt_pct(getattr(row, col))}")
            w("</pre>")

        ov = league_tbl.sort_values("ml_pct", ascending=False).head(5)
        section("Best overall (ML win%)", ov, "ml_pct")

        home = league_tbl.sort_values("home_pct", ascending=False).head(5)
        section("Best home teams", home, "home_pct")

        away = league_tbl.sort_values("away_pct", ascending=False).head(5)
        section("Best away teams", away, "away_pct")

        fav = league_tbl.sort_values("fav_pct", ascending=False).head(5)
        section("Best favorites", fav, "fav_pct")

        dog = league_tbl.sort_values("dog_pct", ascending=False).head(5)
        section("Best underdogs", dog, "dog_pct")

        w("</div>")

    # ---------- Write ----------
    w("</body></html>")
    with open(outfile, "w") as f:
        f.write("\n".join(lines))


# ========================= MAIN =========================

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