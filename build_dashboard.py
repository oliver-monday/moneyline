#!/usr/bin/env python3
"""
Generates a lightly-styled HTML dashboard summarizing today's NBA matchups.
Uses nba_master.csv.

Presentation upgrades:
- Collapsible matchup sections
- Summary line below matchup title
- Removed redundant Current Odds + Venue boxes
- Nested indentation presentation for team stats
- Removed the literal word “Bucket”
- Highlight entire nested block when the row matches today’s game state
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

# --------------------------- helpers ----------------------------------

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

def fmt_streak(st):
    if st == 0:
        return "—"
    if st > 0:
        return f"W{st}"
    return f"L{abs(st)}"

def label(text: str) -> str:
    return f"{text}"

def nest(text: str, indent: int = 4) -> str:
    return " " * indent + text

def maybe_block(text_lines, highlight=False):
    """Wraps MULTI-LINE blocks in <span class='hl'> ... </span> when highlight=True."""
    if not highlight:
        return text_lines
    wrapped = ["<span class='hl'>"]
    wrapped.extend(text_lines)
    wrapped.append("</span>")
    return wrapped

# ------------------------- load & results ------------------------------

def load_master():
    df = pd.read_csv(MASTER_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in ["home_ml","away_ml","home_score","away_score","home_spread","away_spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def calc_return(row):
    if (not row["has_result"]) or (not row["has_ml"]):
        return 0.0
    odds = row["ml"]
    if row["is_win"]:
        return odds/100 if odds > 0 else 100/(-odds)
    else:
        return -1.0

def build_team_results(master):
    rows = []
    for _, r in master.iterrows():
        gid = r["game_id"]
        date = r["game_date"]

        rows.append(dict(
            game_id=gid, game_date=date, team=r["home_team_name"],
            opponent=r["away_team_name"], is_home=True,
            ml=r["home_ml"], spread=r["home_spread"],
            pf=r["home_score"], pa=r["away_score"]
        ))
        rows.append(dict(
            game_id=gid, game_date=date, team=r["away_team_name"],
            opponent=r["home_team_name"], is_home=False,
            ml=r["away_ml"], spread=r["away_spread"],
            pf=r["away_score"], pa=r["home_score"]
        ))

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
    total = len(played)
    wins = int(played["is_win"].sum())
    losses = total - wins

    def pct(w, l): return w/(w+l) if w+l>0 else np.nan

    fav  = played[played["is_fav"]]
    dog  = played[played["is_dog"]]
    home = played[played["is_home"]]
    away = played[~played["is_home"]]

    home_fav = played[played["is_home_fav"]]
    home_dog = played[played["is_home_dog"]]
    away_fav = played[played["is_away_fav"]]
    away_dog = played[played["is_away_dog"]]

    total_roi = played["roi"].sum()
    fav_roi   = fav["roi"].sum()
    dog_roi   = dog["roi"].sum()

    return dict(
        ml_record=f"{wins}-{losses}",
        ml_win_pct=pct(wins,losses),

        fav_record=f"{int(fav['is_win'].sum())}-{len(fav)-fav['is_win'].sum()}",
        fav_win_pct=pct(fav["is_win"].sum(), len(fav)-fav["is_win"].sum()),

        dog_record=f"{int(dog['is_win'].sum())}-{len(dog)-dog['is_win'].sum()}",
        dog_win_pct=pct(dog["is_win"].sum(), len(dog)-dog["is_win"].sum()),

        home_record=f"{int(home['is_win'].sum())}-{len(home)-home['is_win'].sum()}",
        home_win_pct=pct(home["is_win"].sum(), len(home)-home["is_win"].sum()),

        away_record=f"{int(away['is_win'].sum())}-{len(away)-away['is_win'].sum()}",
        away_win_pct=pct(away["is_win"].sum(), len(away)-away["is_win"].sum()),

        home_fav_record=f"{int(home_fav['is_win'].sum())}-{len(home_fav)-home_fav['is_win'].sum()}",
        home_fav_win_pct=pct(home_fav["is_win"].sum(), len(home_fav)-home_fav["is_win"].sum()),

        home_dog_record=f"{int(home_dog['is_win'].sum())}-{len(home_dog)-home_dog['is_win'].sum()}",
        home_dog_win_pct=pct(home_dog["is_win"].sum(), len(home_dog)-home_dog["is_win"].sum()),

        away_fav_record=f"{int(away_fav['is_win'].sum())}-{len(away_fav)-away_fav['is_win'].sum()}",
        away_fav_win_pct=pct(away_fav["is_win"].sum(), len(away_fav)-away_fav["is_win"].sum()),

        away_dog_record=f"{int(away_dog['is_win'].sum())}-{len(away_dog)-away_dog['is_win'].sum()}",
        away_dog_win_pct=pct(away_dog["is_win"].sum(), len(away_dog)-away_dog["is_win"].sum()),

        roi_units=total_roi,
        roi_pct= total_roi/total if total>0 else np.nan,
        fav_roi_units=fav_roi,
        fav_roi_pct=fav_roi/len(fav) if len(fav)>0 else np.nan,
        dog_roi_units=dog_roi,
        dog_roi_pct=dog_roi/len(dog) if len(dog)>0 else np.nan
    )

def summarize_bucket(df, bucket):
    b = df[(df["ml_bucket"]==bucket) & df["has_result"] & df["has_ml"]]
    if b.empty:
        return dict(bucket_record="—", bucket_win_pct=np.nan, n=0)
    w = int(b["is_win"].sum())
    return dict(
        bucket_record=f"{w}-{len(b)-w}",
        bucket_win_pct=w/len(b),
        n=len(b)
    )

def opponent_adjusted_stats(team_df, league_tbl):
    hist = team_df[team_df["has_result"] & team_df["has_ml"]]
    if league_tbl.empty or "ml_pct" not in league_tbl.columns:
        return dict(vs_strong_record="—",vs_strong_pct=np.nan,vs_strong_n=0,
                    vs_weak_record="—",vs_weak_pct=np.nan,vs_weak_n=0)

    merged = hist.merge(league_tbl[["team","ml_pct"]],
                        left_on="opponent", right_on="team", how="left",
                        suffixes=("", "_opp"))
    col = "ml_pct_opp" if "ml_pct_opp" in merged.columns else "ml_pct"
    merged = merged.dropna(subset=[col])
    if merged.empty:
        return dict(vs_strong_record="—",vs_strong_pct=np.nan,vs_strong_n=0,
                    vs_weak_record="—",vs_weak_pct=np.nan,vs_weak_n=0)

    pct_vals = league_tbl["ml_pct"].dropna()
    top = pct_vals.quantile(0.80)
    bot = pct_vals.quantile(0.20)

    strong = merged[merged[col] >= top]
    weak   = merged[merged[col] <= bot]

    def rec(df):
        if df.empty: return "—", np.nan, 0
        w = df["is_win"].sum()
        return f"{int(w)}-{int(len(df)-w)}", w/len(df), len(df)

    srec, spct, sn = rec(strong)
    wrec, wpct, wn = rec(weak)

    return dict(vs_strong_record=srec,vs_strong_pct=spct,vs_strong_n=sn,
                vs_weak_record=wrec,vs_weak_pct=wpct,vs_weak_n=wn)

def recent_form(team_df):
    hist = team_df[team_df["has_result"] & team_df["has_ml"]].sort_values("game_date")
    if hist.empty:
        return dict(last5_pct=np.nan,last10_pct=np.nan,streak=0)
    last5  = hist.tail(5)["is_win"].mean()
    last10 = hist.tail(10)["is_win"].mean()

    st=0
    for _, row in hist.sort_values("game_date",ascending=False).iterrows():
        if row["is_win"]:
            if st>=0: st+=1
            else: break
        else:
            if st<=0: st-=1
            else: break

    return dict(last5_pct=last5,last10_pct=last10,streak=st)

# -------------------------- HTML RENDERING -----------------------------

def build_html(slate, team_results, league_tbl, outfile):

    CSS = """
    <style>
        body { font-family: Arial; margin: 30px; color: #222; }
        .game-card { border: 1px solid #ccc; padding: 15px; margin-top: 25px;
                     border-radius: 6px; background: #fafafa; }
        summary { font-weight: bold; cursor: pointer; margin-top: 5px; }
        pre { background: #fff; border: 1px solid #ddd; padding: 10px; border-radius: 4px; }
        .hl { background-color: #fff3b0; padding: 0 2px; border-radius: 3px; }
        .league-block { margin-top: 40px; }
    </style>
    """

    lines=[]
    w=lines.append
    w("<html><head>"+CSS+"</head><body>")

    today=slate["game_date"].iloc[0]
    w("<h1>NBA Moneyline Dashboard</h1>")
    w(f"<h2>{today}</h2>")

    for _, g in slate.iterrows():
        home=g["home_team_name"]
        away=g["away_team_name"]
        home_ml=g["home_ml"]
        away_ml=g["away_ml"]
        home_prob=american_to_prob(home_ml)
        away_prob=american_to_prob(away_ml)

        hist_home=team_results[(team_results["team"]==home)&(team_results["game_date"]<g["game_date"])]
        hist_away=team_results[(team_results["team"]==away)&(team_results["game_date"]<g["game_date"])]

        home_sum=summarize_team(hist_home)
        away_sum=summarize_team(hist_away)

        bucket_home=pick_bucket(home_ml)
        bucket_away=pick_bucket(away_ml)

        bH=summarize_bucket(hist_home,bucket_home)
        bA=summarize_bucket(hist_away,bucket_away)

        oppH=opponent_adjusted_stats(hist_home,league_tbl)
        oppA=opponent_adjusted_stats(hist_away,league_tbl)

        formH=recent_form(hist_home)
        formA=recent_form(hist_away)

        home_is_fav = home_ml <0
        home_is_dog = home_ml >0
        away_is_fav = away_ml <0
        away_is_dog = away_ml >0

        summary_line=f"{away[:3].upper()} {fmt_odds(away_ml)} ({fmt_pct(away_prob)}) | {home[:3].upper()} {fmt_odds(home_ml)} ({fmt_pct(home_prob)})"

        w("<div class='game-card'>")
        w(f"<h3>{away} @ {home}</h3>")
        w(f"<div><strong>{summary_line}</strong></div>")

        w("<details><summary>Click to expand</summary>")
        w("<br>")

        # ----------- HOME TEAM BLOCK -------------
        block=[]
        block.append(f"{label('ML record:')}")
        block.append(nest(f"{home_sum['ml_record']} ({fmt_pct(home_sum['ml_win_pct'])})"))

        block.append(f"{label('As favorite:')}")
        block.append(nest(f"{home_sum['fav_record']} ({fmt_pct(home_sum['fav_win_pct'])})"))

        block.append(f"{label('As underdog:')}")
        block.append(nest(f"{home_sum['dog_record']} ({fmt_pct(home_sum['dog_win_pct'])})"))

        block.append(f"{label('Home:')}")
        block.append(nest(f"{home_sum['home_record']} ({fmt_pct(home_sum['home_win_pct'])})"))

        block.append(f"{label('Away:')}")
        block.append(nest(f"{home_sum['away_record']} ({fmt_pct(home_sum['away_win_pct'])})"))

        block.append(f"{label('As home favorite:')}")
        block.append(nest(f"{home_sum['home_fav_record']} ({fmt_pct(home_sum['home_fav_win_pct'])})"))

        block.append(f"{label('As home underdog:')}")
        block.append(nest(f"{home_sum['home_dog_record']} ({fmt_pct(home_sum['home_dog_win_pct'])})"))

        if bucket_home:
            block.append(label(bucket_home+":"))
            block.append(nest(f"{bH['bucket_record']} ({fmt_pct(bH['bucket_win_pct'])}, n={bH['n']})"))

        block.append(label("ROI all ML bets:"))
        block.append(nest(f"{home_sum['roi_units']:+0.1f}u ({fmt_pct(home_sum['roi_pct'])})"))

        block.append(label("ROI as favorite:"))
        block.append(nest(f"{home_sum['fav_roi_units']:+0.1f}u ({fmt_pct(home_sum['fav_roi_pct'])})"))

        block.append(label("ROI as underdog:"))
        block.append(nest(f"{home_sum['dog_roi_units']:+0.1f}u ({fmt_pct(home_sum['dog_roi_pct'])})"))

        block.append(label("Vs strong opps:"))
        block.append(nest(f"{oppH['vs_strong_record']} ({fmt_pct(oppH['vs_strong_pct'])}, n={oppH['vs_strong_n']})"))

        block.append(label("Vs weak opps:"))
        block.append(nest(f"{oppH['vs_weak_record']} ({fmt_pct(oppH['vs_weak_pct'])}, n={oppH['vs_weak_n']})"))

        block.append(label("Recent form:"))
        block.append(nest(f"Last 5:   {fmt_pct(formH['last5_pct'])}"))
        block.append(nest(f"Last 10:  {fmt_pct(formH['last10_pct'])}"))
        block.append(nest(f"Streak:   {fmt_streak(formH['streak'])}"))

        block = maybe_block(block, highlight=(home_is_fav or home_is_dog))

        w("<div class='subheader'>"+home.upper()+" (HOME)</div>")
        w("<pre>")
        for line in block: w(line)
        w("</pre><br>")

        # ----------- AWAY TEAM BLOCK -------------
        block=[]
        block.append(label("ML record:"))
        block.append(nest(f"{away_sum['ml_record']} ({fmt_pct(away_sum['ml_win_pct'])})"))

        block.append(label("As favorite:"))
        block.append(nest(f"{away_sum['fav_record']} ({fmt_pct(away_sum['fav_win_pct'])})"))

        block.append(label("As underdog:"))
        block.append(nest(f"{away_sum['dog_record']} ({fmt_pct(away_sum['dog_win_pct'])})"))

        block.append(label("Home:"))
        block.append(nest(f"{away_sum['home_record']} ({fmt_pct(away_sum['home_win_pct'])})"))

        block.append(label("Away:"))
        block.append(nest(f"{away_sum['away_record']} ({fmt_pct(away_sum['away_win_pct'])})"))

        block.append(label("As away favorite:"))
        block.append(nest(f"{away_sum['away_fav_record']} ({fmt_pct(away_sum['away_fav_win_pct'])})"))

        block.append(label("As away underdog:"))
        block.append(nest(f"{away_sum['away_dog_record']} ({fmt_pct(away_sum['away_dog_win_pct'])})"))

        if bucket_away:
            block.append(label(bucket_away+":"))
            block.append(nest(f"{bA['bucket_record']} ({fmt_pct(bA['bucket_win_pct'])}, n={bA['n']})"))

        block.append(label("ROI all ML bets:"))
        block.append(nest(f"{away_sum['roi_units']:+0.1f}u ({fmt_pct(away_sum['roi_pct'])})"))

        block.append(label("ROI as favorite:"))
        block.append(nest(f"{away_sum['fav_roi_units']:+0.1f}u ({fmt_pct(away_sum['fav_roi_pct'])})"))

        block.append(label("ROI as underdog:"))
        block.append(nest(f"{away_sum['dog_roi_units']:+0.1f}u ({fmt_pct(away_sum['dog_roi_pct'])})"))

        block.append(label("Vs strong opps:"))
        block.append(nest(f"{oppA['vs_strong_record']} ({fmt_pct(oppA['vs_strong_pct'])}, n={oppA['vs_strong_n']})"))

        block.append(label("Vs weak opps:"))
        block.append(nest(f"{oppA['vs_weak_record']} ({fmt_pct(oppA['vs_weak_pct'])}, n={oppA['vs_weak_n']})"))

        block.append(label("Recent form:"))
        block.append(nest(f"Last 5:   {fmt_pct(formA['last5_pct'])}"))
        block.append(nest(f"Last 10:  {fmt_pct(formA['last10_pct'])}"))
        block.append(nest(f"Streak:   {fmt_streak(formA['streak'])}"))

        block = maybe_block(block, highlight=(away_is_fav or away_is_dog))

        w("<div class='subheader'>"+away.upper()+" (AWAY)</div>")
        w("<pre>")
        for line in block: w(line)
        w("</pre>")

        w("</details>")
        w("</div>")  # end game card

    # --------- LEAGUE OVERVIEW ----------
    if not league_tbl.empty:
        w("<div class='league-block'>")
        w("<h2>League Overview</h2>")

        lv=league_tbl
        overall=lv.sort_values("ml_pct",ascending=False).head(5)
        home_best=lv.sort_values("home_pct",ascending=False).head(5)
        away_best=lv.sort_values("away_pct",ascending=False).head(5)
        fav_best=lv.sort_values("fav_pct",ascending=False).head(5)
        dog_best=lv.sort_values("dog_pct",ascending=False).head(5)

        def block(title,df,col):
            w(f"<h3>{title}</h3><pre>")
            for i,row in enumerate(df.itertuples(index=False),start=1):
                w(f"{i}. {row.team:22s} {fmt_pct(getattr(row,col))}")
            w("</pre>")

        block("Best overall (ML win%)",overall,"ml_pct")
        block("Best home teams",home_best,"home_pct")
        block("Best away teams",away_best,"away_pct")
        block("Best favorites",fav_best,"fav_pct")
        block("Best underdogs",dog_best,"dog_pct")

        w("</div>")

    w("</body></html>")

    with open(outfile,"w") as f:
        f.write("\n".join(lines))


# -------------------------- MAIN --------------------------------------

def main():
    master=load_master()
    today=dt.date.today()
    if today not in master["game_date"].unique():
        today=master["game_date"].max()

    slate=master[master["game_date"]==today]
    if slate.empty:
        print("No games for",today)
        return

    results=build_team_results(master)
    league_tbl=league_overview(results)

    outfile=f"dashboard_{today}.html"
    build_html(slate,results,league_tbl,outfile)

    print("Dashboard written →",outfile)

if __name__=="__main__":
    main()