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
- Team Form Index (0–100) combining last10 form + opponent strength
- Mismatch Index (team vs opponent)
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
    return f"{x * 100:0.1f}%"

def maybe_hl_block(lines, highlight):
    if not highlight:
        return lines
    return ["<span class='hl'>", *lines, "</span>"]

def label(text: str) -> str:
    return text

def value(text: str, indent: int = 4) -> str:
    return " " * indent + text

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
    if pd.isna(odds): return 0.0
    if row["is_win"]:
        return odds/100 if odds>0 else 100/(-odds)
    return -1.0

# ------------ team results --------------------------------------------

def build_team_results(master):
    rows = []
    for _, r in master.iterrows():
        gid = r["game_id"]; date = r["game_date"]
        rows.append({
            "game_id": gid, "game_date": date,
            "team": r["home_team_name"], "opponent": r["away_team_name"],
            "is_home": True, "ml": r["home_ml"], "spread": r["home_spread"],
            "pf": r["home_score"], "pa": r["away_score"]
        })
        rows.append({
            "game_id": gid, "game_date": date,
            "team": r["away_team_name"], "opponent": r["home_team_name"],
            "is_home": False, "ml": r["away_ml"], "spread": r["away_spread"],
            "pf": r["away_score"], "pa": r["home_score"]
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
    total = len(played); wins = int(played["is_win"].sum()); losses = total - wins
    def pct(w,l): return w/(w+l) if w+l>0 else np.nan
    fav = played[played["is_fav"]]; dog = played[played["is_dog"]]
    home = played[played["is_home"]]; away = played[~played["is_home"]]
    home_fav = played[played["is_home_fav"]]; home_dog = played[played["is_home_dog"]]
    away_fav = played[played["is_away_fav"]]; away_dog = played[played["is_away_dog"]]
    total_roi = played["roi"].sum(); roi_pct = total_roi/total if total>0 else np.nan
    fav_roi = fav["roi"].sum(); fav_roi_pct = fav_roi/len(fav) if len(fav)>0 else np.nan
    dog_roi = dog["roi"].sum(); dog_roi_pct = dog_roi/len(dog) if len(dog)>0 else np.nan
    return {
        "ml_record": f"{wins}-{losses}", "ml_win_pct": pct(wins,losses),
        "fav_record": f"{int(fav['is_win'].sum())}-{len(fav)-fav['is_win'].sum()}",
        "fav_win_pct": pct(fav["is_win"].sum(), len(fav)-fav["is_win"].sum()),
        "dog_record": f"{int(dog['is_win'].sum())}-{len(dog)-dog['is_win'].sum()}",
        "dog_win_pct": pct(dog["is_win"].sum(), len(dog)-dog["is_win"].sum()),
        "home_record": f"{int(home['is_win'].sum())}-{len(home)-home['is_win'].sum()}",
        "home_win_pct": pct(home["is_win"].sum(), len(home)-home["is_win"].sum()),
        "away_record": f"{int(away['is_win'].sum())}-{len(away)-away['is_win'].sum()}",
        "away_win_pct": pct(away["is_win"].sum(), len(away)-away["is_win"].sum()),
        "home_fav_record": f"{int(home_fav['is_win'].sum())}-{len(home_fav)-home_fav['is_win'].sum()}",
        "home_fav_win_pct": pct(home_fav["is_win"].sum(), len(home_fav)-home_fav["is_win"].sum()),
        "home_dog_record": f"{int(home_dog['is_win'].sum())}-{len(home_dog)-home_dog['is_win'].sum()}",
        "home_dog_win_pct": pct(home_dog["is_win"].sum(), len(home_dog)-home_dog["is_win"].sum()),
        "away_fav_record": f"{int(away_fav['is_win'].sum())}-{len(away_fav)-away_fav['is_win'].sum()}",
        "away_fav_win_pct": pct(away_fav["is_win"].sum(), len(away_fav)-away_fav["is_win"].sum()),
        "away_dog_record": f"{int(away_dog['is_win'].sum())}-{len(away_dog)-away_dog['is_win'].sum()}",
        "away_dog_win_pct": pct(away_dog["is_win"].sum(), len(away_dog)-away_dog["is_win"].sum()),
        "roi_units": total_roi, "roi_pct": roi_pct,
        "fav_roi_units": fav_roi, "fav_roi_pct": fav_roi_pct,
        "dog_roi_units": dog_roi, "dog_roi_pct": dog_roi_pct,
    }

def summarize_bucket(df, bucket):
    b=df[(df["ml_bucket"]==bucket)&df["has_result"]&df["has_ml"]]
    if b.empty: return {"bucket_record":"—","bucket_win_pct":np.nan,"n":0}
    wins=int(b["is_win"].sum()); losses=len(b)-wins
    return {"bucket_record":f"{wins}-{losses}","bucket_win_pct":wins/(wins+losses),"n":len(b)}

# ------------ opponent-adjusted & form ---------------------------------

def league_overview(team_results):
    hist = team_results[team_results["has_result"] & team_results["has_ml"]]
    rows=[]
    for team, df in hist.groupby("team"):
        w=df["is_win"].sum(); l=len(df)-w; ml_pct=w/(w+l) if (w+l)>0 else np.nan
        fav=df[df["is_fav"]]; w_f=fav["is_win"].sum()
        home=df[df["is_home"]]; w_h=home["is_win"].sum()
        away=df[~df["is_home"]]; w_a=away["is_win"].sum()
        dog=df[df["is_dog"]]; w_d=dog["is_win"].sum()
        rows.append({
            "team":team,"ml_pct":ml_pct,
            "home_pct":w_h/len(home) if len(home)>0 else np.nan,
            "away_pct":w_a/len(away) if len(away)>0 else np.nan,
            "fav_pct":w_f/len(fav) if len(fav)>0 else np.nan,
            "dog_pct":w_d/len(dog) if len(dog)>0 else np.nan,
        })
    return pd.DataFrame(rows)

def opponent_adjusted_stats(team_df, league_tbl):
    if league_tbl.empty or "ml_pct" not in league_tbl.columns:
        return {
            "vs_strong_record":"—","vs_strong_pct":np.nan,"vs_strong_n":0,
            "vs_weak_record":"—","vs_weak_pct":np.nan,"vs_weak_n":0
        }
    played=team_df[team_df["has_result"]&team_df["has_ml"]]
    if played.empty:
        return {
            "vs_strong_record":"—","vs_strong_pct":np.nan,"vs_strong_n":0,
            "vs_weak_record":"—","vs_weak_pct":np.nan,"vs_weak_n":0
        }
    merged=played.merge(
        league_tbl[["team","ml_pct"]],
        left_on="opponent", right_on="team",
        how="left", suffixes=("", "_opp")
    )
    col="ml_pct_opp" if "ml_pct_opp" in merged.columns else "ml_pct"
    merged=merged.dropna(subset=[col])
    if merged.empty:
        return {
            "vs_strong_record":"—","vs_strong_pct":np.nan,"vs_strong_n":0,
            "vs_weak_record":"—","vs_weak_pct":np.nan,"vs_weak_n":0
        }
    base=league_tbl["ml_pct"].dropna()
    if base.empty:
        return {
            "vs_strong_record":"—","vs_strong_pct":np.nan,"vs_strong_n":0,
            "vs_weak_record":"—","vs_weak_pct":np.nan,"vs_weak_n":0
        }
    top_cut=base.quantile(0.80)
    bot_cut=base.quantile(0.20)
    strong=merged[merged[col]>=top_cut]; weak=merged[merged[col]<=bot_cut]
    def rec(df_):
        if df_.empty: return "—",np.nan,0
        w=df_["is_win"].sum(); l=len(df_)-w
        return f"{int(w)}-{int(l)}", w/(w+l), len(df_)
    s_rec,s_pct,s_n = rec(strong)
    w_rec,w_pct,w_n = rec(weak)
    return {
        "vs_strong_record":s_rec, "vs_strong_pct":s_pct, "vs_strong_n":s_n,
        "vs_weak_record":w_rec, "vs_weak_pct":w_pct, "vs_weak_n":w_n
    }

def recent_form(team_df):
    played=team_df[team_df["has_result"]&team_df["has_ml"]].sort_values("game_date")
    if played.empty:
        return {"last5_pct":np.nan,"last10_pct":np.nan,"streak":0}
    def window(df,n):
        sub=df.tail(n)
        return sub["is_win"].mean() if not sub.empty else np.nan
    last5=window(played,5); last10=window(played,10)
    streak=0
    for _,row in played.sort_values("game_date",ascending=False).iterrows():
        if row["is_win"]:
            if streak>=0: streak+=1
            else: break
        else:
            if streak<=0: streak-=1
            else: break
    return {"last5_pct":last5,"last10_pct":last10,"streak":streak}

def fmt_streak(st):
    if st==0: return "—"
    if st>0: return f"W{st}"
    return f"L{abs(st)}"

# ------------ Team Form Index -----------------------------------------

def _to_float_or_nan(x):
    try: return float(x)
    except: return np.nan

def compute_form_index(last10_pct, vs_strong_pct, vs_weak_pct):
    last10=_to_float_or_nan(last10_pct)
    vs_strong=_to_float_or_nan(vs_strong_pct)
    vs_weak=_to_float_or_nan(vs_weak_pct)
    if np.isnan(last10):
        return np.nan, "insufficient sample"
    base = last10
    adj = 0.0
    if not np.isnan(vs_strong): adj += 0.20*(vs_strong-0.5)
    if not np.isnan(vs_weak): adj += 0.10*(vs_weak-0.5)
    idx = max(0.0, min(1.0, base+adj))
    score = idx*100
    if score>=70: desc="Strong uptrend"
    elif score>=60: desc="Mild uptrend"
    elif score>=45: desc="Stable"
    elif score>=35: desc="Mild downtrend"
    else: desc="Cold stretch"
    return score, desc

# ======================================================================
# === MISMATCH INDEX (new) =============================================
# ======================================================================

def compute_mismatch_index(team_form, opp_form):
    """
    Simple, interpretable: difference of Form Index scores.
    Output: mismatch_score (-100 to 100), descriptor.
    """
    if pd.isna(team_form) or pd.isna(opp_form):
        return np.nan, "insufficient sample"

    raw = team_form - opp_form  # could be -100..+100

    # Descriptor
    if raw >= 40: desc = "Major advantage"
    elif raw >= 20: desc = "Clear advantage"
    elif raw > -20: desc = "Balanced"
    elif raw > -40: desc = "Clear disadvantage"
    else: desc = "Major disadvantage"

    return raw, desc

# ======================================================================
# === PERFORMANCE RELATIVE TO MARKET (PRM) =============================
# ======================================================================

def compute_prm(trailing_roi):
    """
    trailing_roi: list or pandas Series of last N ROI values (we use last 10)
    Returns:
        (prm_score, descriptor)
    Scoring:
        > +2.5u  → Strongly undervalued
        > +1.0u  → Mildly undervalued
        > -1.0u  → Fairly priced
        > -2.5u  → Mildly overvalued
        else     → Strongly overvalued
    """
    if trailing_roi is None or len(trailing_roi) == 0:
        return np.nan, "insufficient sample"

    total = np.nansum(trailing_roi)

    # Descriptor logic
    if total >= 2.5:
        desc = "Strongly undervalued"
    elif total >= 1.0:
        desc = "Mildly undervalued"
    elif total > -1.0:
        desc = "Fairly priced"
    elif total > -2.5:
        desc = "Mildly overvalued"
    else:
        desc = "Strongly overvalued"

    return total, desc

# ------------ HTML rendering ------------------------------------------

def build_html(slate, team_results, league_tbl, outfile):

    CSS = """
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; color: #222; }
        .game-card { border: 1px solid #ccc; padding: 17px; margin-top: 25px;
                     border-radius: 6px; background: #fafafa; }
        .subheader { font-weight: bold; margin-top: 10px; margin-bottom: 10px}
        pre {
            background: #fff;
            border: 1px solid #ddd;
            padding: 12px;
            border-radius: 4px;
            font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
            font-size: 12px;
            line-height: 1.35;
        }
        .hl { background-color: #fff3b0; padding: 2px 4px; border-radius: 4px; display: inline-block; }
        .league-block { margin-top: 30px; }
        details { margin-top: 8px; }
        summary { cursor: pointer; font-weight: bold; }
        .summary-line { margin-top: 4px; font-weight: 500; }
    </style>
    """

    lines=[]; w=lines.append

    w("<html><head>"); w(CSS); w("</head><body>")

    today = slate["game_date"].iloc[0]
    w("<h1>NBA Moneyline Dashboard</h1>")
    w(f"<h2>{today}</h2>")

    # ---------- Per-game cards ----------
    for _, g in slate.iterrows():

        home = g["home_team_name"]
        away = g["away_team_name"]
        home_ml = g["home_ml"]; away_ml = g["away_ml"]

        home_prob = american_to_prob(home_ml)
        away_prob = american_to_prob(away_ml)

        hist_home = team_results[(team_results["team"]==home)&(team_results["game_date"]<g["game_date"])]
        hist_away = team_results[(team_results["team"]==away)&(team_results["game_date"]<g["game_date"])]

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

        # Form Index
        home_form_score, home_form_desc = compute_form_index(formH["last10_pct"], oppH["vs_strong_pct"], oppH["vs_weak_pct"])
        away_form_score, away_form_desc = compute_form_index(formA["last10_pct"], oppA["vs_strong_pct"], oppA["vs_weak_pct"])

        # === NEW: Mismatch Index ===
        home_mis_score, home_mis_desc = compute_mismatch_index(home_form_score, away_form_score)
        away_mis_score, away_mis_desc = compute_mismatch_index(away_form_score, home_form_score)

        home_is_fav = home_ml < 0; home_is_dog = home_ml > 0
        away_is_fav = away_ml < 0; away_is_dog = away_ml > 0

        # --- PRM: Performance Relative to Market (last 10 ROI) ---
        home_last10_roi = hist_home.sort_values("game_date")["roi"].tail(10)
        away_last10_roi = hist_away.sort_values("game_date")["roi"].tail(10)

        home_prm_score, home_prm_desc = compute_prm(home_last10_roi)
        away_prm_score, away_prm_desc = compute_prm(away_last10_roi)

        # Summary line (using tricodes)
        away_abbr=g.get("away_team_abbrev",""); home_abbr=g.get("home_team_abbrev","")
        summary_line=f"{away_abbr} {fmt_odds(away_ml)} ({fmt_pct(away_prob)}) | {home_abbr} {fmt_odds(home_ml)} ({fmt_pct(home_prob)})"

        w("<div class='game-card'>")
        w(f"<h3>{away} @ {home}</h3>")
        w(f"<div class='summary-line'>{summary_line}</div>")
        w("<details>"); w("<summary>Click to expand</summary>")

        # ---------------- HOME TEAM ----------------
        w(f"<div class='subheader'>{home.upper()} (HOME)</div>")
        w("<pre>")

        w(label("ML record:")); w(value(f"{home_sum['ml_record']} ({fmt_pct(home_sum['ml_win_pct'])})"))

        bl = maybe_hl_block([label("As favorite:"), value(f"{home_sum['fav_record']} ({fmt_pct(home_sum['fav_win_pct'])})")], highlight=home_is_fav)
        for x in bl: w(x)

        bl = maybe_hl_block([label("As underdog:"), value(f"{home_sum['dog_record']} ({fmt_pct(home_sum['dog_win_pct'])})")], highlight=home_is_dog)
        for x in bl: w(x)

        bl = maybe_hl_block([label("Home:"), value(f"{home_sum['home_record']} ({fmt_pct(home_sum['home_win_pct'])})")], highlight=True)
        for x in bl: w(x)

        w(label("Away:")); w(value(f"{home_sum['away_record']} ({fmt_pct(home_sum['away_win_pct'])})"))

        bl = maybe_hl_block([label("As home favorite:"), value(f"{home_sum['home_fav_record']} ({fmt_pct(home_sum['home_fav_win_pct'])})")], highlight=home_is_fav)
        for x in bl: w(x)

        bl = maybe_hl_block([label("As home underdog:"), value(f"{home_sum['home_dog_record']} ({fmt_pct(home_sum['home_dog_win_pct'])})")], highlight=home_is_dog)
        for x in bl: w(x)

        if bucket_home:
            b=bucket_home_stats
            w(label(f"{bucket_home}:")); w(value(f"{b['bucket_record']} ({fmt_pct(b['bucket_win_pct'])}, n={b['n']})"))

        w(label("ROI all ML bets:")); w(value(f"{home_sum['roi_units']:+0.1f}u ({fmt_pct(home_sum['roi_pct'])})"))
        w(label("ROI as favorite:")); w(value(f"{home_sum['fav_roi_units']:+0.1f}u ({fmt_pct(home_sum['fav_roi_pct'])})"))
        w(label("ROI as underdog:")); w(value(f"{home_sum['dog_roi_units']:+0.1f}u ({fmt_pct(home_sum['dog_roi_pct'])})"))

        w(label("Vs strong opps:")); w(value(f"{oppH['vs_strong_record']} ({fmt_pct(oppH['vs_strong_pct'])}, n={oppH['vs_strong_n']})"))
        w(label("Vs weak opps:")); w(value(f"{oppH['vs_weak_record']} ({fmt_pct(oppH['vs_weak_pct'])}, n={oppH['vs_weak_n']})"))

        w("Recent form:")
        w(value(f"Last 5:   {fmt_pct(formH['last5_pct'])}"))
        w(value(f"Last 10:  {fmt_pct(formH['last10_pct'])}"))
        w(value(f"Streak:   {fmt_streak(formH['streak'])}"))

        # === Insert Form Index at bottom ===
        if pd.isna(home_form_score):
            w(label("Form index:")); w(value("— (insufficient sample)"))
        else:
            w(label("Form index:")); w(value(f"{home_form_score:0.0f} ({home_form_desc})"))

        # === NEW: Insert Mismatch Index right after Form Index ===
        if pd.isna(home_mis_score):
            w(label("Mismatch index:")); w(value("— (insufficient sample)"))
        else:
            w(label("Mismatch index:"))
            w(value(f"{home_mis_score:+0.0f} ({home_mis_desc})"))

        # === PRM: Performance Relative to Market ===
        if pd.isna(home_prm_score):
            w(label("PRM (vs market):")); w(value("— (insufficient sample)"))
        else:
            w(label("PRM (vs market):"))
            w(value(f"{home_prm_score:+0.1f}u ({home_prm_desc})"))

        w("</pre>")

        # ---------------- AWAY TEAM ----------------
        w(f"<div class='subheader'>{away.upper()} (AWAY)</div>")
        w("<pre>")

        w(label("ML record:")); w(value(f"{away_sum['ml_record']} ({fmt_pct(away_sum['ml_win_pct'])})"))

        bl = maybe_hl_block([label("As favorite:"), value(f"{away_sum['fav_record']} ({fmt_pct(away_sum['fav_win_pct'])})")], highlight=away_is_fav)
        for x in bl: w(x)

        bl = maybe_hl_block([label("As underdog:"), value(f"{away_sum['dog_record']} ({fmt_pct(away_sum['dog_win_pct'])})")], highlight=away_is_dog)
        for x in bl: w(x)

        w(label("Home:")); w(value(f"{away_sum['home_record']} ({fmt_pct(away_sum['home_win_pct'])})"))

        bl = maybe_hl_block([label("Away:"), value(f"{away_sum['away_record']} ({fmt_pct(away_sum['away_win_pct'])})")], highlight=True)
        for x in bl: w(x)

        bl = maybe_hl_block([label("As away favorite:"), value(f"{away_sum['away_fav_record']} ({fmt_pct(away_sum['away_fav_win_pct'])})")], highlight=away_is_fav)
        for x in bl: w(x)

        bl = maybe_hl_block([label("As away underdog:"), value(f"{away_sum['away_dog_record']} ({fmt_pct(away_sum['away_dog_win_pct'])})")], highlight=away_is_dog)
        for x in bl: w(x)

        if bucket_away:
            b=bucket_away_stats
            w(label(f"{bucket_away}:")); w(value(f"{b['bucket_record']} ({fmt_pct(b['bucket_win_pct'])}, n={b['n']})"))

        w(label("ROI all ML bets:")); w(value(f"{away_sum['roi_units']:+0.1f}u ({fmt_pct(away_sum['roi_pct'])})"))
        w(label("ROI as favorite:")); w(value(f"{away_sum['fav_roi_units']:+0.1f}u ({fmt_pct(away_sum['fav_roi_pct'])})"))
        w(label("ROI as underdog:")); w(value(f"{away_sum['dog_roi_units']:+0.1f}u ({fmt_pct(away_sum['dog_roi_pct'])})"))

        w(label("Vs strong opps:")); w(value(f"{oppA['vs_strong_record']} ({fmt_pct(oppA['vs_strong_pct'])}, n={oppA['vs_strong_n']})"))
        w(label("Vs weak opps:")); w(value(f"{oppA['vs_weak_record']} ({fmt_pct(oppA['vs_weak_pct'])}, n={oppA['vs_weak_n']})"))

        w("Recent form:")
        w(value(f"Last 5:   {fmt_pct(formA['last5_pct'])}"))
        w(value(f"Last 10:  {fmt_pct(formA['last10_pct'])}"))
        w(value(f"Streak:   {fmt_streak(formA['streak'])}"))

        # === Form Index (away) ===
        if pd.isna(away_form_score):
            w(label("Form index:")); w(value("— (insufficient sample)"))
        else:
            w(label("Form index:")); w(value(f"{away_form_score:0.0f} ({away_form_desc})"))

        # === Mismatch Index (away) ===
        if pd.isna(away_mis_score):
            w(label("Mismatch index:")); w(value("— (insufficient sample)"))
        else:
            w(label("Mismatch index:"))
            w(value(f"{away_mis_score:+0.0f} ({away_mis_desc})"))

        # === PRM: Performance Relative to Market ===
        if pd.isna(away_prm_score):
            w(label("PRM (vs market):")); w(value("— (insufficient sample)"))
        else:
            w(label("PRM (vs market):"))
            w(value(f"{away_prm_score:+0.1f}u ({away_prm_desc})"))    

        w("</pre>")
        w("</details>")
        w("</div>")  # end game-card

    # --------- League overview ---------
    if not league_tbl.empty:
        w("<div class='league-block'>"); w("<h2>League Overview</h2>")
        lv=league_tbl
        def block(title, df, col):
            w(f"<h3>{title}</h3>"); w("<pre>")
            for i,row in enumerate(df.itertuples(index=False),start=1):
                w(f"{i}. {row.team:22s} {fmt_pct(getattr(row,col))}")
            w("</pre>")
        block("Best overall (ML win%)", lv.sort_values("ml_pct",ascending=False).head(5), "ml_pct")
        block("Best home teams", lv.sort_values("home_pct",ascending=False).head(5), "home_pct")
        block("Best away teams", lv.sort_values("away_pct",ascending=False).head(5), "away_pct")
        block("Best favorites", lv.sort_values("fav_pct",ascending=False).head(5), "fav_pct")
        block("Best underdogs", lv.sort_values("dog_pct",ascending=False).head(5), "dog_pct")
        w("</div>")

    w("</body></html>")
    with open(outfile,"w") as f:
        f.write("\n".join(lines))

# ---------- main -------------------------------------------------------

def main():
    master = load_master()
    today = dt.date.today()
    if today not in master["game_date"].unique():
        today = master["game_date"].max()
    slate = master[master["game_date"] == today]
    if slate.empty:
        print("No games for", today); return
    team_results = build_team_results(master)
    league_tbl = league_overview(team_results)
    outfile = f"dashboard_{today}.html"
    build_html(slate, team_results, league_tbl, outfile)
    print("Dashboard written →", outfile)

if __name__ == "__main__":
    main()