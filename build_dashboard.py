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

import json
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import re

MASTER_PATH = Path("nba_master.csv")

ML_BUCKETS = [
    (-10000, -300, "Big favorite (≤ -300)"),
    (-299, -151, "Med. favorite (-299 to -151)"),
    (-150, -101, "Small favorite (-150 to -101)"),
    (-100, 100, "Coinflip (-100 to +100)"),
    (101, 200, "Small dog (+101 to +200)"),
    (201, 10000, "Big dog (≥ +201)"),
]

# Approximate arena coordinates for travel-distance modeling (miles)
CITY_COORDS = {
    ("Atlanta", "GA"): (33.7573, -84.3963),
    ("Boston", "MA"): (42.3662, -71.0621),
    ("Brooklyn", "NY"): (40.6826, -73.9754),
    ("Charlotte", "NC"): (35.2251, -80.8392),
    ("Chicago", "IL"): (41.8807, -87.6742),
    ("Cleveland", "OH"): (41.4965, -81.6882),
    ("Dallas", "TX"): (32.7905, -96.8104),
    ("Denver", "CO"): (39.7487, -105.0077),
    ("Detroit", "MI"): (42.3410, -83.0550),
    ("San Francisco", "CA"): (37.7680, -122.3877),  # Warriors
    ("Houston", "TX"): (29.7508, -95.3621),
    ("Indianapolis", "IN"): (39.7639, -86.1555),
    ("Los Angeles", "CA"): (34.0430, -118.2673),     # Lakers / Clippers
    ("Memphis", "TN"): (35.1382, -90.0506),
    ("Miami", "FL"): (25.7814, -80.1870),
    ("Milwaukee", "WI"): (43.0451, -87.9165),
    ("Minneapolis", "MN"): (44.9795, -93.2760),
    ("New Orleans", "LA"): (29.9489, -90.0810),
    ("New York", "NY"): (40.7505, -73.9934),
    ("Oklahoma City", "OK"): (35.4634, -97.5151),
    ("Orlando", "FL"): (28.5392, -81.3839),
    ("Philadelphia", "PA"): (39.9012, -75.1720),
    ("Phoenix", "AZ"): (33.4457, -112.0712),
    ("Portland", "OR"): (45.5316, -122.6668),
    ("Sacramento", "CA"): (38.6480, -121.5180),
    ("San Antonio", "TX"): (29.4270, -98.4375),
    ("Toronto", "ON"): (43.6435, -79.3791),
    ("Salt Lake City", "UT"): (40.7683, -111.9011),
    ("Washington", "DC"): (38.8981, -77.0209),
}

def haversine_miles(coord1, coord2):
    """Great-circle distance between two (lat, lon) tuples in miles."""
    if coord1 is None or coord2 is None:
        return np.nan

    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R = 3958.8  # Earth radius in miles
    return float(R * c)


def compute_schedule_stress(team_hist, today, venue_city, venue_state):
    """
    Composite schedule stress for today's game based on:
      - rest days
      - games played in last 7 days
      - travel distance from last venue to today's venue
      - consecutive road games
    Returns dict with score, desc, and raw components.
    """
    if team_hist.empty:
        return {
            "score": np.nan,
            "desc": "insufficient sample",
            "rest_days": None,
            "is_b2b": None,
            "games_7": 0,
            "travel_miles": np.nan,
            "road_streak": 0,
        }

    hist = team_hist.sort_values("game_date")
    last_game = hist.iloc[-1]
    last_date = last_game["game_date"]

    # Days between game *dates*
    days_diff = (today - last_date).days
    if days_diff < 0:
        # Corrupt future date; bail out
        return {
            "score": np.nan,
            "desc": "insufficient sample",
            "rest_days": None,
            "is_b2b": None,
            "games_7": 0,
            "travel_miles": np.nan,
            "road_streak": 0,
        }

    # Full off-days between games (e.g. last Tue, today Thu → 1 full rest day)
    rest_days = max(0, days_diff - 1)
    is_b2b = (days_diff == 1)

    # Games in last 7 days (excluding today)
    window_start = today - dt.timedelta(days=7)
    games_7 = int(
        hist[(hist["game_date"] >= window_start) & (hist["game_date"] < today)].shape[0]
    )

    # Travel from last venue to today's venue
    prev_coord = CITY_COORDS.get(
        (last_game.get("venue_city"), last_game.get("venue_state"))
    )
    today_coord = CITY_COORDS.get((venue_city, venue_state))
    travel_miles = haversine_miles(prev_coord, today_coord)

    # Consecutive road games (going backwards from last game)
    road_streak = 0
    for _, row in hist.sort_values("game_date", ascending=False).iterrows():
        if not row["is_home"]:
            road_streak += 1
        else:
            break

    # Compute consecutive home games
    home_streak = 0
    # iterate from most recent backward
    for _, row in hist.sort_values("game_date", ascending=False).iterrows():  
        if row["is_home"]:  # game was at home
            home_streak += 1
        else:
            break

    # Enforce mutual exclusivity between home_streak and road_streak
    if road_streak > 0 and home_streak > 0:
        # Look at the most recent game to decide which streak is real
        last_game = hist.sort_values("game_date", ascending=False).iloc[0]
        if last_game["is_home"]:
            road_streak = 0
        else:
            home_streak = 0

    # ---- Composite score 0–100 ----
    score = 0.0

    # Rest component
    if rest_days >= 3:
        rest_comp = 0.0
    elif rest_days == 2:
        rest_comp = 10.0
    elif rest_days == 1:
        rest_comp = 25.0
    else:  # 0 rest days → back-to-back
        rest_comp = 40.0
    score += rest_comp

    # Games last 7 days
    score += min(games_7 * 4.0, 20.0)

    # Travel miles (cap at 20)
    if not np.isnan(travel_miles):
        score += min(travel_miles / 50.0, 20.0)

    # Road stretch (cap at 15)
    score += min(road_streak * 5.0, 15.0)

    score = min(100.0, score)

    # Descriptor
    if score <= 20:
        desc = "Fully rested"
    elif score <= 35:
        desc = "Mild fatigue"
    elif score <= 55:
        desc = "Moderate fatigue"
    elif score <= 75:
        desc = "High fatigue"
    else:
        desc = "Severe fatigue spot"

    return {
        "score": score,
        "desc": desc,
        "rest_days": rest_days,
        "is_b2b": is_b2b,
        "games_7": games_7,
        "travel_miles": travel_miles,
        "road_streak": road_streak,
        "home_streak": home_streak,
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

def label(text: str) -> str:
    if not text:
        return "<span class='detail-label detail-label-empty'>&nbsp;</span>"
    return f"<span class='detail-label'>{text}</span>"

def _pill(text: str, cls: str) -> str:
    return f"<span class='pill {cls}'>{text}</span>"

def _record_pill(text: str, highlight: bool = False) -> str:
    m = re.search(r"(\d+)-(\d+)\s*\((\d+(?:\.\d+)?)%\)", text)
    if not m:
        return text
    record = m.group(0)
    if highlight:
        pill = _pill(record, "pill-yellow")
    else:
        pct = float(m.group(3))
        if pct >= 80:
            cls = "pill-green"
        elif pct <= 20:
            cls = "pill-red"
        else:
            cls = "pill-gray"
        pill = _pill(record, cls)
    return text.replace(record, pill, 1)

def _qual_pill(text: str) -> str:
    phrases = {
        "Major advantage": "pill-green",
        "Clear advantage": "pill-green",
        "Strong uptrend": "pill-green",
        "Major disadvantage": "pill-red",
        "Clear disadvantage": "pill-red",
        "High fatigue": "pill-red",
        "Severe fatigue spot": "pill-red",
        "Severe fatigue": "pill-red",
        "Weak downtrend": "pill-red",
        "Cold stretch": "pill-red",
        "Market undervaluing": "pill-green",
        "Market overvaluing": "pill-red",
        "Market roughly efficient": "pill-gray",
    }
    base = (text or "").strip()
    if not base:
        return text

    # Handle "X (Descriptor)" by removing parens and pilling descriptor.
    m = re.match(r"^(.*)\(([^)]+)\)\s*$", base)
    if m:
        left = m.group(1).strip()
        desc = m.group(2).strip()
        for phrase, cls in phrases.items():
            if desc.lower() == phrase.lower():
                return f"{left} {_pill(desc, cls)}".strip()
        return text

    # Handle whole-value descriptors (e.g., Market signal)
    for phrase, cls in phrases.items():
        if base.lower() == phrase.lower():
            return _pill(base, cls)
    return text

def value_html(text: str, highlight: bool = False) -> str:
    s = "" if text is None else str(text)
    s = _record_pill(s, highlight=highlight)
    if not highlight:
        s = _qual_pill(s)
    return f"<span class='detail-value'>{s}</span>"

def line(label_text: str, value_text: str, highlight: bool = False) -> str:
    return f"<div class='detail-line'>{label(label_text)}{value_html(value_text, highlight=highlight)}</div>"

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
            "is_home": True,
            "ml": r["home_ml"], "spread": r["home_spread"],
            "pf": r["home_score"], "pa": r["away_score"],
            "venue_city": r.get("venue_city"),
            "venue_state": r.get("venue_state"),
        })
        rows.append({
            "game_id": gid, "game_date": date,
            "team": r["away_team_name"], "opponent": r["home_team_name"],
            "is_home": False,
            "ml": r["away_ml"], "spread": r["away_spread"],
            "pf": r["away_score"], "pa": r["home_score"],
            "venue_city": r.get("venue_city"),
            "venue_state": r.get("venue_state"),
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

    # ----------------------------------------------------------------------
    # PRM helper: count undervalued / overvalued games + signal
    # ----------------------------------------------------------------------
def analyze_prm_pattern(roi_series):
    """
    roi_series: last N ROI values (typically last 10)

    Returns:
        undervalued_count  - # games with positive ROI
        overvalued_count   - # games with negative ROI
        signal_text        - Natural-language interpretation
    """
    if roi_series is None or len(roi_series) == 0:
        return 0, 0, "insufficient sample"

    vals = [x for x in roi_series if not pd.isna(x)]

    underval = sum(1 for x in vals if x > 0)
    overval = sum(1 for x in vals if x < 0)

    # Natural-language market signal
    if underval - overval >= 4:
        signal = "Market undervaluing"
    elif overval - underval >= 4:
        signal = "Market overvaluing"
    else:
        signal = "Market roughly efficient"

    return underval, overval, signal

# ------------ injuries output -----------------------------------------

def _short_status(s: str) -> str:
    t = (s or "").strip().lower()
    if not t:
        return ""
    if "out" in t:
        return "OUT"
    if "doubt" in t:
        return "OUT"
    if "question" in t:
        return "Q"
    if "prob" in t:
        return "PROB"
    if "day" in t or "dtd" in t:
        return "DTD"
    return s.strip().upper()


def _parse_injury_string(inj_text):
    """
    Accepts strings like:
      "Tari Eason (Out); D. Finney-Smith (Out); F. VanVleet (Questionable)"
    Returns:
      [{"name": "...", "status":"OUT", "details":"..."}...]
    """
    if inj_text is None:
        return []
    s = str(inj_text).strip()
    if not s or s.lower() in ("nan", "none"):
        return []

    parts = re.split(r"[;\n]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", p)
        if m:
            name = m.group(1).strip()
            status_raw = m.group(2).strip()
        else:
            name = p
            status_raw = ""

        out.append({
            "name": name,
            "status": _short_status(status_raw),
            "details": p,
        })
    return out


def write_injury_files(slate: pd.DataFrame, asof_date: dt.date) -> None:
    """
    Build:
      - data/injuries_today.json  (used by players.html; overwritten daily)
      - logs/injury_log.csv       (history; append-ish; not deployed)
    Source of truth: nba_master.csv

    Key behavior:
      Choose the earliest game_date >= asof_date that actually has injury text.
    """
    master = pd.read_csv("nba_master.csv", dtype=str)

    # Robust date normalization
    master_dates = pd.to_datetime(master["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    home_inj = master.get("home_injuries", pd.Series([""] * len(master))).fillna("").astype(str).str.strip()
    away_inj = master.get("away_injuries", pd.Series([""] * len(master))).fillna("").astype(str).str.strip()

    has_inj = home_inj.ne("") | away_inj.ne("")
    master_inj = master.loc[has_inj].copy()
    inj_dates = pd.to_datetime(master_inj["game_date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()

    fallback = pd.to_datetime(asof_date, errors="coerce").strftime("%Y-%m-%d")

    # Choose earliest date >= fallback that has injuries; else use latest date with injuries; else fallback
    inj_dates_sorted = sorted(inj_dates)
    candidates = [d for d in inj_dates_sorted if d >= fallback]
    if candidates:
        target_ymd = candidates[0]
    elif inj_dates_sorted:
        target_ymd = inj_dates_sorted[-1]
    else:
        target_ymd = fallback

    todays = master[master_dates == target_ymd].copy()

    injuries_today = {}
    log_rows = []

    for _, g in todays.iterrows():
        for side in ("home", "away"):
            team_abbrev = (g.get(f"{side}_team_abbrev") or "").strip().upper()
            inj_text = g.get(f"{side}_injuries")

            if not team_abbrev:
                continue
            if inj_text is None:
                continue

            s = str(inj_text).strip()
            if not s or s.lower() in ("nan", "none"):
                continue

            entries = _parse_injury_string(s)
            if not entries:
                continue

            injuries_today.setdefault(team_abbrev, [])
            injuries_today[team_abbrev].extend(entries)

            for e in entries:
                log_rows.append({
                    "asof_date": target_ymd,
                    "team_abbrev": team_abbrev,
                    "player_name": e.get("name", ""),
                    "status": e.get("status", ""),
                    "details": e.get("details", ""),
                    "source": "rotowire",
                })

    # Always write JSON (even if empty)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "injuries_today.json", "w") as f:
        json.dump(injuries_today, f, indent=2)

    # Write log only if we found injuries
    if log_rows:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "injury_log.csv"

        new_df = pd.DataFrame(log_rows)
        if log_path.exists():
            old_df = pd.read_csv(log_path, dtype=str)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.drop_duplicates(
            subset=["asof_date", "team_abbrev", "player_name"],
            keep="last",
        )
        combined.to_csv(log_path, index=False)

    print(f"[injuries] target_date={target_ymd} teams={len(injuries_today)} rows_logged={len(log_rows)}")


# ------------ HTML rendering ------------------------------------------

def build_html(slate, team_results, league_tbl, outfile):

    CSS = """
    <style>
        html { -webkit-text-size-adjust: 100%; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            margin: 16px;
            color: #111;
        }
        .nav {
            display:grid;
            grid-template-columns: 1fr 1fr;
            gap:12px;
            margin: 10px 0 18px;
            width: 100%;
        }
        .nav a {
            text-decoration:none;
            padding:12px 16px;
            border:1px solid #ddd;
            border-radius:14px;
            color:#111;
            text-align:center;
            font-size: 18px;
            font-weight: 600;
        }
        .nav a.active { background:#111; color:#fff; border-color:#111; }
        @media (max-width: 520px) {
            .nav a {
                padding: 14px 16px;
                font-size: 18px;
                border-radius: 16px;
            }
        }
        .muted { color:#666; }
        @media (max-width: 520px) {
            body { margin: 12px; }
        }
        .masthead { width: fit-content; max-width: 100%; }
        .brand { display:flex; align-items:center; gap:10px; width: 100%; margin: 0 0 10px; }
        .brand-logo { height: 80px; width: auto; object-fit: contain; }
        .brand-text { font-size: 80px; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
        @media (min-width: 521px) {
            .brand-text { font-size: 80px; }
            .brand-logo { height: 80px; }
        }
        .page-subtitle { margin: 0 0 18px; }
        @media (max-width: 520px) {
            .brand-text { font-size: clamp(70px, 16vw, 108px); }
            .brand-logo { height: clamp(70px, 16vw, 108px); }
            .masthead { width: 100%; }
        }
        .game-card {
            border: 1px solid #eee;
            padding: 12px;
            margin: 10px 0;
            border-radius: 14px;
            background: #fff;
        }
        .game-details summary {
            cursor: pointer;
            list-style: none;
            position: relative;
            padding-right: 24px;
        }
        .game-details summary::-webkit-details-marker { display: none; }
        .game-details summary::after {
            content: "›";
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            opacity: 0.35;
            transition: transform 0.15s ease;
        }
        .game-details[open] summary::after {
            transform: translateY(-50%) rotate(90deg);
        }
        .game-title { font-size: 20px; font-weight: 700; }
        .game-high {
            background: #e8f7e8 !important;   /* soft green */
            border-color: #6ac46a !important;
        }
        .game-avoid {
            background: #fbeaea !important;   /* soft red */
            border-color: #e08b8b !important;
        }
        .subheader { font-weight: bold; margin-bottom: 10px; padding-top: 4px; }
        .team-card {
            background: #fff;
            border: 1px solid #e6e6e6;
            border-radius: 18px;
            padding: 16px;
            margin: 14px 0;
            box-shadow: 0 1px 0 rgba(0,0,0,.03);
        }
        .details-block {
            display: flex;
            flex-direction: column;
            gap: 8px;
            font-size: 14px;
            line-height: 1.45;
        }
        .detail-line { display: flex; gap: 10px; align-items: flex-start; }
        .detail-label { font-weight: 600; color: #444; min-width: 190px; }
        .detail-label-empty { visibility: hidden; }
        .detail-value { color: #111; }
        .league-list .detail-line { gap: 6px; }
        .league-list .detail-label { min-width: 26px; }
        .pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            border: 1px solid #ddd;
            font-weight: 600;
        }
        .pill-yellow { background: #fff3b0; border-color: #e3d27a; }
        .pill-green { background: #e8f7e8; border-color: #6ac46a; }
        .pill-red { background: #fbeaea; border-color: #e08b8b; }
        .pill-gray { background: #fff; border-color: #ddd; color: #333; }
        .league-block { margin-top: 30px; }
        details { margin-top: 8px; }
        summary { font-weight: bold; color: #111; }
        .summary-line { margin-top: 4px; margin-bottom: 12px; font-weight: 600; color: #666; }
        .two-col {
            display: flex;
            gap: 30px;
        }
        .col {
            width: 50%;
        }
        @media (max-width: 900px) {
            .two-col { flex-direction: column; gap: 14px; }
            .col { width: 100%; }
        }
        @media (max-width: 600px) {
            .detail-line { justify-content: space-between; }
            .detail-label { min-width: auto; }
            .detail-value { margin-left: auto; text-align: right; }
            .league-list .detail-line { justify-content: flex-start; }
            .league-list .detail-value { margin-left: 0; text-align: left; }
        }
    </style>
    """

    lines=[]; w=lines.append

    w("<html><head>")
    w("<title>NBA GPT</title>")
    w('<meta name="viewport" content="width=device-width,initial-scale=1" />')
    w('<link rel="icon" type="image/png" href="./NBAGPTlogo.png">')
    w('<link rel="apple-touch-icon" href="./NBAGPTlogo.png">')
    w('<link rel="manifest" href="./manifest.webmanifest">')
    w('<meta name="theme-color" content="#ffffff">')
    w('<meta name="apple-mobile-web-app-capable" content="yes">')
    w('<meta name="apple-mobile-web-app-status-bar-style" content="default">')
    w(CSS)
    w("</head><body>")

    today = slate["game_date"].iloc[0]
    today_display = pd.Timestamp(today).strftime("%m-%d-%Y")
    w('<div class="masthead">')
    w('<div class="brand"><img class="brand-logo" src="./NBAGPTlogo.png" alt="NBA GPT logo"><div class="brand-text">NBA GPT</div></div>')
    w('<div class="nav"><a class="active" href="./index.html">Moneylines</a><a href="./players.html">Player Props</a></div>')
    w('</div>')
    w(f"<div class=\"muted page-subtitle\">{today_display}</div>")

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

        # PRM pattern analysis (undervalued/overvalued counts + market signal)
        home_underval, home_overval, home_prm_signal = analyze_prm_pattern(home_last10_roi)
        away_underval, away_overval, away_prm_signal = analyze_prm_pattern(away_last10_roi)

        # Schedule Stress Index (home & away)
        today_date = g["game_date"]
        venue_city = g.get("venue_city")
        venue_state = g.get("venue_state")

        home_ss = compute_schedule_stress(hist_home, today_date, venue_city, venue_state)
        away_ss = compute_schedule_stress(hist_away, today_date, venue_city, venue_state)

        # Injuries
        home_inj = g.get("home_injuries")
        away_inj = g.get("away_injuries")

        # Summary line (using tricodes)
        away_abbr=g.get("away_team_abbrev",""); home_abbr=g.get("home_team_abbrev","")
        summary_line=f"{away_abbr} {fmt_odds(away_ml)} ({fmt_pct(away_prob)}) | {home_abbr} {fmt_odds(home_ml)} ({fmt_pct(home_prob)})"

        # === Confidence classification (NEW) ===
        confidence_level = "neutral"

        # A high-confidence favorite:
        # - strong mismatch advantage
        # - good form
        # - modest schedule disadvantage
        if (home_mis_score >= 20 and home_form_score >= 60 and home_ss["score"] <= 40):
            confidence_level = "high"
        elif (away_mis_score >= 20 and away_form_score >= 60 and away_ss["score"] <= 40):
            confidence_level = "high"

        # AVOID conditions:
        if (abs(home_mis_score) < 10 and abs(away_mis_score) < 10):
            confidence_level = "stay-away"
        elif (home_ss["score"] >= 70 or away_ss["score"] >= 70):
            confidence_level = "stay-away"

        card_class = "game-card"
        if confidence_level == "high":
            card_class += " game-high"
        elif confidence_level == "stay-away":
            card_class += " game-avoid"

        w(f"<div class='{card_class}'>")
        w("<details class='game-details'>")
        w("<summary>")
        w(f"<div class='game-title'>{away} @ {home}</div>")
        w(f"<div class='summary-line'>{summary_line}</div>")
        w("</summary>")

        # ---------------- HOME TEAM ----------------
        w("<div class='team-card'>")
        w(f"<div class='subheader'>{home.upper()} (HOME)</div>")
        w("<div class='two-col'>")
        w("<div class='col'><div class='details-block'>")
        
        # LEFT column items go below:

        w(line("Record:", f"{home_sum['ml_record']} ({fmt_pct(home_sum['ml_win_pct'])})"))
        w(line("As favorite:", f"{home_sum['fav_record']} ({fmt_pct(home_sum['fav_win_pct'])})", highlight=home_is_fav))
        w(line("As underdog:", f"{home_sum['dog_record']} ({fmt_pct(home_sum['dog_win_pct'])})", highlight=home_is_dog))
        w(line("Home:", f"{home_sum['home_record']} ({fmt_pct(home_sum['home_win_pct'])})", highlight=True))
        w(line("Away:", f"{home_sum['away_record']} ({fmt_pct(home_sum['away_win_pct'])})"))
        w(line("As home favorite:", f"{home_sum['home_fav_record']} ({fmt_pct(home_sum['home_fav_win_pct'])})", highlight=home_is_fav))
        w(line("As home underdog:", f"{home_sum['home_dog_record']} ({fmt_pct(home_sum['home_dog_win_pct'])})", highlight=home_is_dog))

        if bucket_home:
            b=bucket_home_stats
            w(line(f"{bucket_home}:", f"{b['bucket_record']} ({fmt_pct(b['bucket_win_pct'])})"))

        w(line("ROI all ML bets:", f"{home_sum['roi_units']:+0.1f}u ({fmt_pct(home_sum['roi_pct'])})"))
        w(line("ROI as favorite:", f"{home_sum['fav_roi_units']:+0.1f}u ({fmt_pct(home_sum['fav_roi_pct'])})"))
        w(line("ROI as underdog:", f"{home_sum['dog_roi_units']:+0.1f}u ({fmt_pct(home_sum['dog_roi_pct'])})"))

        w(line("Vs strong opps:", f"{oppH['vs_strong_record']} ({fmt_pct(oppH['vs_strong_pct'])})"))
        w(line("Vs weak opps:", f"{oppH['vs_weak_record']} ({fmt_pct(oppH['vs_weak_pct'])})"))

        w("</div></div>")
        w("<div class='col'><div class='details-block'>")
        
        # RIGHT column items go below:

        w(line("Recent form:", f"Last 5: {fmt_pct(formH['last5_pct'])}"))
        w(line("", f"Last 10: {fmt_pct(formH['last10_pct'])}"))
        w(line("", f"Streak: {fmt_streak(formH['streak'])}"))

        # === Insert Form Index at bottom ===
        if pd.isna(home_form_score):
            w(line("Form index:", "— (insufficient sample)"))
        else:
            w(line("Form index:", f"{home_form_score:0.0f} ({home_form_desc})"))

        # === NEW: Insert Mismatch Index right after Form Index ===
        if pd.isna(home_mis_score):
            w(line("Mismatch index:", "— (insufficient sample)"))
        else:
            w(line("Mismatch index:", f"{home_mis_score:+0.0f} ({home_mis_desc})"))

        # === PRM: Performance Relative to Market (full block) ===
        if pd.isna(home_prm_score):
            w(line("PRM last 10:", "— (insufficient sample)"))
        else:
            w(line("PRM last 10:", f"{home_prm_score:+0.1f}u"))
            w(line("Mispricing trend:", f"undervalued in {home_underval} of last 10"))
            w(line("Market signal:", home_prm_signal))

        # ---- Schedule Stress (home) ----
        if pd.isna(home_ss["score"]):
            w(line("Schedule stress:", "— (insufficient sample)"))
        else:
            w(line("Schedule stress:", f"{home_ss['score']:0.0f} ({home_ss['desc']})"))

            # Rest line
            if home_ss["is_b2b"]:
                rest_text = "<span class='pill pill-red'>B2B</span>"
            else:
                rest_text = f"{home_ss['rest_days']} days"
            w(line("Rest:", rest_text))

            # Games last 7 days
            w(line("Games last 7 days:", str(home_ss["games_7"])))

            # Travel miles
            if np.isnan(home_ss["travel_miles"]):
                travel_text = "n/a"
            else:
                travel_text = f"{home_ss['travel_miles']:0.0f} miles"
            w(line("Travel:", travel_text))

            # Road stretch
            # Only show road stretch if meaningful
            if home_ss["road_streak"] >= 1:
                if home_ss["road_streak"] == 1:
                    w(line("Road stretch:", "1 straight road game"))
                else:
                    w(line("Road stretch:", f"{home_ss['road_streak']} straight road games"))

            # Homestand indicator (only show if meaningful)
            if home_ss["home_streak"] >= 2:
                if home_ss["home_streak"] == 2:
                    w(line("Home stand:", "2 straight home games"))
                else:
                    w(line("Home stand:", f"{home_ss['home_streak']} straight home games"))
            
            # --- Injury Report (home) ---
            if home_inj and isinstance(home_inj, str) and home_inj.strip():
                parts = [x.strip() for x in home_inj.split(";") if x.strip()]
                for i, part in enumerate(parts):
                    w(line("Injuries:" if i == 0 else "", part))

        w("</div></div>")
        w("</div>")   # end two-col
        w("</div>")   # end team-card

        # ---------------- AWAY TEAM ----------------
        w("<div class='team-card'>")
        w(f"<div class='subheader'>{away.upper()} (AWAY)</div>")
        w("<div class='two-col'>")
        w("<div class='col'><div class='details-block'>")
        
        # LEFT column items go below:

        w(line("Record:", f"{away_sum['ml_record']} ({fmt_pct(away_sum['ml_win_pct'])})"))
        w(line("As favorite:", f"{away_sum['fav_record']} ({fmt_pct(away_sum['fav_win_pct'])})", highlight=away_is_fav))
        w(line("As underdog:", f"{away_sum['dog_record']} ({fmt_pct(away_sum['dog_win_pct'])})", highlight=away_is_dog))
        w(line("Home:", f"{away_sum['home_record']} ({fmt_pct(away_sum['home_win_pct'])})"))
        w(line("Away:", f"{away_sum['away_record']} ({fmt_pct(away_sum['away_win_pct'])})", highlight=True))
        w(line("As away favorite:", f"{away_sum['away_fav_record']} ({fmt_pct(away_sum['away_fav_win_pct'])})", highlight=away_is_fav))
        w(line("As away underdog:", f"{away_sum['away_dog_record']} ({fmt_pct(away_sum['away_dog_win_pct'])})", highlight=away_is_dog))

        if bucket_away:
            b=bucket_away_stats
            w(line(f"{bucket_away}:", f"{b['bucket_record']} ({fmt_pct(b['bucket_win_pct'])})"))

        w(line("ROI all ML bets:", f"{away_sum['roi_units']:+0.1f}u ({fmt_pct(away_sum['roi_pct'])})"))
        w(line("ROI as favorite:", f"{away_sum['fav_roi_units']:+0.1f}u ({fmt_pct(away_sum['fav_roi_pct'])})"))
        w(line("ROI as underdog:", f"{away_sum['dog_roi_units']:+0.1f}u ({fmt_pct(away_sum['dog_roi_pct'])})"))

        w(line("Vs strong opps:", f"{oppA['vs_strong_record']} ({fmt_pct(oppA['vs_strong_pct'])})"))
        w(line("Vs weak opps:", f"{oppA['vs_weak_record']} ({fmt_pct(oppA['vs_weak_pct'])})"))

        w("</div></div>")
        w("<div class='col'><div class='details-block'>")
        
        # RIGHT column items go below:

        w(line("Recent form:", f"Last 5: {fmt_pct(formA['last5_pct'])}"))
        w(line("", f"Last 10: {fmt_pct(formA['last10_pct'])}"))
        w(line("", f"Streak: {fmt_streak(formA['streak'])}"))

        # === Form Index (away) ===
        if pd.isna(away_form_score):
            w(line("Form index:", "— (insufficient sample)"))
        else:
            w(line("Form index:", f"{away_form_score:0.0f} ({away_form_desc})"))

        # === Mismatch Index (away) ===
        if pd.isna(away_mis_score):
            w(line("Mismatch index:", "— (insufficient sample)"))
        else:
            w(line("Mismatch index:", f"{away_mis_score:+0.0f} ({away_mis_desc})"))

        # === PRM: Performance Relative to Market (full block) ===
        if pd.isna(away_prm_score):
            w(line("PRM last 10:", "— (insufficient sample)"))
        else:
            w(line("PRM last 10:", f"{away_prm_score:+0.1f}u"))
            w(line("Mispricing trend:", f"undervalued in {away_underval} of last 10"))
            w(line("Market signal:", away_prm_signal))

        # ---- Schedule Stress (away) ----
        if pd.isna(away_ss["score"]):
            w(line("Schedule stress:", "— (insufficient sample)"))
        else:
            w(line("Schedule stress:", f"{away_ss['score']:0.0f} ({away_ss['desc']})"))

            if away_ss["is_b2b"]:
                rest_text = "<span class='pill pill-red'>B2B</span>"
            else:
                rest_text = f"{away_ss['rest_days']} days"
            w(line("Rest:", rest_text))

            w(line("Games last 7 days:", str(away_ss["games_7"])))

            if np.isnan(away_ss["travel_miles"]):
                travel_text = "n/a"
            else:
                travel_text = f"{away_ss['travel_miles']:0.0f} miles"
            w(line("Travel:", travel_text))

            # Only show road stretch if meaningful
            if away_ss["road_streak"] >= 1:
                if away_ss["road_streak"] == 1:
                    w(line("Road stretch:", "1 straight road game"))
                else:
                    w(line("Road stretch:", f"{away_ss['road_streak']} straight road games"))

            if away_ss["home_streak"] >= 2:
                if away_ss["home_streak"] == 2:
                    w(line("Home stand:", "2 straight home games"))
                else:
                    w(line("Home stand:", f"{away_ss['home_streak']} straight home games"))

        # --- Injury Report (away) ---
        if away_inj and isinstance(away_inj, str) and away_inj.strip():
            parts = [x.strip() for x in away_inj.split(";") if x.strip()]
            for i, part in enumerate(parts):
                w(line("Injuries:" if i == 0 else "", part))

        w("</div></div>")
        w("</div>")   # end two-col
        w("</div>")   # end team-card
        w("</details>")
        w("</div>")  # end game-card

    # --------- League overview ---------
    if not league_tbl.empty:
        w("<div class='league-block'>"); w("<h2>League Overview</h2>")
        lv=league_tbl
        def block(title, df, col):
            w(f"<h3>{title}</h3>")
            w("<div class='details-block league-list'>")
            for i, row in enumerate(df.itertuples(index=False), start=1):
                w(line(f"{i}.", f"{row.team} {fmt_pct(getattr(row, col))}"))
            w("</div>")
        block("Best overall Win %", lv.sort_values("ml_pct",ascending=False).head(10), "ml_pct")
        block("Best Home Teams", lv.sort_values("home_pct",ascending=False).head(10), "home_pct")
        block("Best Away Teams", lv.sort_values("away_pct",ascending=False).head(10), "away_pct")
        block("Best Favorites", lv.sort_values("fav_pct",ascending=False).head(10), "fav_pct")
        block("Best Underdogs", lv.sort_values("dog_pct",ascending=False).head(10), "dog_pct")
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
    write_injury_files(slate, today)
    outfile = f"dashboard_{today}.html"
    build_html(slate, team_results, league_tbl, outfile)
    print("Dashboard written →", outfile)

if __name__ == "__main__":
    main()
