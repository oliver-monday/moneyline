#!/usr/bin/env python3
"""
Build a threshold-focused snapshot from player_game_log.csv for the Players (props) dashboard.

Per player, per window (L5/L10/L20) over *games played* (minutes > 0):
  - Minutes: avg + min (role stability)
  - True floor: min PTS/REB/AST
  - Threshold hit counts + rates (Kalshi-style lines):
      PTS: 10,15,20,25,30,35
      REB/AST: 2,4,6,8,10,12,14

DNP rows are tracked separately (dnp_rows) but not included in rolling windows.

Usage:
  python build_player_snapshot.py --in player_game_log.csv --out player_snapshot.csv --season-end-year 2026
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def season_end_year_for_date(d: dt.date) -> int:
    return d.year + 1 if d.month >= 10 else d.year


def normalize_name(s: str) -> str:
    return " ".join(
        str(s or "")
        .lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace(".", " ")
        .replace(",", " ")
        .split()
    )


def safe_min(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.min())


def rate_and_hits_ge(x: pd.Series, thresh: float) -> Tuple[int, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return 0, float("nan")
    hits = int((x >= thresh).sum())
    return hits, float(hits / len(x))


def iqr(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    q75 = np.nanpercentile(x, 75)
    q25 = np.nanpercentile(x, 25)
    return float(q75 - q25)


def trimmed_mean(x: pd.Series, trim: float = 0.1) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    vals = x.values
    n = len(vals)
    if n < 10:
        return float(np.mean(vals))
    k = int(np.floor(n * trim))
    if k == 0:
        return float(np.mean(vals))
    vals = np.sort(vals)
    if n > 2 * k:
        vals = vals[k:-k]
    return float(np.mean(vals)) if len(vals) else float("nan")


def consistency_score(iqr_val: float, scale: float) -> float:
    if not np.isfinite(iqr_val) or not scale:
        return float("nan")
    ratio = min(iqr_val / scale, 2.0)
    score = 100.0 * (1.0 - (ratio / 2.0))
    return float(np.clip(score, 0, 100))


def consistency_tier(score: float) -> str:
    if not np.isfinite(score):
        return ""
    if score < 50:
        return "Inconsistent"
    if score <= 64:
        return "Consistency OK"
    if score <= 72:
        return "Solid Consistency"
    if score <= 78:
        return "Consistent"
    if score <= 84:
        return "Very Consistent"
    return "Elite Consistency"


def core_avg(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return float("nan")
    med = float(np.nanmedian(series))
    tmean = trimmed_mean(series, trim=0.1)
    if np.isfinite(med) and np.isfinite(tmean):
        return float(0.5 * med + 0.5 * tmean)
    return float("nan")


def normalize_game_id(val) -> str:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


SPLIT_EPS = 0.5


def split_core_stats(df: pd.DataFrame, stat_col: str) -> Tuple[float, float, int, int]:
    if df is None or df.empty:
        return float("nan"), float("nan"), 0, 0
    home = df[df["home_away"] == "H"]
    away = df[df["home_away"] == "A"]
    core_home = core_avg(home[stat_col]) if len(home) else float("nan")
    core_away = core_avg(away[stat_col]) if len(away) else float("nan")
    return core_home, core_away, int(len(home)), int(len(away))


CORE_TREND_EPS = 0.5
CORE_TREND_Z_THRESH = 0.60
CONS_TREND_DELTA_THRESH = 6.0


def trend_dir_from_z(z: float, thresh: float = CORE_TREND_Z_THRESH) -> str:
    if not np.isfinite(z):
        return "na"
    if z >= thresh:
        return "up"
    if z <= -thresh:
        return "down"
    return "flat"


def trend_dir_from_delta(delta: float, thresh: float = CONS_TREND_DELTA_THRESH) -> str:
    if not np.isfinite(delta):
        return "na"
    if delta >= thresh:
        return "up"
    if delta <= -thresh:
        return "down"
    return "flat"


def trend_summary_dir(a: str, b: str) -> str:
    if a == "na" or b == "na":
        return "na"
    if a == b:
        return a
    return "flat"


def core_trend(core_10, core_base, iqr_base, gate_ok: bool) -> Tuple[float, str]:
    if not gate_ok:
        return float("nan"), "na"
    if not (np.isfinite(core_10) and np.isfinite(core_base) and np.isfinite(iqr_base)):
        return float("nan"), "na"
    denom = max(float(iqr_base), CORE_TREND_EPS)
    z = float(core_10 - core_base) / denom
    return z, trend_dir_from_z(z)


def cons_trend(cons_10, cons_base, gate_ok: bool) -> Tuple[float, str]:
    if not gate_ok:
        return float("nan"), "na"
    if not (np.isfinite(cons_10) and np.isfinite(cons_base)):
        return float("nan"), "na"
    delta = float(cons_10 - cons_base)
    return delta, trend_dir_from_delta(delta)


def build_windows(df_player: pd.DataFrame, windows: List[int], thresholds: Dict[str, List[float]]) -> Dict[str, float]:
    """df_player sorted desc by game_date and filtered to games played (minutes>0)."""
    stat_col = {"pts": "pts", "reb": "reb", "ast": "ast", "3pt": "tpm"}
    out: Dict[str, float] = {}
    for w in windows:
        d = df_player.head(w)
        gp = int(len(d))
        out[f"gp_{w}"] = gp

        out[f"min_avg_{w}"] = float(d["minutes"].mean()) if gp else float("nan")
        out[f"min_min_{w}"] = safe_min(d["minutes"]) if gp else float("nan")

        # Minutes thresholds (role reliability)
        MIN_T = [24, 28, 30, 32]
        for t in MIN_T:
            tkey = str(int(t))
            if gp:
                hits, rate = rate_and_hits_ge(d["minutes"], t)
            else:
                hits, rate = 0, float("nan")
            out[f"min_ge{tkey}_hits_{w}"] = hits
            out[f"min_ge{tkey}_rate_{w}"] = rate

        for stat in ("pts", "reb", "ast", "3pt"):
            col = stat_col.get(stat, stat)
            out[f"{stat}_min_{w}"] = safe_min(d[col]) if gp else float("nan")

            for t in thresholds.get(stat, []):
                tkey = str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
                if gp:
                    hits, rate = rate_and_hits_ge(d[col], t)
                else:
                    hits, rate = 0, float("nan")
                out[f"{stat}_ge{tkey}_hits_{w}"] = hits
                out[f"{stat}_ge{tkey}_rate_{w}"] = rate

    return out


def build_load_metrics(df_player: pd.DataFrame, windows: List[int]) -> Dict[str, float]:
    """Compute load deltas/labels and recent travel string for a player."""
    out: Dict[str, float] = {}
    if df_player.empty:
        for w in windows:
            out[f"load_min_delta_{w}"] = float("nan")
            out[f"load_pts_delta_{w}"] = float("nan")
            out[f"load_score_{w}"] = float("nan")
            out[f"load_label_{w}"] = ""
        out["travel_last3"] = ""
        return out

    season_min_avg = float(df_player["minutes"].mean())
    season_pts_avg = float(df_player["pts"].mean())

    last3 = df_player.head(3)
    if len(last3) == 3:
        travel_map = {"A": "@", "H": "vs"}
        travel_last3 = " ".join(travel_map.get(str(x), "") for x in last3["home_away"])
    else:
        travel_last3 = ""
    out["travel_last3"] = travel_last3.strip()

    for w in windows:
        d = df_player.head(w)
        gp = int(len(d))
        if gp and np.isfinite(season_min_avg) and np.isfinite(season_pts_avg):
            win_min_avg = float(d["minutes"].mean())
            win_pts_avg = float(d["pts"].mean())
            min_delta = win_min_avg - season_min_avg
            pts_delta = win_pts_avg - season_pts_avg
            score = 0.7 * min_delta + 0.3 * pts_delta
            if score >= 3.0:
                label = "HIGH"
            elif score >= 1.5:
                label = "MED"
            else:
                label = "Normal"
        else:
            min_delta = float("nan")
            pts_delta = float("nan")
            score = float("nan")
            label = ""
        out[f"load_min_delta_{w}"] = min_delta
        out[f"load_pts_delta_{w}"] = pts_delta
        out[f"load_score_{w}"] = score
        out[f"load_label_{w}"] = label

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="player_game_log.csv", help="Input player_game_log.csv")
    ap.add_argument("--out", dest="out", default="player_snapshot.csv", help="Output snapshot CSV (recommended: docs/data/player_snapshot.csv)")
    ap.add_argument("--season-end-year", type=int, default=None, help="Filter season_end_year (e.g., 2026)")
    ap.add_argument("--asof", default=None, help="Only include games with game_date <= asof (YYYY-MM-DD). Default: today")
    ap.add_argument("--master", default=None, help="Optional nba_master.csv to normalize game_date by game_id")
    ap.add_argument("--windows", default="5,10,20", help="Comma-separated rolling windows (games played)")
    ap.add_argument("--whitelist", default="playerprops/player_whitelist.csv", help="Optional whitelist CSV to filter active players")
    ap.add_argument("--injuries", default="data/injuries_today.json", help="Optional injuries JSON for auto-inactive OFS filtering")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if args.master:
        m = pd.read_csv(args.master, dtype=str)
        if "game_id" in m.columns and "game_date" in m.columns:
            gid_to_date = dict(zip(m["game_id"], m["game_date"]))
            df["game_date"] = df["game_id"].astype(str).map(gid_to_date).fillna(df["game_date"].astype(str))

    required = {"season_end_year","game_id","game_date","team_abbrev","opp_abbrev","home_away","player_id","player_name","minutes","pts","reb","ast","dnp"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {args.inp}: {sorted(missing)}")


    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df = df[df["game_date"].notna()].copy()

    asof = dt.date.today() if not args.asof else dt.datetime.strptime(args.asof, "%Y-%m-%d").date()
    df = df[df["game_date"] <= asof].copy()

    season_end = args.season_end_year or season_end_year_for_date(asof)
    df["game_id"] = df["game_id"].map(normalize_game_id)
    df["team_abbrev"] = df["team_abbrev"].astype(str).str.upper().str.strip()
    df["season_end_year"] = pd.to_numeric(df["season_end_year"], errors="coerce").fillna(0).astype(int)
    df = df[df["season_end_year"] == int(season_end)].copy()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    windows = sorted(list(dict.fromkeys(windows)))

    thresholds = {
        "pts": [10, 15, 20, 25, 30, 35],
        "reb": [2, 4, 6, 8, 10, 12, 14],
        "ast": [2, 4, 6, 8, 10, 12, 14],
        "3pt": [1, 2, 3, 4, 5, 6],
    }

    if "tpm" not in df.columns:
        df["tpm"] = 0
    for c in ["minutes","pts","reb","ast","tpm","dnp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df_played = df[(df["minutes"] > 0) & (df["dnp"] != 1)].copy()
    team_log_path = os.path.join("data", "team_game_log.csv")
    team_totals = None
    if os.path.exists(team_log_path):
        team_totals = pd.read_csv(team_log_path)
        for col in ["season_end_year","game_id","team_abbrev"]:
            if col not in team_totals.columns:
                team_totals[col] = ""
        team_totals["season_end_year"] = pd.to_numeric(team_totals["season_end_year"], errors="coerce").fillna(0).astype(int)
        team_totals["game_id"] = team_totals["game_id"].map(normalize_game_id)
        team_totals["team_abbrev"] = team_totals["team_abbrev"].astype(str).str.upper().str.strip()
        keep_cols = ["season_end_year","game_id","team_abbrev","team_pts","team_reb","team_ast","team_tpm"]
        team_totals = team_totals[keep_cols].copy()
    if team_totals is not None and not team_totals.empty:
        df_played = df_played.merge(team_totals, on=["season_end_year","game_id","team_abbrev"], how="left")
        missing_team_totals = int(df_played[["team_pts","team_reb","team_ast"]].isna().any(axis=1).sum())
        if missing_team_totals:
            print(f"[share] missing_team_totals_rows={missing_team_totals}")
    else:
        df_played["team_pts"] = float("nan")
        df_played["team_reb"] = float("nan")
        df_played["team_ast"] = float("nan")
        print("[share] missing_team_totals_rows=all")
    for stat in ("pts", "reb", "ast"):
        total_col = f"team_{stat}"
        share_col = f"share_{stat}"
        df_played[share_col] = np.where(
            df_played[total_col] > 0,
            np.clip(df_played[stat] / df_played[total_col], 0, 1),
            np.nan,
        )

    rows = []
    for (pid, pname), g in df.groupby(["player_id","player_name"], sort=False):
        g = g.sort_values("game_date", ascending=False)

        gp = g[(g["minutes"] > 0) & (g["dnp"] != 1)].copy().sort_values("game_date", ascending=False)
        gp_feat = df_played[(df_played["player_id"] == pid) & (df_played["player_name"] == pname)].copy()
        gp_feat = gp_feat.sort_values("game_date", ascending=False)
        last_row = gp.iloc[0] if len(gp) else g.iloc[0]
        base = {
            "season_end_year": int(season_end),
            "player_id": str(pid),
            "player_name": str(pname),
            "last_game_date": str(last_row["game_date"]),
            "last_team_abbrev": str(last_row["team_abbrev"]).upper(),
            "last_opp_abbrev": str(last_row["opp_abbrev"]).upper(),
            "last_home_away": str(last_row["home_away"]),
            "last_pts": float(last_row["pts"]) if len(gp) else float("nan"),
            "last_reb": float(last_row["reb"]) if len(gp) else float("nan"),
            "last_ast": float(last_row["ast"]) if len(gp) else float("nan"),
            "last_min": float(last_row["minutes"]) if len(gp) else float("nan"),
            "season_avg_pts": float(gp["pts"].mean()) if len(gp) else float("nan"),
            "season_avg_reb": float(gp["reb"].mean()) if len(gp) else float("nan"),
            "season_avg_ast": float(gp["ast"].mean()) if len(gp) else float("nan"),
            "season_avg_3pt": float(gp["tpm"].mean()) if len(gp) else float("nan"),
            "pts_avg_season": float(gp["pts"].mean()) if len(gp) else float("nan"),
            "reb_avg_season": float(gp["reb"].mean()) if len(gp) else float("nan"),
            "ast_avg_season": float(gp["ast"].mean()) if len(gp) else float("nan"),
            "3pt_avg_season": float(gp["tpm"].mean()) if len(gp) else float("nan"),
            "games_total_rows": int(len(g)),
            "games_played_rows": int((g["minutes"] > 0).sum()),
            "dnp_rows": int((g["dnp"] == 1).sum()),
        }

        # Volatility (IQR) over last 10 games played
        last10 = gp.head(10)
        base["pts_iqr_10"] = iqr(last10["pts"]) if len(last10) else float("nan")
        base["reb_iqr_10"] = iqr(last10["reb"]) if len(last10) else float("nan")
        base["ast_iqr_10"] = iqr(last10["ast"]) if len(last10) else float("nan")
        base["3pt_iqr_10"] = iqr(last10["tpm"]) if len(last10) else float("nan")

        # Multi-window core averages + volatility
        last20 = gp.head(20)
        season_gp = gp
        for stat in ("pts", "reb", "ast"):
            base[f"{stat}_core_10"] = core_avg(last10[stat]) if len(last10) else float("nan")
            base[f"{stat}_core_20"] = core_avg(last20[stat]) if len(last20) else float("nan")
            base[f"{stat}_core_season"] = core_avg(season_gp[stat]) if len(season_gp) else float("nan")
            base[f"core_{stat}_10"] = base[f"{stat}_core_10"]
            base[f"core_{stat}_20"] = base[f"{stat}_core_20"]
            base[f"core_{stat}_season"] = base[f"{stat}_core_season"]
            base[f"{stat}_iqr_20"] = iqr(last20[stat]) if len(last20) else float("nan")
            base[f"{stat}_iqr_season"] = iqr(season_gp[stat]) if len(season_gp) else float("nan")

        base["3pt_core_10"] = core_avg(last10["tpm"]) if len(last10) else float("nan")
        base["3pt_core_20"] = core_avg(last20["tpm"]) if len(last20) else float("nan")
        base["3pt_core_season"] = core_avg(season_gp["tpm"]) if len(season_gp) else float("nan")
        base["core_3pt_10"] = base["3pt_core_10"]
        base["core_3pt_20"] = base["3pt_core_20"]
        base["core_3pt_season"] = base["3pt_core_season"]
        base["3pt_iqr_20"] = iqr(last20["tpm"]) if len(last20) else float("nan")
        base["3pt_iqr_season"] = iqr(season_gp["tpm"]) if len(season_gp) else float("nan")

        base["tpm_mean_10"] = float(last10["tpm"].mean()) if len(last10) else float("nan")
        base["tpm_mean_20"] = float(last20["tpm"].mean()) if len(last20) else float("nan")
        base["tpm_mean_season"] = float(season_gp["tpm"].mean()) if len(season_gp) else float("nan")
        base["tpm_nonzero_rate_10"] = float((last10["tpm"] > 0).mean()) if len(last10) else float("nan")
        base["tpm_nonzero_rate_20"] = float((last20["tpm"] > 0).mean()) if len(last20) else float("nan")
        base["tpm_nonzero_rate_season"] = float((season_gp["tpm"] > 0).mean()) if len(season_gp) else float("nan")

        gp10 = int(len(last10))
        gp20 = int(len(last20))
        gpSeason = int(len(season_gp))
        base["threept_eligible_10"] = bool(
            gp10 >= 7 and (
                (np.isfinite(base["tpm_mean_10"]) and base["tpm_mean_10"] >= 1.0) or
                (np.isfinite(base["tpm_nonzero_rate_10"]) and base["tpm_nonzero_rate_10"] >= 0.60)
            )
        )
        base["threept_eligible_20"] = bool(
            gp20 >= 14 and (
                (np.isfinite(base["tpm_mean_20"]) and base["tpm_mean_20"] >= 1.0) or
                (np.isfinite(base["tpm_nonzero_rate_20"]) and base["tpm_nonzero_rate_20"] >= 0.60)
            )
        )
        base["threept_eligible_season"] = bool(
            gpSeason >= 25 and (
                (np.isfinite(base["tpm_mean_season"]) and base["tpm_mean_season"] >= 0.9) or
                (np.isfinite(base["tpm_nonzero_rate_season"]) and base["tpm_nonzero_rate_season"] >= 0.55)
            )
        )

        split_windows = {"10": last10, "20": last20, "season": season_gp}
        for win, df_win in split_windows.items():
            if win == "10":
                gp_ok = gp10 >= 7
            elif win == "20":
                gp_ok = gp20 >= 14
            else:
                gp_ok = gpSeason >= 25
            core_home, core_away, gp_home, gp_away = split_core_stats(df_win, "pts")
            base[f"gp_home_{win}"] = gp_home
            base[f"gp_away_{win}"] = gp_away
            base[f"split_usable_{win}"] = bool(gp_ok and gp_home >= 4 and gp_away >= 4)
            base[f"split_usable_3pt_{win}"] = bool(base[f"split_usable_{win}"] and base.get(f"threept_eligible_{win}"))
            for stat, col in (("pts", "pts"), ("reb", "reb"), ("ast", "ast"), ("3pt", "tpm")):
                home_core, away_core, _, _ = split_core_stats(df_win, col)
                base[f"core_home_{stat}_{win}"] = home_core
                base[f"core_away_{stat}_{win}"] = away_core
                delta = float(home_core - away_core) if np.isfinite(home_core) and np.isfinite(away_core) else float("nan")
                base[f"split_core_delta_{stat}_{win}"] = delta
                iqr_key = f"{stat}_iqr_{win}" if win != "season" else f"{stat}_iqr_season"
                iqr_val = base.get(iqr_key)
                if not np.isfinite(iqr_val):
                    iqr_val = base.get(f"{stat}_iqr_season")
                denom = max(float(iqr_val), SPLIT_EPS) if np.isfinite(iqr_val) else SPLIT_EPS
                base[f"split_core_z_{stat}_{win}"] = delta / denom if np.isfinite(delta) else float("nan")

        # Minutes stability + DNP rate
        base["minutes_iqr_10"] = iqr(last10["minutes"]) if len(last10) else float("nan")
        base["minutes_iqr_20"] = iqr(last20["minutes"]) if len(last20) else float("nan")
        base["minutes_iqr_season"] = iqr(season_gp["minutes"]) if len(season_gp) else float("nan")
        last10_all = g.head(10)
        last20_all = g.head(20)
        base["dnp_rate_10"] = float(((last10_all["dnp"] == 1) | (last10_all["minutes"] == 0)).mean()) if len(last10_all) else float("nan")
        base["dnp_rate_20"] = float(((last20_all["dnp"] == 1) | (last20_all["minutes"] == 0)).mean()) if len(last20_all) else float("nan")
        base["dnp_rate_season"] = float(((g["dnp"] == 1) | (g["minutes"] == 0)).mean()) if len(g) else float("nan")

        # Team share trends (last 5 vs last 20)
        share_l5 = gp_feat.head(5)
        share_l20 = gp_feat.head(20)
        for stat in ("pts", "reb", "ast"):
            col = f"share_{stat}"
            l5 = float(share_l5[col].mean()) if len(share_l5) else float("nan")
            l20 = float(share_l20[col].mean()) if len(share_l20) else float("nan")
            base[f"share_{stat}_l5"] = l5
            base[f"share_{stat}_l20"] = l20
            base[f"share_{stat}_trend_eligible"] = bool(len(share_l5) >= 4 and len(share_l20) >= 12)
            if np.isfinite(l5) and np.isfinite(l20):
                base[f"share_{stat}_trend_pp"] = (l5 - l20) * 100.0
            else:
                base[f"share_{stat}_trend_pp"] = float("nan")

        # Landmine rates (actual == threshold - 1) over last 10/20/season games
        for stat, thresholds_list in thresholds.items():
            if stat == "3pt":
                continue
            for t in thresholds_list:
                tkey = str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
                landmine10 = (last10[stat] == (t - 1)).mean() if len(last10) else float("nan")
                landmine20 = (last20[stat] == (t - 1)).mean() if len(last20) else float("nan")
                landmine_season = (season_gp[stat] == (t - 1)).mean() if len(season_gp) else float("nan")
                base[f"{stat}_landmine_{tkey}_10"] = float(landmine10) if np.isfinite(landmine10) else float("nan")
                base[f"{stat}_landmine_{tkey}_20"] = float(landmine20) if np.isfinite(landmine20) else float("nan")
                base[f"{stat}_landmine_{tkey}_season"] = float(landmine_season) if np.isfinite(landmine_season) else float("nan")

        # Consistency scores (0-100) for 10/20/season
        def window_consistency(tag: str, pts_iqr_val, reb_iqr_val, ast_iqr_val, min_iqr_val, dnp_rate_val, landmine_avg):
            parts = []
            for val, scale in ((pts_iqr_val, 10), (reb_iqr_val, 5), (ast_iqr_val, 4)):
                if np.isfinite(val):
                    parts.append(min(val / scale, 2.0) / 2.0)
            vol_norm = float(np.mean(parts)) if parts else float("nan")
            min_norm = min(min_iqr_val / 8.0, 2.0) / 2.0 if np.isfinite(min_iqr_val) else float("nan")
            dnp_norm = dnp_rate_val if np.isfinite(dnp_rate_val) else float("nan")
            land_norm = landmine_avg if np.isfinite(landmine_avg) else float("nan")
            weights = [0.4, 0.2, 0.2, 0.2]
            comps = [vol_norm, min_norm, dnp_norm, land_norm]
            total = 0.0
            wsum = 0.0
            for w, c in zip(weights, comps):
                if np.isfinite(c):
                    total += w * c
                    wsum += w
            if wsum == 0:
                return float("nan")
            score = 100.0 - float(np.clip(total / wsum * 100.0, 0, 100))
            return score

        def landmine_avg_for(tag: str):
            vals = []
            for stat, thresholds_list in thresholds.items():
                if stat == "3pt":
                    continue
                for t in thresholds_list:
                    tkey = str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
                    key = f"{stat}_landmine_{tkey}_{tag}"
                    v = base.get(key)
                    if np.isfinite(v):
                        vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")

        landmine_avg_10 = landmine_avg_for("10")
        landmine_avg_20 = landmine_avg_for("20")
        landmine_avg_season = landmine_avg_for("season")

        base["consistency_10"] = window_consistency(
            "10",
            base.get("pts_iqr_10"),
            base.get("reb_iqr_10"),
            base.get("ast_iqr_10"),
            base.get("minutes_iqr_10"),
            base.get("dnp_rate_10"),
            landmine_avg_10,
        )
        base["consistency_20"] = window_consistency(
            "20",
            base.get("pts_iqr_20"),
            base.get("reb_iqr_20"),
            base.get("ast_iqr_20"),
            base.get("minutes_iqr_20"),
            base.get("dnp_rate_20"),
            landmine_avg_20,
        )
        base["consistency_season"] = window_consistency(
            "season",
            base.get("pts_iqr_season"),
            base.get("reb_iqr_season"),
            base.get("ast_iqr_season"),
            base.get("minutes_iqr_season"),
            base.get("dnp_rate_season"),
            landmine_avg_season,
        )
        base["consistency_score"] = base.get("consistency_10")
        base["consistency_3pt_10"] = consistency_score(base.get("3pt_iqr_10"), 2.0)
        base["consistency_3pt_20"] = consistency_score(base.get("3pt_iqr_20"), 2.0)
        base["consistency_3pt_season"] = consistency_score(base.get("3pt_iqr_season"), 2.0)
        base["consistency_tier_3pt_10"] = consistency_tier(base.get("consistency_3pt_10"))
        base["consistency_tier_3pt_20"] = consistency_tier(base.get("consistency_3pt_20"))
        base["consistency_tier_3pt_season"] = consistency_tier(base.get("consistency_3pt_season"))
        base["consistency_tier_10"] = consistency_tier(base.get("consistency_10"))
        base["consistency_tier_20"] = consistency_tier(base.get("consistency_20"))
        base["consistency_tier_season"] = consistency_tier(base.get("consistency_season"))
        base["consistency_tier"] = base.get("consistency_tier_10")

        gp_ok_10_20 = gp10 >= 7 and gp20 >= 14
        gp_ok_10_season = gp10 >= 7 and gpSeason >= 25
        for stat in ("pts", "reb", "ast", "3pt"):
            core10 = base.get(f"{stat}_core_10")
            core20 = base.get(f"{stat}_core_20")
            coreSeason = base.get(f"{stat}_core_season")
            iqr20 = base.get(f"{stat}_iqr_20")
            iqrSeason = base.get(f"{stat}_iqr_season")
            gate_10_20 = gp_ok_10_20
            gate_10_season = gp_ok_10_season
            if stat == "3pt":
                if not (base.get("threept_eligible_10") and base.get("threept_eligible_20")):
                    gate_10_20 = False
                if not (base.get("threept_eligible_10") and base.get("threept_eligible_season")):
                    gate_10_season = False

            z_10_20, dir_10_20 = core_trend(core10, core20, iqr20, gate_10_20)
            z_10_season, dir_10_season = core_trend(core10, coreSeason, iqrSeason, gate_10_season)
            base[f"trend_core_z_{stat}_10_20"] = z_10_20
            base[f"trend_core_z_{stat}_10_season"] = z_10_season
            base[f"trend_core_dir_{stat}_10_20"] = dir_10_20
            base[f"trend_core_dir_{stat}_10_season"] = dir_10_season
            base[f"trend_core_summary_dir_{stat}"] = trend_summary_dir(dir_10_20, dir_10_season)

            if stat == "3pt":
                cons10 = base.get("consistency_3pt_10")
                cons20 = base.get("consistency_3pt_20")
                consSeason = base.get("consistency_3pt_season")
            else:
                cons10 = base.get("consistency_10")
                cons20 = base.get("consistency_20")
                consSeason = base.get("consistency_season")
            d_10_20, cons_dir_10_20 = cons_trend(cons10, cons20, gate_10_20)
            d_10_season, cons_dir_10_season = cons_trend(cons10, consSeason, gate_10_season)
            base[f"trend_cons_delta_{stat}_10_20"] = d_10_20
            base[f"trend_cons_delta_{stat}_10_season"] = d_10_season
            base[f"trend_cons_dir_{stat}_10_20"] = cons_dir_10_20
            base[f"trend_cons_dir_{stat}_10_season"] = cons_dir_10_season
            base[f"trend_cons_summary_dir_{stat}"] = trend_summary_dir(cons_dir_10_20, cons_dir_10_season)

        base.update(build_windows(gp, windows, thresholds))
        base.update(build_load_metrics(gp, windows))
        rows.append(base)

    out = pd.DataFrame(rows)

    if "min_avg_10" in out.columns and "pts_ge20_rate_10" in out.columns:
        out = out.sort_values(["min_avg_10", "pts_ge20_rate_10"], ascending=False)

    meta_cols = [
        "season_end_year","player_id","player_name",
        "last_game_date","last_team_abbrev","last_opp_abbrev","last_home_away",
        "games_total_rows","games_played_rows","dnp_rows"
    ]
    other_cols = [c for c in out.columns if c not in meta_cols]
    out = out[meta_cols + sorted(other_cols)]

    ofs_names = set()
    if args.injuries and os.path.exists(args.injuries):
        try:
            with open(args.injuries, "r", encoding="utf-8") as f:
                injuries = json.load(f) or {}
        except Exception:
            injuries = {}
        for team, items in (injuries or {}).items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "")).strip().upper()
                details = str(item.get("details", "")).strip().lower()
                if status == "OFS" or "out for season" in details:
                    name = normalize_name(item.get("name", ""))
                    if name:
                        ofs_names.add(name)
        if ofs_names:
            try:
                with open("data/ofs_today.json", "w", encoding="utf-8") as f:
                    json.dump(sorted(ofs_names), f, indent=2)
            except Exception:
                pass

    if args.whitelist:
        try:
            wl = pd.read_csv(args.whitelist)
        except Exception:
            wl = None
        if wl is not None and "player_name" in wl.columns and "team_abbr" in wl.columns:
            wl = wl.copy()
            wl["player_name_norm"] = wl["player_name"].map(normalize_name)
            wl["team_abbr_norm"] = wl["team_abbr"].astype(str).str.upper().str.strip()
            if "active" not in wl.columns:
                wl["active"] = 1
            wl["active"] = pd.to_numeric(wl["active"], errors="coerce").fillna(1).astype(int)
            out = out.copy()
            out["player_name_norm"] = out["player_name"].map(normalize_name)
            out["team_abbr_norm"] = out["last_team_abbrev"].astype(str).str.upper().str.strip()
            out = out.merge(
                wl[["player_name_norm", "team_abbr_norm", "active"]],
                how="left",
                on=["player_name_norm", "team_abbr_norm"],
            )
            name_active = wl.drop_duplicates("player_name_norm")[["player_name_norm", "active"]]
            out = out.merge(name_active, how="left", on="player_name_norm", suffixes=("", "_name"))
            active = pd.to_numeric(out["active"], errors="coerce")
            active = active.fillna(pd.to_numeric(out["active_name"], errors="coerce")).fillna(1).astype(int)
            out["auto_inactive"] = out["player_name_norm"].isin(ofs_names)
            out = out[(active == 1) & (~out["auto_inactive"])].copy()
            out = out.drop(columns=["player_name_norm", "team_abbr_norm", "active", "active_name", "auto_inactive"], errors="ignore")

    features = {}
    asof_str = asof.strftime("%Y-%m-%d")
    tier_counts = {}
    threept_counts = {"eligible": 0, "ineligible": 0}
    for _, row in out.iterrows():
        pid = str(row.get("player_id", "")).strip()
        if not pid:
            continue
        tier = str(row.get("consistency_tier", "")).strip()
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if bool(row.get("threept_eligible_10")):
            threept_counts["eligible"] += 1
        else:
            threept_counts["ineligible"] += 1
        features[pid] = {
            "player_name": row.get("player_name", ""),
            "team_abbrev": row.get("last_team_abbrev", ""),
            "pts_iqr_10": row.get("pts_iqr_10"),
            "pts_iqr_20": row.get("pts_iqr_20"),
            "pts_iqr_season": row.get("pts_iqr_season"),
            "reb_iqr_10": row.get("reb_iqr_10"),
            "reb_iqr_20": row.get("reb_iqr_20"),
            "reb_iqr_season": row.get("reb_iqr_season"),
            "ast_iqr_10": row.get("ast_iqr_10"),
            "ast_iqr_20": row.get("ast_iqr_20"),
            "ast_iqr_season": row.get("ast_iqr_season"),
            "minutes_iqr_10": row.get("minutes_iqr_10"),
            "minutes_iqr_20": row.get("minutes_iqr_20"),
            "minutes_iqr_season": row.get("minutes_iqr_season"),
            "dnp_rate_10": row.get("dnp_rate_10"),
            "dnp_rate_20": row.get("dnp_rate_20"),
            "dnp_rate_season": row.get("dnp_rate_season"),
            "core_pts_20": row.get("core_pts_20"),
            "core_reb_20": row.get("core_reb_20"),
            "core_ast_20": row.get("core_ast_20"),
            "core_pts_10": row.get("core_pts_10"),
            "core_reb_10": row.get("core_reb_10"),
            "core_ast_10": row.get("core_ast_10"),
            "core_pts_season": row.get("core_pts_season"),
            "core_reb_season": row.get("core_reb_season"),
            "core_ast_season": row.get("core_ast_season"),
            "pts_core_10": row.get("pts_core_10"),
            "pts_core_20": row.get("pts_core_20"),
            "pts_core_season": row.get("pts_core_season"),
            "reb_core_10": row.get("reb_core_10"),
            "reb_core_20": row.get("reb_core_20"),
            "reb_core_season": row.get("reb_core_season"),
            "ast_core_10": row.get("ast_core_10"),
            "ast_core_20": row.get("ast_core_20"),
            "ast_core_season": row.get("ast_core_season"),
            "consistency_score": row.get("consistency_score"),
            "consistency_10": row.get("consistency_10"),
            "consistency_20": row.get("consistency_20"),
            "consistency_season": row.get("consistency_season"),
            "consistency_tier": row.get("consistency_tier"),
            "consistency_tier_10": row.get("consistency_tier_10"),
            "consistency_tier_20": row.get("consistency_tier_20"),
            "consistency_tier_season": row.get("consistency_tier_season"),
            "consistency_3pt_10": row.get("consistency_3pt_10"),
            "consistency_3pt_20": row.get("consistency_3pt_20"),
            "consistency_3pt_season": row.get("consistency_3pt_season"),
            "consistency_tier_3pt_10": row.get("consistency_tier_3pt_10"),
            "consistency_tier_3pt_20": row.get("consistency_tier_3pt_20"),
            "consistency_tier_3pt_season": row.get("consistency_tier_3pt_season"),
            "threept_eligible_10": row.get("threept_eligible_10"),
            "threept_eligible_20": row.get("threept_eligible_20"),
            "threept_eligible_season": row.get("threept_eligible_season"),
            "tpm_mean_10": row.get("tpm_mean_10"),
            "tpm_mean_20": row.get("tpm_mean_20"),
            "tpm_mean_season": row.get("tpm_mean_season"),
            "tpm_nonzero_rate_10": row.get("tpm_nonzero_rate_10"),
            "tpm_nonzero_rate_20": row.get("tpm_nonzero_rate_20"),
            "tpm_nonzero_rate_season": row.get("tpm_nonzero_rate_season"),
            "share_pts_l5": row.get("share_pts_l5"),
            "share_pts_l20": row.get("share_pts_l20"),
            "share_pts_trend_pp": row.get("share_pts_trend_pp"),
            "share_pts_trend_eligible": row.get("share_pts_trend_eligible"),
            "share_reb_l5": row.get("share_reb_l5"),
            "share_reb_l20": row.get("share_reb_l20"),
            "share_reb_trend_pp": row.get("share_reb_trend_pp"),
            "share_reb_trend_eligible": row.get("share_reb_trend_eligible"),
            "share_ast_l5": row.get("share_ast_l5"),
            "share_ast_l20": row.get("share_ast_l20"),
            "share_ast_trend_pp": row.get("share_ast_trend_pp"),
            "share_ast_trend_eligible": row.get("share_ast_trend_eligible"),
        }
        for win in ("10", "20", "season"):
            features[pid][f"gp_home_{win}"] = row.get(f"gp_home_{win}")
            features[pid][f"gp_away_{win}"] = row.get(f"gp_away_{win}")
            features[pid][f"split_usable_{win}"] = row.get(f"split_usable_{win}")
            features[pid][f"split_usable_3pt_{win}"] = row.get(f"split_usable_3pt_{win}")
            for stat in ("pts", "reb", "ast", "3pt"):
                features[pid][f"core_home_{stat}_{win}"] = row.get(f"core_home_{stat}_{win}")
                features[pid][f"core_away_{stat}_{win}"] = row.get(f"core_away_{stat}_{win}")
                features[pid][f"split_core_delta_{stat}_{win}"] = row.get(f"split_core_delta_{stat}_{win}")
                features[pid][f"split_core_z_{stat}_{win}"] = row.get(f"split_core_z_{stat}_{win}")
        for stat in ("pts", "reb", "ast", "3pt"):
            features[pid][f"trend_core_summary_dir_{stat}"] = row.get(f"trend_core_summary_dir_{stat}")
            features[pid][f"trend_cons_summary_dir_{stat}"] = row.get(f"trend_cons_summary_dir_{stat}")
            for suffix in ("10_20", "10_season"):
                features[pid][f"trend_core_z_{stat}_{suffix}"] = row.get(f"trend_core_z_{stat}_{suffix}")
                features[pid][f"trend_core_dir_{stat}_{suffix}"] = row.get(f"trend_core_dir_{stat}_{suffix}")
                features[pid][f"trend_cons_delta_{stat}_{suffix}"] = row.get(f"trend_cons_delta_{stat}_{suffix}")
                features[pid][f"trend_cons_dir_{stat}_{suffix}"] = row.get(f"trend_cons_dir_{stat}_{suffix}")
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/player_features.json", "w", encoding="utf-8") as f:
            json.dump({"asof_date": asof_str, "players": features}, f, indent=2)
    except Exception:
        pass
    if "split_usable_10" in out.columns:
        usable_10 = int(pd.to_numeric(out["split_usable_10"], errors="coerce").fillna(0).astype(int).sum())
        usable_3pt_10 = 0
        if "split_usable_3pt_10" in out.columns:
            usable_3pt_10 = int(pd.to_numeric(out["split_usable_3pt_10"], errors="coerce").fillna(0).astype(int).sum())
        print(f"[splits] usable_10={usable_10} usable_3pt_10={usable_3pt_10}")
    if "trend_core_summary_dir_pts" in out.columns:
        counts = out["trend_core_summary_dir_pts"].fillna("na").value_counts()
        summary_counts = {k: int(v) for k, v in counts.items()}
        print(f"[trend] core_summary_pts={summary_counts}")
    try:
        share_stats = {}
        share_up = {}
        share_down = {}
        share_flat = {}
        for stat in ("pts", "reb", "ast"):
            elig_col = f"share_{stat}_trend_eligible"
            trend_col = f"share_{stat}_trend_pp"
            if elig_col not in out.columns or trend_col not in out.columns:
                continue
            elig = out[elig_col].fillna(False).astype(bool)
            trends = pd.to_numeric(out[trend_col], errors="coerce")
            share_stats[stat] = int(elig.sum())
            share_up[stat] = int((elig & (trends >= 3.0)).sum())
            share_down[stat] = int((elig & (trends <= -3.0)).sum())
            share_flat[stat] = int((elig & (trends > -3.0) & (trends < 3.0)).sum())
        if share_stats:
            print(f"[share] eligible={share_stats} up={share_up} down={share_down} flat={share_flat}")
    except Exception:
        pass
    if tier_counts:
        print(f"[consistency] tiers={tier_counts}")
    if threept_counts["eligible"] or threept_counts["ineligible"]:
        print(f"[3pt] eligibility_10={threept_counts}")

    out.to_csv(args.out, index=False)
    print(f"Wrote snapshot: {args.out} (rows={len(out)})")


if __name__ == "__main__":
    main()
