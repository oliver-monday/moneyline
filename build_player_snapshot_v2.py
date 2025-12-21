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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def season_end_year_for_date(d: dt.date) -> int:
    return d.year + 1 if d.month >= 10 else d.year


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


def build_windows(df_player: pd.DataFrame, windows: List[int], thresholds: Dict[str, List[float]]) -> Dict[str, float]:
    """df_player sorted desc by game_date and filtered to games played (minutes>0)."""
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

        for stat in ("pts", "reb", "ast"):
            out[f"{stat}_min_{w}"] = safe_min(d[stat]) if gp else float("nan")

            for t in thresholds.get(stat, []):
                tkey = str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
                if gp:
                    hits, rate = rate_and_hits_ge(d[stat], t)
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
    df = df[df["season_end_year"].astype(int) == int(season_end)].copy()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    windows = sorted(list(dict.fromkeys(windows)))

    thresholds = {
        "pts": [10, 15, 20, 25, 30, 35],
        "reb": [2, 4, 6, 8, 10, 12, 14],
        "ast": [2, 4, 6, 8, 10, 12, 14],
    }

    for c in ["minutes","pts","reb","ast","dnp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    rows = []
    for (pid, pname), g in df.groupby(["player_id","player_name"], sort=False):
        g = g.sort_values("game_date", ascending=False)

        gp = g[(g["minutes"] > 0) & (g["dnp"] != 1)].copy().sort_values("game_date", ascending=False)
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
            "games_total_rows": int(len(g)),
            "games_played_rows": int((g["minutes"] > 0).sum()),
            "dnp_rows": int((g["dnp"] == 1).sum()),
        }

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

    out.to_csv(args.out, index=False)
    print(f"Wrote snapshot: {args.out} (rows={len(out)})")


if __name__ == "__main__":
    main()
