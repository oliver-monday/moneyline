#!/usr/bin/env python3
"""
Persist today's L10 targets snapshot and compute yesterday post-mortem.
Outputs:
  - logs/targets_snapshot_YYYY-MM-DD.json (not deployed)
  - data/targets_postmortem.json (deployed)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

PTS_T = [10, 15, 20, 25, 30, 35]
RA_T = [2, 4, 6, 8, 10, 12, 14]


def num(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v


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


def tokens(norm_name: str) -> List[str]:
    return [t for t in (norm_name or "").split(" ") if t]


def abbrev_match(inj_norm: str, player_norm: str) -> bool:
    it = tokens(inj_norm)
    pt = tokens(player_norm)
    if len(it) < 2 or len(pt) < 2:
        return False
    first = it[0]
    if len(first) != 1:
        return False
    rest = " ".join(it[1:])
    player_rest = " ".join(pt[1:])
    return (pt[0][0] == first) and (rest in player_rest)


def match_injury(team: str, player_name: str, injuries_by_team: Dict[str, List[Dict[str, str]]]):
    items = (injuries_by_team or {}).get(team, []) or []
    target = normalize_name(player_name)
    best = None
    for item in items:
        name_norm = normalize_name(item.get("name", ""))
        if not name_norm:
            continue
        hit = (
            target.find(name_norm) >= 0
            or name_norm.find(target) >= 0
            or abbrev_match(name_norm, target)
        )
        if hit:
            if not best or len(name_norm) > len(best["name_norm"]):
                best = {**item, "name_norm": name_norm}
    return best


def is_out_status(status: str) -> bool:
    return str(status or "").strip().upper() == "OUT"


def season_avg(row: Dict[str, str], stat: str):
    v = num(row.get(f"season_avg_{stat}"))
    if v is not None:
        return v
    return num(row.get(f"{stat}_avg_season"))


def decision_target(row: Dict[str, str], stat: str, thresholds: List[int], win: str, band: float):
    gp = int(float(row.get(f"gp_{win}", 0) or 0))
    avg = season_avg(row, stat)
    if not gp or avg is None:
        return None

    in_band = [t for t in thresholds if abs(avg - t) <= band]
    if not in_band:
        return None

    candidates = []
    for t in in_band:
        hits = int(float(row.get(f"{stat}_ge{t}_hits_{win}", 0) or 0))
        rate = hits / gp if gp else 0
        if rate >= 0.8:
            candidates.append({"t": t, "hits": hits, "gp": gp, "rate": rate})
    if not candidates:
        return None

    perfect = [c for c in candidates if c["rate"] == 1]
    pick_from = perfect if perfect else candidates
    best = sorted(pick_from, key=lambda x: x["t"], reverse=True)[0]
    return {"stat": stat, **best}


def collect_targets(row: Dict[str, str], win: str):
    targets = []
    pts = decision_target(row, "pts", PTS_T, win, 2.5)
    reb = decision_target(row, "reb", RA_T, win, 1.0)
    ast = decision_target(row, "ast", RA_T, win, 1.0)
    if pts:
        targets.append(pts)
    if reb:
        targets.append(reb)
    if ast:
        targets.append(ast)
    return targets


def determine_slate_date(master_path: str) -> str | None:
    if not os.path.exists(master_path):
        return None
    master = pd.read_csv(master_path, dtype=str)
    dates = [str(x).strip() for x in master.get("game_date", []) if str(x).strip()]
    if not dates:
        return None
    today_local = dt.date.today().strftime("%Y-%m-%d")
    if today_local in dates:
        return today_local
    return sorted(set(dates))[-1]


def build_game_map(master: pd.DataFrame, slate_date: str):
    games_today = master[master["game_date"].astype(str).str.strip() == slate_date].copy()
    team_map = {}
    for _, g in games_today.iterrows():
        home = str(g.get("home_team_abbrev", "")).strip().upper()
        away = str(g.get("away_team_abbrev", "")).strip().upper()
        if not home or not away:
            continue
        game_id = str(g.get("game_id", "")).strip()
        team_map[home] = {"opp": away, "game_id": game_id}
        team_map[away] = {"opp": home, "game_id": game_id}
    return team_map


def write_snapshot(snapshot_path: Path, entries: List[Dict[str, object]]) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def prune_old_snapshots(logs_dir: Path, cutoff: dt.date) -> None:
    pattern = re.compile(r"targets_snapshot_(\d{4}-\d{2}-\d{2})\.json$")
    for path in logs_dir.glob("targets_snapshot_*.json"):
        m = pattern.match(path.name)
        if not m:
            continue
        try:
            snap_date = dt.datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            continue
        if snap_date < cutoff:
            try:
                path.unlink()
            except Exception:
                pass


def compute_postmortem(snapshot_entries, game_log_df, target_date: str) -> Dict[str, object]:
    empty = {
        "asof_date": target_date,
        "summary": {"targets_total": 0, "hits": 0, "misses": 0, "hit_rate_pct": 0},
        "hits": [],
        "closest_misses": [],
        "reaches_hit": [],
    }
    if not snapshot_entries:
        return empty

    game_log_df = game_log_df.copy()
    game_log_df["game_date"] = game_log_df["game_date"].astype(str).str.strip()
    day_logs = game_log_df[game_log_df["game_date"] == target_date].copy()
    if day_logs.empty:
        return empty

    day_logs["player_name_norm"] = day_logs["player_name"].map(normalize_name)
    day_logs["team_abbrev_norm"] = day_logs["team_abbrev"].astype(str).str.upper().str.strip()
    day_logs = day_logs.sort_values("minutes", ascending=False)
    day_logs = day_logs.drop_duplicates(subset=["player_name_norm", "team_abbrev_norm"], keep="first")
    log_index = {
        (r["player_name_norm"], r["team_abbrev_norm"]): r for _, r in day_logs.iterrows()
    }

    hits = []
    misses = []
    reaches = []
    total = 0
    hit_count = 0

    for entry in snapshot_entries:
        name_norm = normalize_name(entry.get("player_name", ""))
        team_norm = str(entry.get("team_abbrev", "")).upper().strip()
        key = (name_norm, team_norm)
        row = log_index.get(key)
        if row is None:
            continue
        stat = entry.get("stat")
        threshold = int(entry.get("threshold", 0) or 0)
        actual = int(float(row.get(stat, 0) or 0))
        total += 1

        if actual >= threshold:
            hit_count += 1
            if len(hits) < 20:
                hits.append({
                    "player_name": entry.get("player_name", ""),
                    "team_abbrev": team_norm,
                    "opp": entry.get("opp", row.get("opp_abbrev", "")),
                    "stat": stat,
                    "threshold": threshold,
                    "actual": actual,
                })
        else:
            misses.append({
                "player_name": entry.get("player_name", ""),
                "team_abbrev": team_norm,
                "opp": entry.get("opp", row.get("opp_abbrev", "")),
                "stat": stat,
                "threshold": threshold,
                "actual": actual,
                "delta": threshold - actual,
            })

        incs = [5, 10] if stat == "pts" else [2, 4]
        reached = None
        for inc in reversed(incs):
            if actual >= threshold + inc:
                reached = threshold + inc
                break
        if reached is not None and len(reaches) < 5:
            reaches.append({
                "player_name": entry.get("player_name", ""),
                "team_abbrev": team_norm,
                "opp": entry.get("opp", row.get("opp_abbrev", "")),
                "stat": stat,
                "target_threshold": threshold,
                "reached_threshold": reached,
                "actual": actual,
            })

    closest = []
    for m in misses:
        stat = m["stat"]
        threshold = m["threshold"]
        actual = m["actual"]
        if stat == "pts":
            ok = actual >= threshold - 3
        else:
            ok = actual >= threshold - 2
        if ok and actual < threshold:
            closest.append(m)
    closest = sorted(closest, key=lambda x: x["delta"])[:10]

    hit_rate = int(round((hit_count / total) * 100)) if total else 0

    return {
        "asof_date": target_date,
        "summary": {
            "targets_total": total,
            "hits": hit_count,
            "misses": max(0, total - hit_count),
            "hit_rate_pct": hit_rate,
        },
        "hits": hits[:20],
        "closest_misses": closest,
        "reaches_hit": reaches[:5],
    }


def main() -> int:
    master_path = "nba_master.csv"
    snapshot_path = "player_snapshot.csv"
    game_log_path = "player_game_log.csv"
    injuries_path = "data/injuries_today.json"

    slate_date = determine_slate_date(master_path)
    if not slate_date:
        print("[targets] No slate date found; skipping.")
        return 0

    master = pd.read_csv(master_path, dtype=str)
    game_map = build_game_map(master, slate_date)

    snapshot_df = None
    if not os.path.exists(snapshot_path):
        print("[targets] Missing player_snapshot.csv; writing empty snapshot.")
    else:
        snapshot_df = pd.read_csv(snapshot_path, dtype=str).fillna("")

    injuries_by_team = {}
    if os.path.exists(injuries_path):
        try:
            with open(injuries_path, "r", encoding="utf-8") as f:
                injuries_by_team = json.load(f) or {}
        except Exception:
            injuries_by_team = {}

    entries = []
    if snapshot_df is not None:
        for _, row in snapshot_df.iterrows():
            team = str(row.get("last_team_abbrev", "")).strip().upper()
            if not team or team not in game_map:
                continue
            injury = match_injury(team, row.get("player_name", ""), injuries_by_team)
            status = (injury or {}).get("status", "")
            if is_out_status(status):
                continue

            targets = collect_targets(row, "10")
            if not targets:
                continue
            opp = game_map[team].get("opp", "")
            game_id = game_map[team].get("game_id", "")
            for t in targets:
                entries.append({
                    "asof_date": slate_date,
                    "player_name": row.get("player_name", ""),
                    "team_abbrev": team,
                    "opp": opp,
                    "game_id": game_id,
                    "stat": t["stat"],
                    "threshold": int(t["t"]),
                    "hits": int(t["hits"]),
                    "window_games": int(t["gp"]),
                    "hit_rate": float(t["rate"]),
                })

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = logs_dir / f"targets_snapshot_{slate_date}.json"
    write_snapshot(snapshot_file, entries)
    cutoff = dt.datetime.strptime(slate_date, "%Y-%m-%d").date() - dt.timedelta(days=60)
    prune_old_snapshots(logs_dir, cutoff)

    postmortem_date = (dt.datetime.strptime(slate_date, "%Y-%m-%d").date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    snapshot_yesterday = logs_dir / f"targets_snapshot_{postmortem_date}.json"
    if snapshot_yesterday.exists():
        with open(snapshot_yesterday, "r", encoding="utf-8") as f:
            snapshot_entries = json.load(f) or []
    else:
        snapshot_entries = []

    if os.path.exists(game_log_path):
        game_log_df = pd.read_csv(game_log_path, dtype=str)
    else:
        game_log_df = pd.DataFrame()

    postmortem = compute_postmortem(snapshot_entries, game_log_df, postmortem_date)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "targets_postmortem.json", "w", encoding="utf-8") as f:
        json.dump(postmortem, f, indent=2)

    print(f"[targets] snapshot_date={slate_date} entries={len(entries)} postmortem_date={postmortem_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
