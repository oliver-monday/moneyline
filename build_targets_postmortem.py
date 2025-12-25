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
from zoneinfo import ZoneInfo

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


def determine_postmortem_date_from_game_log(path: str = "player_game_log.csv") -> str | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return None
    if "game_date" not in df.columns:
        return None
    dates = [str(x).strip() for x in df["game_date"] if str(x).strip()]
    if not dates:
        return None
    return sorted(set(dates))[-1]


def newest_snapshot_date(logs_dir: Path) -> str | None:
    pattern = re.compile(r"targets_snapshot_(\d{4}-\d{2}-\d{2})\.json$")
    dates = []
    for path in logs_dir.glob("targets_snapshot_*.json"):
        m = pattern.match(path.name)
        if not m:
            continue
        dates.append(m.group(1))
    if not dates:
        return None
    return sorted(dates)[-1]


def newest_snapshot_path(logs_dir: Path) -> Path | None:
    date = newest_snapshot_date(logs_dir)
    if not date:
        return None
    return logs_dir / f"targets_snapshot_{date}.json"


def latest_final_date_from_master(master_path: str, local_today: dt.date) -> str | None:
    if not os.path.exists(master_path):
        return None
    master = pd.read_csv(master_path, dtype=str)
    if "game_date" not in master.columns:
        return None
    def is_numeric(v: str) -> bool:
        return str(v or "").strip().isdigit()

    dates = []
    for _, row in master.iterrows():
        gd = str(row.get("game_date", "")).strip()
        if not gd:
            continue
        try:
            gd_date = dt.datetime.strptime(gd, "%Y-%m-%d").date()
        except Exception:
            continue
        if gd_date > local_today:
            continue
        scored = is_numeric(row.get("home_score")) and is_numeric(row.get("away_score"))
        if scored:
            dates.append(gd)
    if not dates:
        return None
    return sorted(set(dates))[-1]


def snapshot_date_at_or_before(logs_dir: Path, target_date: str) -> str | None:
    pattern = re.compile(r"targets_snapshot_(\d{4}-\d{2}-\d{2})\.json$")
    candidates = []
    for path in logs_dir.glob("targets_snapshot_*.json"):
        m = pattern.match(path.name)
        if not m:
            continue
        snap_date = m.group(1)
        if snap_date <= target_date:
            candidates.append(snap_date)
    if not candidates:
        return None
    return sorted(candidates)[-1]


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
    indexes = build_game_log_indexes(game_log_df, target_date)
    if not any(indexes):
        return empty

    hits = []
    misses = []
    reaches = []
    total = 0
    hit_count = 0

    for entry in snapshot_entries:
        row = find_game_log_row(entry, indexes)
        if row is None:
            continue
        stat = entry.get("stat")
        threshold = int(entry.get("threshold", 0) or 0)
        actual_val = resolve_stat_value(row, stat)
        actual = int(actual_val) if actual_val is not None else 0
        team_norm = str(row.get("team_abbrev", "")).upper().strip()
        total += 1

        if actual >= threshold:
            hit_count += 1
            if len(hits) < 20:
                hits.append({
                    "player_id": entry.get("player_id"),
                    "player_name": entry.get("player_name", ""),
                    "team_abbrev": team_norm,
                    "opp": entry.get("opp", row.get("opp_abbrev", "")),
                    "stat": stat,
                    "threshold": threshold,
                    "actual": actual,
                })
        else:
            misses.append({
                "player_id": entry.get("player_id"),
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
                "player_id": entry.get("player_id"),
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


def build_log_index(game_log_df: pd.DataFrame, target_date: str):
    if game_log_df.empty:
        return {}
    df = game_log_df.copy()
    df["game_date"] = df["game_date"].astype(str).str.strip()
    day_logs = df[df["game_date"] == target_date].copy()
    if day_logs.empty:
        return {}
    day_logs["player_name_norm"] = day_logs["player_name"].map(normalize_name)
    day_logs["team_abbrev_norm"] = day_logs["team_abbrev"].astype(str).str.upper().str.strip()
    day_logs = day_logs.sort_values("minutes", ascending=False)
    day_logs = day_logs.drop_duplicates(subset=["player_name_norm", "team_abbrev_norm"], keep="first")
    return {
        (r["player_name_norm"], r["team_abbrev_norm"]): r for _, r in day_logs.iterrows()
    }


def build_game_log_indexes(game_log_df: pd.DataFrame, target_date: str):
    if game_log_df.empty:
        return {}, {}, {}, {}
    df = game_log_df.copy()
    df["game_date"] = df["game_date"].astype(str).str.strip()
    day_logs = df[df["game_date"] == target_date].copy()
    if day_logs.empty:
        return {}, {}, {}, {}
    day_logs["player_name_norm"] = day_logs["player_name"].map(normalize_name)
    day_logs["player_id_norm"] = day_logs["player_id"].astype(str).str.strip()
    day_logs["game_id_norm"] = day_logs["game_id"].astype(str).str.strip()
    day_logs["team_abbrev_norm"] = day_logs["team_abbrev"].astype(str).str.upper().str.strip()
    day_logs["opp_abbrev_norm"] = day_logs["opp_abbrev"].astype(str).str.upper().str.strip()
    def minutes_val(row):
        for col in ("minutes", "mins", "min", "minutes_raw"):
            if col in row and pd.notna(row[col]):
                try:
                    return float(row[col])
                except Exception:
                    return 0.0
        return 0.0
    by_pid_gid = {}
    by_name_gid = {}
    by_name_team = {}
    by_name_team_opp = {}
    for _, row in day_logs.iterrows():
        gid = str(row.get("game_id_norm", "")).strip()
        pid = str(row.get("player_id_norm", "")).strip()
        name = row.get("player_name_norm", "")
        team = row.get("team_abbrev_norm", "")
        opp = row.get("opp_abbrev_norm", "")
        mval = minutes_val(row)
        if gid and pid:
            key = (pid, gid)
            cur = by_pid_gid.get(key)
            if not cur or mval > cur[0]:
                by_pid_gid[key] = (mval, row)
        if gid and name:
            key = (name, gid)
            cur = by_name_gid.get(key)
            if not cur or mval > cur[0]:
                by_name_gid[key] = (mval, row)
        if name and team:
            key = (name, team)
            cur = by_name_team.get(key)
            if not cur or mval > cur[0]:
                by_name_team[key] = (mval, row)
        if name and team and opp:
            key = (name, team, opp)
            cur = by_name_team_opp.get(key)
            if not cur or mval > cur[0]:
                by_name_team_opp[key] = (mval, row)
    by_pid_gid = {k: v[1] for k, v in by_pid_gid.items()}
    by_name_gid = {k: v[1] for k, v in by_name_gid.items()}
    by_name_team = {k: v[1] for k, v in by_name_team.items()}
    by_name_team_opp = {k: v[1] for k, v in by_name_team_opp.items()}
    return by_pid_gid, by_name_gid, by_name_team, by_name_team_opp


def resolve_stat_value(row, stat: str):
    stat_norm = str(stat or "").lower().strip()
    candidates = {
        "pts": ["pts", "points"],
        "reb": ["reb", "rebounds"],
        "ast": ["ast", "assists"],
    }.get(stat_norm, [stat_norm])
    for col in candidates:
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                continue
    return None


def find_game_log_row(entry, indexes):
    by_pid_gid, by_name_gid, by_name_team, by_name_team_opp = indexes
    pid = str(entry.get("player_id", "")).strip()
    gid = str(entry.get("game_id", "")).strip()
    name_norm = normalize_name(entry.get("player_name", ""))
    team_norm = str(entry.get("team_abbrev", "")).upper().strip()
    opp_norm = str(entry.get("opp", "")).upper().strip()
    row = None
    if pid and gid:
        row = by_pid_gid.get((pid, gid))
    if row is None and gid and name_norm:
        row = by_name_gid.get((name_norm, gid))
    if row is None and name_norm and team_norm and opp_norm:
        row = by_name_team_opp.get((name_norm, team_norm, opp_norm))
    if row is None and name_norm and team_norm:
        row = by_name_team.get((name_norm, team_norm))
    return row


def main() -> int:
    master_path = "nba_master.csv"
    snapshot_path = "player_snapshot.csv"
    game_log_path = "player_game_log.csv"
    injuries_path = "data/injuries_today.json"

    local_today = dt.datetime.now(ZoneInfo("America/Los_Angeles")).date()
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
                    "player_id": row.get("player_id", ""),
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
    cutoff = local_today - dt.timedelta(days=120)
    min_keep = local_today - dt.timedelta(days=30)
    for path in logs_dir.glob("targets_snapshot_*.json"):
        m = re.match(r"targets_snapshot_(\d{4}-\d{2}-\d{2})\.json$", path.name)
        if not m:
            continue
        try:
            snap_date = dt.datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            continue
        if snap_date >= min_keep:
            continue
        if snap_date < cutoff:
            try:
                path.unlink()
            except Exception:
                pass

    results_date = determine_postmortem_date_from_game_log(game_log_path)
    if not results_date:
        results_date = latest_final_date_from_master(master_path, local_today)
    if not results_date:
        results_date = (local_today - dt.timedelta(days=1)).strftime("%Y-%m-%d")

    snapshot_entries = []
    note = None
    snapshot_path = logs_dir / f"targets_snapshot_{results_date}.json"
    if not snapshot_path.exists():
        fallback_date = snapshot_date_at_or_before(logs_dir, results_date)
        if fallback_date:
            snapshot_path = logs_dir / f"targets_snapshot_{fallback_date}.json"
            results_date = fallback_date
            note = f"Using snapshot from {fallback_date} for results_date {results_date}"
        else:
            snapshot_path = None
            note = f"No targets snapshot found for {results_date}"

    if snapshot_path and snapshot_path.exists():
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot_entries = json.load(f) or []

    if os.path.exists(game_log_path):
        game_log_df = pd.read_csv(game_log_path, dtype=str)
    else:
        game_log_df = pd.DataFrame()

    top_targets = []
    summary_top_total = 0
    summary_top_hits = 0
    indexes = build_game_log_indexes(game_log_df, results_date)
    enriched_found = 0
    failed_samples = []
    if snapshot_path and snapshot_path.exists():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snap_items = json.load(f) or []
        except Exception:
            snap_items = []
        candidates = []
        for item in snap_items:
            if not isinstance(item, dict):
                continue
            candidates.append(item)
        def rank_key(item):
            rate = item.get("hit_rate")
            if rate is None:
                hits = num(item.get("hits")) or 0
                gp = num(item.get("window_games")) or 0
                rate = hits / gp if gp else 0
            gp = num(item.get("window_games")) or 0
            return (-rate, -gp)
        candidates = sorted(candidates, key=rank_key)
        top_targets = candidates[:6]

        for item in top_targets:
            row = find_game_log_row(item, indexes)
            stat = item.get("stat")
            threshold = int(item.get("threshold", 0) or 0)
            actual = None
            hit = None
            delta = None
            if row is not None and stat:
                actual_val = resolve_stat_value(row, stat)
                if actual_val is not None:
                    actual = int(actual_val)
                    hit = actual >= threshold
                    delta = threshold - actual
                enriched_found += 1
            else:
                if len(failed_samples) < 3:
                    failed_samples.append({
                        "player": item.get("player_name", ""),
                        "team": item.get("team_abbrev", ""),
                        "opp": item.get("opp", ""),
                        "stat": stat,
                        "threshold": threshold,
                    })
            item["actual"] = actual
            item["hit"] = bool(hit) if hit is not None else False
            item["delta"] = delta
        summary_top_total = len(top_targets)
        summary_top_hits = sum(1 for t in top_targets if t.get("hit"))
        print(f"[targets] Top targets enriched: {enriched_found}/{summary_top_total} (rows found)")
        if failed_samples:
            print(f"[targets] Top targets missing samples: {failed_samples}")

    postmortem = compute_postmortem(snapshot_entries, game_log_df, results_date)
    postmortem["built_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    postmortem["snapshot_file"] = snapshot_path.name if snapshot_path else None
    postmortem["top_targets"] = top_targets
    postmortem["summary"]["top_targets_total"] = summary_top_total
    postmortem["summary"]["top_targets_hits"] = summary_top_hits
    if note:
        postmortem["note"] = note
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "targets_postmortem.json", "w", encoding="utf-8") as f:
        json.dump(postmortem, f, indent=2)

    print(
        f"[targets] local_today={local_today} slate_date={slate_date} results_date={results_date} "
        f"snapshot_used={postmortem.get('snapshot_file')} entries={len(entries)} "
        f"total={postmortem.get('summary', {}).get('targets_total')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
