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
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd

PTS_T = [10, 15, 20, 25, 30, 35]
RA_T = [2, 4, 6, 8, 10, 12, 14]
TPM_T = [1, 2, 3, 4, 5, 6]
LEARNING_FIELDS = [
    "is_home_game",
    "rest_days",
    "is_b2b",
    "share_trend_pp",
    "share_trend_dir",
    "iqr_10",
    "landmine_rate_10",
    "trap_risk",
]


def num(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v


def normalize_pid(value) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def trend_dir_from_pp(pp: float | None, thresh: float = 2.0) -> str | None:
    if pp is None or not isinstance(pp, (int, float)):
        return None
    if pp >= thresh:
        return "UP"
    if pp <= -thresh:
        return "DOWN"
    return "FLAT"


def compute_trap_risk(stat: str, iqr_val: float | None, landmine_rate: float | None) -> bool | None:
    stat_norm = str(stat or "").lower().strip()
    if stat_norm not in {"pts", "reb", "ast"}:
        return None
    if landmine_rate is not None and landmine_rate >= 0.25:
        return True
    if iqr_val is None:
        return False
    if stat_norm == "pts" and iqr_val > 10:
        return True
    if stat_norm == "reb" and iqr_val > 5:
        return True
    if stat_norm == "ast" and iqr_val > 4:
        return True
    return False


def build_last_game_lookup(game_log_df: pd.DataFrame) -> Dict[str, List[dt.date]]:
    if game_log_df.empty:
        return {}
    df = game_log_df.copy()
    if "game_date" not in df.columns or "player_id" not in df.columns:
        return {}
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df = df[df["game_date"].notna()].copy()
    df["player_id"] = df["player_id"].map(normalize_pid)
    dates_map: Dict[str, List[dt.date]] = {}
    for pid, group in df.groupby("player_id"):
        dates = sorted(set(group["game_date"].tolist()))
        if dates:
            dates_map[normalize_pid(pid)] = dates
    return dates_map


def rest_info_for_player(pid: str, slate_date: dt.date, dates_map: Dict[str, List[dt.date]]):
    pid = normalize_pid(pid)
    if not pid or pid not in dates_map:
        return None, None
    dates = dates_map.get(pid, [])
    if not dates:
        return None, None
    idx = bisect_left(dates, slate_date)
    if idx <= 0:
        return None, None
    last_date = dates[idx - 1]
    diff_days = (slate_date - last_date).days
    rest_days = max(0, diff_days - 1)
    return rest_days, rest_days == 0


def extract_home_flag(row: Dict[str, str]) -> bool | None:
    ha = str(row.get("home_away", "")).strip().upper()
    if ha in {"H", "HOME"}:
        return True
    if ha in {"A", "AWAY"}:
        return False
    return None


def load_player_features(path: str = "data/player_features.json") -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return {}
    if isinstance(data, dict):
        return {normalize_pid(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def build_snapshot_row_map(snapshot_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    if snapshot_df is None or snapshot_df.empty:
        return {}
    if "player_id" not in snapshot_df.columns:
        return {}
    out = {}
    for _, row in snapshot_df.iterrows():
        pid = normalize_pid(row.get("player_id", ""))
        if not pid:
            continue
        out[pid] = row.to_dict()
    return out


def enrich_learning_fields(entry: Dict[str, object], row: Dict[str, object] | None, rest_map, target_date: dt.date,
                           features_map: Dict[str, Dict[str, object]],
                           snapshot_map: Dict[str, Dict[str, object]]):
    updated = {}
    pid = normalize_pid(entry.get("player_id", ""))
    stat = str(entry.get("stat", "")).lower().strip()
    threshold = int(entry.get("threshold", 0) or 0)
    if entry.get("is_home_game") is None:
        home_flag = extract_home_flag(row or {})
        if home_flag is not None:
            updated["is_home_game"] = home_flag
    if entry.get("rest_days") is None or entry.get("is_b2b") is None:
        rest_days, is_b2b = rest_info_for_player(pid, target_date, rest_map)
        if entry.get("rest_days") is None:
            updated["rest_days"] = rest_days
        if entry.get("is_b2b") is None:
            updated["is_b2b"] = is_b2b
    if entry.get("share_trend_pp") is None or entry.get("share_trend_dir") is None:
        feat = features_map.get(pid, {}) if pid else {}
        key = f"share_{stat}_trend_pp" if stat and stat != "3pt" else "share_3pt_trend_pp"
        pp = num(feat.get(key))
        if entry.get("share_trend_pp") is None:
            updated["share_trend_pp"] = pp
        if entry.get("share_trend_dir") is None:
            updated["share_trend_dir"] = trend_dir_from_pp(pp)
    iqr_val = entry.get("iqr_10")
    landmine_rate = entry.get("landmine_rate_10")
    if (iqr_val is None or landmine_rate is None) and pid:
        snap = snapshot_map.get(pid, {})
        if iqr_val is None and stat:
            iqr_val = num(snap.get(f"{stat}_iqr_10"))
        if landmine_rate is None and stat in {"pts", "reb", "ast"}:
            landmine_rate = num(snap.get(f"{stat}_landmine_{threshold}_10"))
    if entry.get("iqr_10") is None:
        updated["iqr_10"] = iqr_val
    if entry.get("landmine_rate_10") is None:
        updated["landmine_rate_10"] = landmine_rate
    if entry.get("trap_risk") is None:
        updated["trap_risk"] = compute_trap_risk(stat, iqr_val, landmine_rate)
    return updated


def share_trend_from_row(row: Dict[str, str], stat: str) -> tuple[float | None, str | None]:
    stat_norm = str(stat or "").lower().strip()
    key = f"share_{stat_norm}_trend_pp" if stat_norm != "3pt" else "share_3pt_trend_pp"
    pp = num(row.get(key))
    if pp is None:
        return None, None
    return pp, trend_dir_from_pp(pp)


def learning_fields_from_row(row: Dict[str, str], stat: str, threshold: int, team: str, game_map, rest_map, slate_date: dt.date):
    is_home = None
    if team in game_map:
        is_home = bool(game_map[team].get("is_home"))
    pid = str(row.get("player_id", "")).strip()
    rest_days, is_b2b = rest_info_for_player(pid, slate_date, rest_map)
    share_pp, share_dir = share_trend_from_row(row, stat)
    iqr_key = f"{stat}_iqr_10"
    iqr_val = num(row.get(iqr_key))
    landmine_rate = None
    if str(stat or "").lower().strip() in {"pts", "reb", "ast"}:
        lm_key = f"{stat}_landmine_{threshold}_10"
        landmine_rate = num(row.get(lm_key))
    trap_risk = compute_trap_risk(stat, iqr_val, landmine_rate)
    return {
        "is_home_game": is_home,
        "rest_days": rest_days,
        "is_b2b": is_b2b,
        "share_trend_pp": share_pp,
        "share_trend_dir": share_dir,
        "iqr_10": iqr_val,
        "landmine_rate_10": landmine_rate,
        "trap_risk": trap_risk,
    }


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
    tpm = decision_target(row, "3pt", TPM_T, win, 0.5)
    if pts:
        targets.append(pts)
    if reb:
        targets.append(reb)
    if ast:
        targets.append(ast)
    if tpm:
        targets.append(tpm)
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
    dates = []
    for x in df["game_date"]:
        s = str(x).strip()
        if not s:
            continue
        m = re.search(r"\d{4}-\d{2}-\d{2}", s)
        dates.append(m.group(0) if m else s)
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
        team_map[home] = {"opp": away, "game_id": game_id, "is_home": True}
        team_map[away] = {"opp": home, "game_id": game_id, "is_home": False}
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


def compute_postmortem(snapshot_entries, game_log_df, target_date: str,
                       features_map: Dict[str, Dict[str, object]] | None = None,
                       snapshot_map: Dict[str, Dict[str, object]] | None = None) -> Dict[str, object]:
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
    rest_map = build_last_game_lookup(game_log_df)
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
        learning = {k: entry.get(k) for k in LEARNING_FIELDS if k in entry}
        try:
            target_date_obj = dt.datetime.strptime(target_date, "%Y-%m-%d").date()
        except Exception:
            target_date_obj = dt.date.today()
        row_dict = row.to_dict() if hasattr(row, "to_dict") else row
        extra = enrich_learning_fields(entry, row_dict, rest_map, target_date_obj, features_map or {}, snapshot_map or {})
        if extra:
            learning.update(extra)
        stat = entry.get("stat")
        threshold = int(entry.get("threshold", 0) or 0)
        dnp_flag = str(row.get("dnp", "") or "").strip() in ("1", "true", "True", "YES", "yes")
        try:
            minutes_val = float(row.get("minutes", 0) or 0)
        except Exception:
            minutes_val = 0.0
        out_like = dnp_flag or minutes_val == 0
        actual_val = resolve_stat_value(row, stat)
        actual = int(actual_val) if actual_val is not None else 0
        team_norm = str(row.get("team_abbrev", "")).upper().strip()
        if out_like:
            miss_item = {
                "player_id": entry.get("player_id"),
                "player_name": entry.get("player_name", ""),
                "team_abbrev": team_norm,
                "opp": entry.get("opp", row.get("opp_abbrev", "")),
                "stat": stat,
                "threshold": threshold,
                "actual": None,
                "delta": None,
                "out": True,
                "status": "OUT",
            }
            miss_item.update(learning)
            misses.append(miss_item)
            continue
        total += 1

        if actual >= threshold:
            hit_count += 1
            hit_item = {
                "player_id": entry.get("player_id"),
                "player_name": entry.get("player_name", ""),
                "team_abbrev": team_norm,
                "opp": entry.get("opp", row.get("opp_abbrev", "")),
                "stat": stat,
                "threshold": threshold,
                "actual": actual,
            }
            hit_item.update(learning)
            hits.append(hit_item)
        else:
            miss_item = {
                "player_id": entry.get("player_id"),
                "player_name": entry.get("player_name", ""),
                "team_abbrev": team_norm,
                "opp": entry.get("opp", row.get("opp_abbrev", "")),
                "stat": stat,
                "threshold": threshold,
                "actual": actual,
                "delta": threshold - actual,
            }
            miss_item.update(learning)
            misses.append(miss_item)

        if stat == "pts":
            incs = [5, 10]
        elif stat == "3pt":
            incs = [1, 2]
        else:
            incs = [2, 4]
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

    misses_sorted = sorted(
        misses,
        key=lambda x: (x["delta"] is None, x["delta"] if x["delta"] is not None else 0),
    )
    closest = []
    for m in misses_sorted:
        stat = m["stat"]
        threshold = m["threshold"]
        actual = m["actual"]
        if actual is None:
            continue
        if stat == "pts":
            ok = actual >= threshold - 3
        else:
            ok = actual >= threshold - 2
        if ok and actual < threshold:
            closest.append(m)
    closest = closest[:10]

    hit_rate = int(round((hit_count / total) * 100)) if total else 0

    return {
        "asof_date": target_date,
        "summary": {
            "targets_total": total,
            "hits": hit_count,
            "misses": max(0, total - hit_count),
            "hit_rate_pct": hit_rate,
        },
        "hits": hits,
        "misses": misses_sorted,
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
    df["game_date"] = df["game_date"].astype(str).str.strip().str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
    day_logs = df[df["game_date"] == target_date].copy()
    df["player_name_norm"] = df["player_name"].map(normalize_name)
    df["player_id_norm"] = df["player_id"].astype(str).str.strip()
    df["game_id_norm"] = df["game_id"].astype(str).str.strip()
    df["team_abbrev_norm"] = df["team_abbrev"].astype(str).str.upper().str.strip()
    df["opp_abbrev_norm"] = df["opp_abbrev"].astype(str).str.upper().str.strip()
    if not day_logs.empty:
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
    for _, row in df.iterrows():
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
    for _, row in day_logs.iterrows():
        name = row.get("player_name_norm", "")
        team = row.get("team_abbrev_norm", "")
        opp = row.get("opp_abbrev_norm", "")
        mval = minutes_val(row)
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
        "3pt": ["tpm", "3pm", "3pt", "fg3m", "3fgm"],
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
    if os.path.exists(game_log_path):
        game_log_df = pd.read_csv(game_log_path, dtype=str)
    else:
        game_log_df = pd.DataFrame()
    features_map = load_player_features()
    snapshot_row_map = build_snapshot_row_map(snapshot_df) if snapshot_df is not None else {}
    slate_date_obj = None
    try:
        slate_date_obj = dt.datetime.strptime(slate_date, "%Y-%m-%d").date()
    except Exception:
        slate_date_obj = None
    rest_map = build_last_game_lookup(game_log_df) if slate_date_obj else {}
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
                learning = (
                    learning_fields_from_row(row, t["stat"], int(t["t"]), team, game_map, rest_map, slate_date_obj)
                    if slate_date_obj else {}
                )
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
                    **learning,
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

    results_date = latest_final_date_from_master(master_path, local_today)
    if not results_date:
        results_date = determine_postmortem_date_from_game_log(game_log_path)
    if not results_date:
        results_date = (local_today - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        res_date_obj = dt.datetime.strptime(results_date, "%Y-%m-%d").date()
        if res_date_obj >= local_today:
            results_date = (local_today - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        pass

    snapshot_entries = []
    note = None
    snapshot_path = logs_dir / f"targets_snapshot_{results_date}.json"
    if not snapshot_path.exists():
        original_results_date = results_date
        fallback_date = snapshot_date_at_or_before(logs_dir, results_date)
        if fallback_date:
            snapshot_path = logs_dir / f"targets_snapshot_{fallback_date}.json"
            results_date = fallback_date
            note = f"Using snapshot from {fallback_date} for results_date {original_results_date}"
        else:
            snapshot_path = None
            note = f"No targets snapshot found for {results_date}"

    if snapshot_path and snapshot_path.exists():
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot_entries = json.load(f) or []
    else:
        print(f"[targets] WARNING: snapshot missing for results_date={results_date}")

    top_targets = []
    summary_top_total = 0
    summary_top_hits = 0
    indexes = build_game_log_indexes(game_log_df, results_date)
    enriched_found = 0
    failed_samples = []
    out_count = 0
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
        learning_index = {}
        for item in snap_items:
            if not isinstance(item, dict):
                continue
            key = (
                normalize_pid(item.get("player_id", "")),
                str(item.get("stat", "")).strip(),
                str(item.get("threshold", "")).strip(),
                str(item.get("game_id", "")).strip(),
                str(item.get("team_abbrev", "")).strip(),
            )
            learning_index[key] = item
        for item in top_targets:
            key = (
                normalize_pid(item.get("player_id", "")),
                str(item.get("stat", "")).strip(),
                str(item.get("threshold", "")).strip(),
                str(item.get("game_id", "")).strip(),
                str(item.get("team_abbrev", "")).strip(),
            )
            src = learning_index.get(key)
            if not src:
                continue
            for field in LEARNING_FIELDS:
                if field not in item and field in src:
                    item[field] = src.get(field)

        for item in top_targets:
            row = find_game_log_row(item, indexes)
            stat = item.get("stat")
            threshold = int(item.get("threshold", 0) or 0)
            actual = None
            hit = None
            delta = None
            if row is not None and stat:
                dnp_flag = str(row.get("dnp", "") or "").strip() in ("1", "true", "True", "YES", "yes")
                try:
                    minutes_val = float(row.get("minutes", 0) or 0)
                except Exception:
                    minutes_val = 0.0
                out_like = dnp_flag or minutes_val == 0
                if out_like:
                    item["actual"] = None
                    item["hit"] = None
                    item["delta"] = None
                    item["dnp"] = True
                    item["status"] = "OUT"
                    out_count += 1
                    continue
                actual_val = resolve_stat_value(row, stat)
                if actual_val is not None:
                    actual = int(actual_val)
                    hit = actual >= threshold
                    delta = threshold - actual
                item["dnp"] = False
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
        print(f"[targets] Top targets OUT/DNP: {out_count}")
        if failed_samples:
            print(f"[targets] Top targets missing samples: {failed_samples}")

    postmortem = compute_postmortem(snapshot_entries, game_log_df, results_date, features_map, snapshot_row_map)
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
    history_dir = data_dir / "history" / "targets_postmortem"
    history_dir.mkdir(parents=True, exist_ok=True)
    with open(history_dir / f"targets_postmortem_{results_date}.json", "w", encoding="utf-8") as f:
        json.dump(postmortem, f, indent=2)

    print(
        f"[targets] local_today={local_today} slate_date={slate_date} results_date={results_date} "
        f"snapshot_used={postmortem.get('snapshot_file')} entries={len(entries)} "
        f"total={postmortem.get('summary', {}).get('targets_total')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
