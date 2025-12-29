#!/usr/bin/env python3
"""
Build daily performance ledger + season/all-time summaries from targets_postmortem.json.
Writes under data/perf/ and mirrors daily JSON/CSV artifacts.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def season_end_year_for_date(d: dt.date) -> int:
    return d.year + 1 if d.month >= 7 else d.year


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def out_like(item: Dict[str, Any]) -> bool:
    status = str(item.get("status", "")).upper().strip()
    if item.get("dnp") is True or item.get("out") is True:
        return True
    return status in {"OUT", "OFS", "SUSP"}


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshot(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def find_snapshot_for_date(date_str: str) -> Optional[str]:
    cand = os.path.join("logs", f"targets_snapshot_{date_str}.json")
    return cand if os.path.exists(cand) else None


def parse_asof_date(j: Dict[str, Any]) -> str:
    asof = str(j.get("asof_date") or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", asof):
        return asof
    return dt.datetime.utcnow().date().isoformat()


def calc_top_targets(post: Dict[str, Any]) -> Tuple[int, int]:
    summary = post.get("summary") or {}
    top_hits = safe_int(summary.get("top_targets_hits") or summary.get("top_targets_hit"))
    top_total = safe_int(summary.get("top_targets_total") or summary.get("top_targets_count"))
    if top_hits is not None and top_total is not None:
        return top_hits, top_total
    top_list = post.get("top_targets") or []
    hits = 0
    total = 0
    for item in top_list:
        if out_like(item):
            continue
        total += 1
        if item.get("hit") is True:
            hits += 1
    return hits, total


def calc_all_targets(post: Dict[str, Any]) -> Tuple[int, int]:
    summary = post.get("summary") or {}
    all_hits = safe_int(summary.get("hits") or summary.get("hits_total") or summary.get("hit_rate_hits"))
    all_total = safe_int(summary.get("targets_total") or summary.get("hit_rate_total") or summary.get("total"))
    if all_hits is not None and all_total is not None:
        return all_hits, all_total
    hits_list = post.get("hits") or []
    misses_list = post.get("misses") or post.get("closest_misses") or []
    hits = 0
    total = 0
    for item in hits_list:
        if out_like(item):
            continue
        hits += 1
        total += 1
    for item in misses_list:
        if out_like(item):
            continue
        total += 1
    return hits, total


def build_daily_payload(post: Dict[str, Any], asof: str, season_end_year: int) -> Dict[str, Any]:
    built_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    sha = os.environ.get("GITHUB_SHA", "")
    top_hits, top_total = calc_top_targets(post)
    all_hits, all_total = calc_all_targets(post)
    props = {
        "top_targets": {
            "hits": int(top_hits),
            "total": int(top_total),
            "hit_rate": (top_hits / top_total) if top_total else None,
        },
        "all_targets": {
            "hits": int(all_hits),
            "total": int(all_total),
            "hit_rate": (all_hits / all_total) if all_total else None,
        },
        "reaches_hit": safe_int(post.get("summary", {}).get("reaches_hit")) or (len(post.get("reaches_hit") or [])) or None,
        "closest_misses": safe_int(post.get("summary", {}).get("closest_misses")) or (len(post.get("closest_misses") or [])) or None,
    }
    snap_path = find_snapshot_for_date(asof)
    return {
        "date": asof,
        "season_end_year": season_end_year,
        "built_at_utc": built_at,
        "git_sha": sha,
        "props": props,
        "sources": {
            "targets_postmortem": "data/targets_postmortem.json",
            "targets_snapshot": snap_path if snap_path else "",
        },
    }


def write_json_if_changed(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing == payload:
                return
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def collect_daily_files(season_dir: str) -> List[str]:
    daily_dir = os.path.join(season_dir, "daily")
    if not os.path.isdir(daily_dir):
        return []
    files = [os.path.join(daily_dir, f) for f in os.listdir(daily_dir) if f.endswith(".json")]
    return sorted(files)


def summarize_season(season_end_year: int) -> Dict[str, Any]:
    season_dir = os.path.join("data", "perf", f"season_{season_end_year}")
    daily_files = collect_daily_files(season_dir)
    totals_top_hits = totals_top_total = 0
    totals_all_hits = totals_all_total = 0
    dates = []
    for path in daily_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        dates.append(j.get("date"))
        props = j.get("props") or {}
        top = props.get("top_targets") or {}
        all_t = props.get("all_targets") or {}
        totals_top_hits += int(top.get("hits") or 0)
        totals_top_total += int(top.get("total") or 0)
        totals_all_hits += int(all_t.get("hits") or 0)
        totals_all_total += int(all_t.get("total") or 0)

    dates = [d for d in dates if isinstance(d, str)]
    start_date = min(dates) if dates else ""
    end_date = max(dates) if dates else ""
    built_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return {
        "season_end_year": season_end_year,
        "start_date": start_date,
        "end_date": end_date,
        "days_tracked": len(dates),
        "top_targets": {
            "hits": totals_top_hits,
            "total": totals_top_total,
            "hit_rate": (totals_top_hits / totals_top_total) if totals_top_total else None,
        },
        "all_targets": {
            "hits": totals_all_hits,
            "total": totals_all_total,
            "hit_rate": (totals_all_hits / totals_all_total) if totals_all_total else None,
        },
        "last_updated_utc": built_at,
    }


def summarize_all_time() -> Dict[str, Any]:
    perf_root = os.path.join("data", "perf")
    seasons = []
    if os.path.isdir(perf_root):
        for d in os.listdir(perf_root):
            if d.startswith("season_"):
                try:
                    seasons.append(int(d.replace("season_", "")))
                except Exception:
                    pass
    seasons = sorted(seasons)
    totals_top_hits = totals_top_total = 0
    totals_all_hits = totals_all_total = 0
    start_date = ""
    end_date = ""
    days_tracked = 0
    for s in seasons:
        summary_path = os.path.join(perf_root, f"summary_season_{s}.json")
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        days_tracked += int(j.get("days_tracked") or 0)
        top = j.get("top_targets") or {}
        all_t = j.get("all_targets") or {}
        totals_top_hits += int(top.get("hits") or 0)
        totals_top_total += int(top.get("total") or 0)
        totals_all_hits += int(all_t.get("hits") or 0)
        totals_all_total += int(all_t.get("total") or 0)
        s_start = j.get("start_date") or ""
        s_end = j.get("end_date") or ""
        if s_start:
            start_date = s_start if not start_date else min(start_date, s_start)
        if s_end:
            end_date = s_end if not end_date else max(end_date, s_end)
    built_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return {
        "start_date": start_date,
        "end_date": end_date,
        "days_tracked": days_tracked,
        "top_targets": {
            "hits": totals_top_hits,
            "total": totals_top_total,
            "hit_rate": (totals_top_hits / totals_top_total) if totals_top_total else None,
        },
        "all_targets": {
            "hits": totals_all_hits,
            "total": totals_all_total,
            "hit_rate": (totals_all_hits / totals_all_total) if totals_all_total else None,
        },
        "last_updated_utc": built_at,
    }


def build_index() -> Dict[str, Any]:
    perf_root = os.path.join("data", "perf")
    seasons = []
    latest_by_season = {}
    if os.path.isdir(perf_root):
        for d in os.listdir(perf_root):
            if not d.startswith("season_"):
                continue
            try:
                season = int(d.replace("season_", ""))
            except Exception:
                continue
            seasons.append(season)
            daily_files = collect_daily_files(os.path.join(perf_root, d))
            latest = ""
            if daily_files:
                latest = os.path.splitext(os.path.basename(daily_files[-1]))[0]
            if latest:
                latest_by_season[str(season)] = latest
    seasons = sorted(seasons)
    built_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    paths = {"summary_all_time": "data/perf/summary_all_time.json"}
    for s in seasons:
        paths[f"summary_season_{s}"] = f"data/perf/summary_season_{s}.json"
    return {
        "last_updated_utc": built_at,
        "seasons": seasons,
        "latest_by_season": latest_by_season,
        "paths": paths,
    }


def write_legs_csv(snapshot_path: str, post: Dict[str, Any], out_path: str, asof: str, season_end_year: int) -> None:
    snapshot = load_snapshot(snapshot_path)
    if not snapshot:
        return

    hits = post.get("hits") or []
    misses = post.get("misses") or post.get("closest_misses") or []
    top_targets = post.get("top_targets") or []

    def key(item: Dict[str, Any]) -> str:
        return "|".join([
            str(item.get("player_name") or "").lower().strip(),
            str(item.get("team_abbrev") or "").upper().strip(),
            str(item.get("opp") or "").upper().strip(),
            str(item.get("stat") or "").lower().strip(),
            str(item.get("threshold") or "").strip(),
        ])

    status_map = {}
    for item in hits:
        status_map[key(item)] = ("HIT", item.get("actual"), item.get("reached"))
    for item in misses:
        if key(item) not in status_map:
            status_map[key(item)] = ("MISSED", item.get("actual"), item.get("reached"))
    for item in top_targets:
        if key(item) not in status_map:
            status = "HIT" if item.get("hit") is True else ""
            status_map[key(item)] = (status, item.get("actual"), item.get("reached"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "date","season_end_year","player_name","team_abbrev","opp_abbrev",
            "stat","threshold","status","actual","reached","source"
        ])
        for item in snapshot:
            k = key(item)
            status, actual, reached = status_map.get(k, ("", "", ""))
            w.writerow([
                asof,
                season_end_year,
                item.get("player_name", ""),
                item.get("team_abbrev", ""),
                item.get("opp", ""),
                item.get("stat", ""),
                item.get("threshold", ""),
                status,
                actual if actual is not None else "",
                reached if reached is not None else "",
                os.path.basename(snapshot_path),
            ])


def main() -> None:
    post = load_json("data/targets_postmortem.json")
    if not post:
        print("[perf] No targets_postmortem.json found; exiting.")
        return

    asof = parse_asof_date(post)
    try:
        asof_date = dt.datetime.strptime(asof, "%Y-%m-%d").date()
    except Exception:
        asof_date = dt.datetime.utcnow().date()
        asof = asof_date.isoformat()

    season_end_year = int(post.get("season_end_year") or season_end_year_for_date(asof_date))
    daily_payload = build_daily_payload(post, asof, season_end_year)

    daily_path = os.path.join("data", "perf", f"season_{season_end_year}", "daily", f"{asof}.json")
    write_json_if_changed(daily_path, daily_payload)

    snapshot_path = find_snapshot_for_date(asof)
    if snapshot_path:
        legs_path = os.path.join("data", "perf", f"season_{season_end_year}", "legs", f"{asof}.csv")
        write_legs_csv(snapshot_path, post, legs_path, asof, season_end_year)

    # summaries + index
    season_summary = summarize_season(season_end_year)
    write_json_if_changed(os.path.join("data", "perf", f"summary_season_{season_end_year}.json"), season_summary)
    all_time = summarize_all_time()
    write_json_if_changed(os.path.join("data", "perf", "summary_all_time.json"), all_time)
    index = build_index()
    write_json_if_changed(os.path.join("data", "perf", "index.json"), index)

    print(f"[perf] date={asof} season={season_end_year} daily={daily_path}")


if __name__ == "__main__":
    main()
