#!/usr/bin/env python3
"""
Fetch Rotowire injuries and update data/injuries_today.json + logs/injury_log.csv.
Best-effort only: if scrape fails or yields empty, do not overwrite existing JSON.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

ROTOWIRE_URL = "https://www.rotowire.com/basketball/nba-lineups.php"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _short_status(s: str) -> str:
    t = (s or "").strip().lower()
    if not t:
        return ""
    if "out for season" in t or "ofs" in t:
        return "OFS"
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


def fetch_rotowire_html() -> str | None:
    try:
        resp = requests.get(
            ROTOWIRE_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
    except Exception as exc:
        print(f"[injuries] Rotowire fetch error: {exc}")
        return None
    if resp.status_code != 200:
        print(f"[injuries] Rotowire HTTP {resp.status_code}, skipping.")
        return None
    return resp.text


def parse_rotowire_injuries(html: str) -> Dict[str, List[Dict[str, str]]]:
    injuries_today: Dict[str, List[Dict[str, str]]] = {}

    team_block_pattern = re.compile(
        r'data-team="([A-Z]{2,3})"[^>]*>On/Off Court Stats</button>(.*?)</ul>',
        re.DOTALL,
    )
    matches = team_block_pattern.findall(html)

    for team, block in matches:
        team = team.upper()
        entries: List[Dict[str, str]] = []

        for player_html, status_html in re.findall(
            r'<li[^>]*has-injury-status[^>]*>.*?'
            r'<a[^>]*>(.*?)</a>.*?'
            r'<span class="lineup__inj">(.*?)</span>',
            block,
            flags=re.DOTALL,
        ):
            name = re.sub(r"<.*?>", "", player_html).strip()
            status_raw = re.sub(r"<.*?>", "", status_html).strip()
            if not name or not status_raw:
                continue

            status_lower = status_raw.lower()
            if not (
                status_lower in ("out", "doubtful", "questionable")
                or "rest" in status_lower
                or "out for season" in status_lower
                or status_raw.upper() == "OFS"
            ):
                continue

            status = _short_status(status_raw)
            details = f"{name} ({status_raw})"
            entries.append({"name": name, "status": status, "details": details})

        if entries:
            deduped = {(e["name"].strip(), e["status"].strip(), e["details"].strip()): e for e in entries}
            injuries_today[team] = list(deduped.values())

    return injuries_today


def append_injury_log(injuries_today: Dict[str, List[Dict[str, str]]], asof_date: str) -> None:
    if not injuries_today:
        return

    rows = []
    for team, entries in injuries_today.items():
        for e in entries:
            rows.append({
                "asof_date": asof_date,
                "team_abbrev": team,
                "player_name": e.get("name", ""),
                "status": e.get("status", ""),
                "details": e.get("details", ""),
                "source": "rotowire",
            })

    if not rows:
        return

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "injury_log.csv"

    new_df = pd.DataFrame(rows)
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


def write_injuries_json(injuries_today: Dict[str, List[Dict[str, str]]]) -> bool:
    if not injuries_today:
        print("[injuries] Empty injuries set; not overwriting injuries_today.json.")
        return False

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "injuries_today.json"
    tmp_path = data_dir / "injuries_today.json.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(injuries_today, f, indent=2)
    os.replace(tmp_path, out_path)
    return True


def main() -> int:
    html = fetch_rotowire_html()
    if not html:
        return 0

    injuries_today = parse_rotowire_injuries(html)
    if not injuries_today:
        print("[injuries] Parsed 0 injuries; not overwriting injuries_today.json.")
        return 0

    wrote = write_injuries_json(injuries_today)
    if wrote:
        asof_date = dt.date.today().strftime("%Y-%m-%d")
        append_injury_log(injuries_today, asof_date)
        print(f"[injuries] wrote injuries_today.json teams={len(injuries_today)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
