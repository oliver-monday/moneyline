#!/usr/bin/env python3
"""
ESPN NBA daily ingest.

Pulls:
  • Yesterday's FINAL scores
  • Today's games with pre-game MONEYLINE odds (if available)

Writes / upserts into nba_master.csv with the fixed schema:

  game_id
  game_date
  home_team_name
  home_team_abbrev
  home_score
  home_ml
  home_spread
  away_team_name
  away_team_abbrev
  away_score
  away_ml
  away_spread
  venue_city
  venue_state
"""

import argparse
import datetime as dt
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# --------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------

# ✅ This is the GOOD, stable ESPN endpoint
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/604.1"
)

# Canonical column order for nba_master.csv
MASTER_COLUMNS = [
    "game_id",
    "game_date",
    "home_team_name",
    "home_team_abbrev",
    "home_score",
    "home_ml",
    "home_spread",
    "away_team_name",
    "away_team_abbrev",
    "away_score",
    "away_ml",
    "away_spread",
    "venue_city",
    "venue_state",
]


# --------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------

def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


# --------------------------------------------------------------------
# FETCH SCOREBOARD JSON (scores + teams + venue)
# --------------------------------------------------------------------

def fetch_scoreboard(date_obj: dt.date) -> Dict[str, Any]:
    dates_str = date_obj.strftime("%Y%m%d")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    resp = requests.get(SCOREBOARD_URL, params={"dates": dates_str}, headers=headers)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------
# FETCH MONEYLINE ODDS FOR A GAME
# --------------------------------------------------------------------

def fetch_moneylines_for_game(game_id: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (home_ml, away_ml) from ESPN's odds chain.
    If anything fails, returns (None, None).
    """
    try:
        event_url = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}"
        ev = requests.get(event_url).json()

        comp_ref = (ev.get("competitions") or [{}])[0].get("$ref")
        if not comp_ref:
            return None, None

        comp = requests.get(comp_ref).json()

        odds_ref = (comp.get("odds") or {}).get("$ref")
        if not odds_ref:
            return None, None

        odds_data = requests.get(odds_ref).json()
        items = odds_data.get("items", [])
        if not items:
            return None, None

        # Prefer DraftKings if present
        chosen = None
        for o in items:
            prov = (o.get("provider") or {}).get("name", "")
            if prov and prov.lower() == "draftkings":
                chosen = o
                break
        if chosen is None:
            chosen = items[0]

        def to_int_safe(x):
            try:
                return int(x)
            except Exception:
                return None

        home_ml = to_int_safe((chosen.get("homeTeamOdds") or {}).get("moneyLine"))
        away_ml = to_int_safe((chosen.get("awayTeamOdds") or {}).get("moneyLine"))
        return home_ml, away_ml

    except Exception:
        # Any failure → no odds
        return None, None


# --------------------------------------------------------------------
# PARSE SCOREBOARD → CANONICAL ROWS
# --------------------------------------------------------------------

def parse_scoreboard(date_obj: dt.date, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = data.get("events", [])
    rows: List[Dict[str, Any]] = []

    for ev in events:
        game_id = ev.get("id")
        comps = ev.get("competitions", [])
        if not game_id or not comps:
            continue

        comp = comps[0]

        # Identify home/away competitors
        home = None
        away = None
        for t in comp.get("competitors", []):
            if t.get("homeAway") == "home":
                home = t
            else:
                away = t

        if not home or not away:
            continue

        def parse_team(t: Dict[str, Any]) -> Dict[str, Any]:
            team_info = t.get("team", {}) or {}
            return {
                "id": team_info.get("id"),
                "name": team_info.get("displayName"),
                "abbrev": team_info.get("abbreviation"),
                "score": safe_int(t.get("score")),
            }

        home_info = parse_team(home)
        away_info = parse_team(away)

        # Venue
        venue = comp.get("venue") or {}
        venue_addr = venue.get("address") or {}
        venue_city = venue_addr.get("city")
        venue_state = venue_addr.get("state")

        # Build row in FINAL SCHEMA (spreads will be filled or left None)
        row = {
            "game_id": str(game_id),
            "game_date": date_obj.strftime("%Y-%m-%d"),

            "home_team_name": home_info["name"],
            "home_team_abbrev": home_info["abbrev"],
            "home_score": home_info["score"],
            "home_ml": None,        # will be filled later for today's games
            "home_spread": None,    # we are not pulling spreads from ESPN (yet)

            "away_team_name": away_info["name"],
            "away_team_abbrev": away_info["abbrev"],
            "away_score": away_info["score"],
            "away_ml": None,        # will be filled later for today's games
            "away_spread": None,    # we are not pulling spreads from ESPN (yet)

            "venue_city": venue_city,
            "venue_state": venue_state,
        }

        rows.append(row)

    return rows


# --------------------------------------------------------------------
# UPSERT INTO MASTER CSV (STRICT SCHEMA)
# --------------------------------------------------------------------

def upsert_rows(df_new: pd.DataFrame, csv_path: str):
    # Enforce schema on new data
    for col in MASTER_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[MASTER_COLUMNS].copy()
    df_new["game_id"] = df_new["game_id"].astype(str)

    # Load or create empty master
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, dtype={"game_id": str})
        # If old file has extra columns, we drop them now and normalize
        for col in MASTER_COLUMNS:
            if col not in df_old.columns:
                df_old[col] = None
        df_old = df_old[MASTER_COLUMNS].copy()
    else:
        df_old = pd.DataFrame(columns=MASTER_COLUMNS)

    # Drop any existing rows with the same game_ids
    df_old = df_old[~df_old["game_id"].isin(df_new["game_id"])]

    merged = pd.concat([df_old, df_new], ignore_index=True)

    merged.to_csv(csv_path, index=False)
    print(f"Upserted {len(df_new)} rows → {csv_path} (total now {len(merged)})")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="nba_master.csv",
        help="Path to season-long master CSV (default: nba_master.csv)",
    )
    args = parser.parse_args()

    # Ingest strategy: yesterday + today
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    dates = [yesterday, today]

    all_rows: List[Dict[str, Any]] = []

    for d in dates:
        print(f"Fetching ESPN scoreboard for {d}...")
        try:
            data = fetch_scoreboard(d)
        except requests.HTTPError as e:
            print(f"Error fetching {d}: {e}")
            continue

        rows = parse_scoreboard(d, data)
        if not rows:
            print(f"No games found for {d}, skipping.")
            continue

        # Today's games → attach moneyline odds (home_ml / away_ml)
        if d == today:
            for r in rows:
                gid = r["game_id"]
                print(f"Fetching odds for game {gid} ...")
                home_ml, away_ml = fetch_moneylines_for_game(gid)
                r["home_ml"] = home_ml
                r["away_ml"] = away_ml
                # Spreads remain None for now (per our plan)

        # Yesterday's games → scores only; odds fields remain None
        all_rows.extend(rows)

    if not all_rows:
        print("No rows to ingest.")
        return

    df_new = pd.DataFrame(all_rows)
    upsert_rows(df_new, args.out)


if __name__ == "__main__":
    main()