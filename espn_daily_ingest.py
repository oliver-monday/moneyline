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
  home_injuries
  away_injuries
"""
# --------------------------------------------------------------------
# ESPN → Rotowire team code harmonization
# --------------------------------------------------------------------
# ESPN and Rotowire don't always use the exact same abbreviations.
# This map normalizes ESPN's codes to what Rotowire uses in data-team="XYZ".
ESPN_TO_ROTO = {
    # Common mismatches
    "WSH": "WAS",
    "NO": "NOP",
    "NY": "NYK",
    "GS": "GSW",
    "SA": "SAS",
    "UTAH": "UTA",
    # Some sites use these alternates; keep them safe
    "PHO": "PHX",
    "CHA": "CHA",  # identity, included for clarity
}

def to_roto_code(code: str) -> str:
    """
    Normalize ESPN team abbreviation to Rotowire's data-team code.
    Falls back to the input code if no mapping is needed.
    """
    c = (code or "").upper()
    return ESPN_TO_ROTO.get(c, c)

import argparse
import datetime as dt
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import re
import requests


def _last_master_game_date(csv_path: str) -> Optional[dt.date]:
    """Return the max game_date in an existing master CSV, or None."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["game_date"])
    except Exception:
        return None
    if "game_date" not in df.columns or df.empty:
        return None
    s = pd.to_datetime(df["game_date"], errors="coerce").dropna()
    if s.empty:
        return None
    return s.max().date()


def _daterange(start: dt.date, end: dt.date):
    """Inclusive date range generator."""
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


# --------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------

# ✅ This is the GOOD, stable ESPN endpoint
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# Rotowire starting lineups / injuries page
ROTO_LINEUPS_URL = "https://www.rotowire.com/basketball/nba-lineups.php"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/604.1"
)

# Canonical column order for nba_master.csv
MASTER_COLUMNS = [
    "game_id",
    "game_date",
    "game_time_utc",
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
    "home_injuries",   # NEW
    "away_injuries",   # NEW
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


import re
import requests

# --------------------------------------------------------------------
# ROTOWIRE INJURY SCRAPER (NO EXTRA DEPENDENCIES)
# --------------------------------------------------------------------

ROTOWIRE_URL = "https://www.rotowire.com/basketball/nba-lineups.php"

def fetch_rotowire_injuries() -> Dict[str, str]:
    """
    Scrape Rotowire's NBA lineups page and return a mapping:

        { "BOS": "J. Tatum (Out); J. Brown (Questionable)", ... }

    Implementation notes:
      - Best-effort only: if anything breaks, we just return {}.
      - Uses regex on the HTML (no bs4) based on the current structure:
          * Each team block has a button:
                data-team="POR" ... >On/Off Court Stats</button>
          * Followed by a "MAY NOT PLAY" section with
                <li class="lineup__player has-injury-status"> ... </li>
          * Each injury row contains:
                <a ...>Player Name</a>
                <span class="lineup__inj">Out|Questionable|Doubtful|Rest|OFS</span>
    """

    try:
        resp = requests.get(
            ROTOWIRE_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"Rotowire HTTP {resp.status_code}, skipping injuries.")
            return {}
        html = resp.text
    except Exception as e:
        print(f"Rotowire fetch error: {e}")
        return {}

    injuries_per_team: Dict[str, List[str]] = {}

    # 1) For each team, grab the block between the On/Off button and the end of that
    #    MAY NOT PLAY list.
    #
    # Example pattern in the HTML:
    #   data-team="POR" ...>On/Off Court Stats</button>
    #       ...
    #       <li class="lineup__title is-middle">MAY NOT PLAY</li>
    #       <li class="lineup__player is-pct-play-0 has-injury-status" ...> ... </li>
    #       ...
    #   </ul>
    team_block_pattern = re.compile(
        r'data-team="([A-Z]{2,3})"[^>]*>On/Off Court Stats</button>(.*?)</ul>',
        re.DOTALL,
    )

    matches = team_block_pattern.findall(html)

    for team, block in matches:
        team = team.upper()
        players: List[str] = []

        # 2) Inside that block, each injured player row looks like:
        #    <li class="lineup__player ... has-injury-status" ...>
        #        ... <a ...>Player Name</a> ... <span class="lineup__inj">Out</span>
        #    </li>
        for player_html, status_html in re.findall(
            r'<li[^>]*has-injury-status[^>]*>.*?'
            r'<a[^>]*>(.*?)</a>.*?'
            r'<span class="lineup__inj">(.*?)</span>',
            block,
            flags=re.DOTALL,
        ):
            # Strip any nested tags from player name / status
            name = re.sub(r"<.*?>", "", player_html).strip()
            status = re.sub(r"<.*?>", "", status_html).strip()

            if not name or not status:
                continue

            status_lower = status.lower()

            # Keep only "impactful" statuses
            if (
                status_lower in ("out", "doubtful", "questionable")
                or "rest" in status_lower
                or status.upper() == "OFS"      # Rotowire's "offseason" flag
            ):
                players.append(f"{name} ({status})")

        if players:
            injuries_per_team[team] = players

    # 3) Convert to { TEAM: "Player1 (Status); Player2 (Status)" }
    compact: Dict[str, str] = {
        team: "; ".join(players) for team, players in injuries_per_team.items()
    }

    print(f"Rotowire injuries: got {len(compact)} teams with issues.")
    return compact


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
            "game_time_utc": comp.get("date") or ev.get("date"),

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

            # Injuries will be attached later for today's games
            "home_injuries": None,
            "away_injuries": None,
        }

        rows.append(row)

    return rows


# --------------------------------------------------------------------
# UPSERT INTO MASTER CSV (STRICT SCHEMA)
# --------------------------------------------------------------------

def _preserve_existing_odds(df_new: pd.DataFrame, df_old_subset: pd.DataFrame) -> pd.DataFrame:
    """
    For overlapping game_ids, keep existing moneyline odds if the new ingest
    has missing/zero values for that side.
    """
    if df_old_subset.empty:
        return df_new

    merged = df_new.merge(
        df_old_subset[["game_id", "home_ml", "away_ml"]],
        on="game_id",
        how="left",
        suffixes=("", "_old"),
    )

    for side in ["home", "away"]:
        ml_col = f"{side}_ml"
        old_col = f"{ml_col}_old"
        if ml_col not in merged.columns or old_col not in merged.columns:
            continue

        # New is NaN/0, old is non-null and non-zero → keep old value
        mask = merged[ml_col].isna() | (merged[ml_col] == 0)
        mask &= merged[old_col].notna() & (merged[old_col] != 0)

        merged.loc[mask, ml_col] = merged.loc[mask, old_col]
        merged.drop(columns=[old_col], inplace=True)

    return merged


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

    # ---- PATCH: preserve existing odds for overlapping games ----
    if not df_old.empty:
        overlap_old = df_old[df_old["game_id"].isin(df_new["game_id"])].copy()
        if not overlap_old.empty:
            df_new = _preserve_existing_odds(df_new, overlap_old)
    # -------------------------------------------------------------

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
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Disable backfilling missing dates between the last date in the master CSV and yesterday.",
    )
    parser.add_argument(
        "--max-backfill-days",
        type=int,
        default=30,
        help="When backfilling, limit to at most this many days of history (default: 30).",
    )
    args = parser.parse_args()

    # Ingest strategy: always fetch yesterday + today.
    # Backfill strategy (default): look for *missing* dates in a trailing window
    # (max-backfill-days) ending yesterday. This fixes "holes" caused by a master
    # file that kept getting reset while still containing today's date.
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    dates = {yesterday, today}

    existing_dates = set()
    if os.path.exists(args.out):
        try:
            tmp = pd.read_csv(args.out, usecols=["game_date"])
            s = pd.to_datetime(tmp["game_date"], errors="coerce").dropna()
            existing_dates = set(s.dt.date.unique())
        except Exception:
            existing_dates = set()

    if not args.no_backfill and args.max_backfill_days and args.max_backfill_days > 0:
        window_end = yesterday
        window_start = window_end - dt.timedelta(days=args.max_backfill_days - 1)

        for d in _daterange(window_start, window_end):
            dates.add(d)

    # Keep deterministic ordering for logs
    dates = sorted(dates)

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

    # ---------------------------------------------------------
    # Attach injuries for today's games (best effort)
    # ---------------------------------------------------------
    try:
        injuries_map = fetch_rotowire_injuries()
    except Exception as e:
        print(f"Warning: unexpected error in injury scraping: {e}")
        injuries_map = {}

    if injuries_map:
        # Ensure columns exist in df_new
        if "home_injuries" not in df_new.columns:
            df_new["home_injuries"] = None
        if "away_injuries" not in df_new.columns:
            df_new["away_injuries"] = None

        today_str = today.strftime("%Y-%m-%d")
        mask_today = df_new["game_date"] == today_str

        for idx in df_new[mask_today].index:
            raw_home = df_new.at[idx, "home_team_abbrev"]
            raw_away = df_new.at[idx, "away_team_abbrev"]

            home_code = to_roto_code(raw_home)
            away_code = to_roto_code(raw_away)

            df_new.at[idx, "home_injuries"] = injuries_map.get(home_code)
            df_new.at[idx, "away_injuries"] = injuries_map.get(away_code)

    upsert_rows(df_new, args.out)


if __name__ == "__main__":
    main()
