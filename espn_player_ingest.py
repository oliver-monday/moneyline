#!/usr/bin/env python3
"""
ESPN NBA player props ingest (PTS/REB/AST + minutes) for a whitelist.

Outputs (CSV):
  - player_dim.csv (auto-built ESPN athlete_id map)
  - player_game_log.csv (one row per player per game)
  - player_unresolved.csv (whitelist names not yet matched to an ESPN athlete_id)

Designed to be "seamless" with your existing moneyline workflow:
  - game_id is ESPN event id (same key used in nba_master.csv)
  - can backfill from nba_master.csv OR from a date range via scoreboard calls
  - daily mode for yesterday's finals

Data source:
  - ESPN site API scoreboard (event ids)
  - ESPN site API summary endpoint (boxscore player stats)

Usage examples:
  # Backfill from nba_master for the current season (derived from today's date)
  python espn_player_ingest.py --mode backfill --master nba_master.csv

  # Backfill a specific season end year (e.g., NBA_2026 style)
  python espn_player_ingest.py --mode backfill --master nba_master.csv --season-end-year 2026

  # Ingest a single date (YYYY-MM-DD)
  python espn_player_ingest.py --mode date --date 2025-12-18

  # Ingest yesterday
  python espn_player_ingest.py --mode yesterday
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "Mozilla/5.0 (compatible; moneyline/1.0; +https://oliver-monday.github.io/moneyline/)"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def season_end_year_for_date(d: dt.date) -> int:
    """NBA season end year (Basketball-Reference convention)."""
    return d.year + 1 if d.month >= 10 else d.year

def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def normalize_name(s: str) -> str:
    """
    Normalize player names for robust matching:
      - strip diacritics
      - lowercase
      - remove punctuation
      - collapse whitespace
      - normalize suffixes (jr, sr, ii, iii, iv, v)
    """
    if s is None:
        return ""
    s = s.strip()

    # Unicode normalize + remove diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.lower()

    # Normalize curly apostrophes and similar
    s = s.replace("’", "'").replace("`", "'")

    # Remove punctuation except spaces
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = s.replace("'", "")  # drop apostrophes

    # Normalize common suffix tokens
    s = re.sub(r"\b(jr|sr)\b\.?", r"\1", s)
    s = re.sub(r"\b(ii|iii|iv|v)\b\.?", lambda m: m.group(1), s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def minutes_to_decimal(min_raw: str) -> float:
    """
    ESPN often returns MIN as "MM:SS" (or sometimes "MM").
    Returns decimal minutes.
    """
    if min_raw is None:
        return 0.0
    s = str(min_raw).strip()
    if s in ("", "--", "DNP", "NA", "N/A"):
        return 0.0
    if ":" in s:
        mm, ss = s.split(":", 1)
        try:
            mm_i = int(mm)
            ss_i = int(ss)
            return mm_i + ss_i / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

def build_session() -> requests.Session:
    """Requests session with retries for transient ESPN failures."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

def fetch_scoreboard(session: requests.Session, date_obj: dt.date) -> Dict[str, Any]:
    dates_str = date_obj.strftime("%Y%m%d")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    resp = session.get(SCOREBOARD_URL, params={"dates": dates_str}, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_summary(session: requests.Session, event_id: str) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    resp = session.get(SUMMARY_URL, params={"event": str(event_id)}, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def extract_event_ids_from_scoreboard(scoreboard_json: Dict[str, Any], finals_only: bool = True) -> List[str]:
    out: List[str] = []
    events = scoreboard_json.get("events", []) or []
    for ev in events:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        if not finals_only:
            out.append(str(ev_id))
            continue
        # status.type.name often equals "STATUS_FINAL" or status.type.state == "post"
        status = (ev.get("status") or {}).get("type") or {}
        state = status.get("state")  # "post" for final
        name = status.get("name")
        if state == "post" or (isinstance(name, str) and "FINAL" in name.upper()):
            out.append(str(ev_id))
    return out

def infer_game_meta_from_summary(summary_json: Dict[str, Any]) -> Tuple[dt.date, Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - game_date (local date from ISO timestamp)
      - team_map: team_id -> {abbrev, homeAway, score}
    """
    header = summary_json.get("header") or {}
    comps = header.get("competitions") or []
    if not comps:
        # fallback to today; should be rare
        return dt.date.today(), {}

    comp0 = comps[0]
    date_iso = comp0.get("date") or header.get("gameDate") or ""
    try:
        game_dt = dt.datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
        game_date = game_dt.date()
    except Exception:
        game_date = dt.date.today()

    team_map: Dict[str, Dict[str, Any]] = {}
    competitors = comp0.get("competitors") or []
    for c in competitors:
        team = c.get("team") or {}
        tid = str(team.get("id") or "")
        if not tid:
            continue
        team_map[tid] = {
            "team_id": tid,
            "abbrev": team.get("abbreviation") or team.get("shortDisplayName") or "",
            "homeAway": c.get("homeAway") or "",
            "score": c.get("score"),
        }
    return game_date, team_map

def parse_boxscore_players(summary_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse ESPN summary boxscore JSON into player stat rows with:
      team_id, team_abbrev, athlete_id, athlete_name, started, minutes_raw, pts, reb, ast, dnp
    """
    rows: List[Dict[str, Any]] = []

    box = summary_json.get("boxscore") or {}
    players_blocks = box.get("players") or []  # one per team

    for team_block in players_blocks:
        team = team_block.get("team") or {}
        team_id = str(team.get("id") or "")
        team_abbrev = team.get("abbreviation") or team.get("shortDisplayName") or ""
        stats_groups = team_block.get("statistics") or []

        for grp in stats_groups:
            grp_name = (grp.get("name") or "").lower()
            is_starters = "starter" in grp_name  # "starters"
            labels = grp.get("labels") or []
            athletes = grp.get("athletes") or []

            # Build label index maps (exact + normalized)
            label_to_idx = {str(lab).upper(): i for i, lab in enumerate(labels)}
            label_to_idx_norm = {}
            for i, lab in enumerate(labels):
                key = "".join(ch for ch in str(lab).upper() if ch.isalnum())
                if key and key not in label_to_idx_norm:
                    label_to_idx_norm[key] = i

            def get_stat(stats_list: List[str], key: str) -> Optional[str]:
                idx = label_to_idx.get(key.upper())
                if idx is None:
                    norm = "".join(ch for ch in str(key).upper() if ch.isalnum())
                    idx = label_to_idx_norm.get(norm)
                    if idx is None:
                        return None
                if idx >= len(stats_list):
                    return None
                return stats_list[idx]

            for a in athletes:
                athlete = a.get("athlete") or {}
                athlete_id = str(athlete.get("id") or "")
                athlete_name = athlete.get("displayName") or athlete.get("shortName") or ""
                stats_list = a.get("stats") or []

                min_raw = get_stat(stats_list, "MIN")
                pts_raw = get_stat(stats_list, "PTS")
                reb_raw = get_stat(stats_list, "REB")
                ast_raw = get_stat(stats_list, "AST")
                tpm_raw = None
                for key in ("3PM", "3PT", "FG3M", "3FGM", "3PTM", "3P", "3PA_MADE"):
                    tpm_raw = get_stat(stats_list, key)
                    if tpm_raw is not None:
                        break
                if tpm_raw is None:
                    for lab, idx in label_to_idx_norm.items():
                        if lab.startswith(("3PT", "3PM", "FG3M")):
                            if idx < len(stats_list):
                                tpm_raw = stats_list[idx]
                                break

                # DNP signal: ESPN sometimes includes "didNotPlay" or MIN="--"
                did_not_play = bool(a.get("didNotPlay") or a.get("didNotDress") or a.get("notActive") or False)
                if min_raw is None:
                    # some feeds use "MINUTES"
                    min_raw = get_stat(stats_list, "MINUTES")

                minutes = minutes_to_decimal(min_raw)
                dnp = 1 if did_not_play or (str(min_raw).strip() in ("", "--")) else 0

                def to_int(x: Optional[str]) -> int:
                    if x is None:
                        return 0
                    s = str(x).strip()
                    if s in ("", "--"):
                        return 0
                    try:
                        return int(float(s))
                    except ValueError:
                        return 0

                def parse_tpm(x: Optional[str]) -> int:
                    if x is None:
                        return 0
                    s = str(x).strip()
                    if s in ("", "--", "—", "–"):
                        return 0
                    if "-" in s:
                        left = s.split("-", 1)[0].strip()
                        try:
                            return int(float(left))
                        except ValueError:
                            return 0
                    try:
                        return int(float(s))
                    except ValueError:
                        return 0

                row = {
                    "team_id": team_id,
                    "team_abbrev": team_abbrev,
                    "athlete_id": athlete_id,
                    "athlete_name": athlete_name,
                    "started": 1 if is_starters else 0,
                    "minutes_raw": min_raw if min_raw is not None else "",
                    "minutes": round(minutes, 4),
                    "pts": to_int(pts_raw),
                    "reb": to_int(reb_raw),
                    "ast": to_int(ast_raw),
                    "tpm": parse_tpm(tpm_raw),
                    "dnp": dnp,
                }
                # Filter out empty athlete ids (rare)
                if athlete_id:
                    rows.append(row)

    return rows

# --------------------------------------------------------------------
# Whitelist matching + persistence
# --------------------------------------------------------------------

def load_whitelist(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "player_name" not in df.columns or "team_abbr" not in df.columns:
        raise ValueError("Whitelist must have columns: player_name, team_abbr (and optional team_abbr_alt, active).")
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name_norm"] = df["player_name"].map(normalize_name)
    df["team_abbr"] = df["team_abbr"].astype(str).str.strip()
    if "team_abbr_alt" not in df.columns:
        df["team_abbr_alt"] = ""
    else:
        df["team_abbr_alt"] = df["team_abbr_alt"].fillna("").astype(str).str.strip()
    return df

def apply_whitelist_active_to_dim(dim_df: pd.DataFrame, whitelist_df: pd.DataFrame) -> pd.DataFrame:
    if dim_df.empty or "active" not in whitelist_df.columns:
        return dim_df
    if "player_name_norm" not in dim_df.columns or "last_seen_team_abbrev" not in dim_df.columns:
        return dim_df

    wl = whitelist_df.copy()
    wl["active"] = pd.to_numeric(wl["active"], errors="coerce")
    active_map: Dict[Tuple[str, str], int] = {}
    for _, r in wl.iterrows():
        name_norm = r.get("player_name_norm", "")
        if not name_norm or pd.isna(r["active"]):
            continue
        for team in [r.get("team_abbr", ""), r.get("team_abbr_alt", "")]:
            team_norm = str(team or "").upper().strip()
            if team_norm:
                active_map[(name_norm, team_norm)] = int(r["active"])

    out = dim_df.copy()
    if "active" not in out.columns:
        out["active"] = ""
    def _pick_active(row):
        key = (row["player_name_norm"], str(row["last_seen_team_abbrev"] or "").upper().strip())
        return active_map.get(key, row.get("active", ""))
    out["active"] = out.apply(_pick_active, axis=1)
    out["active"] = pd.to_numeric(out["active"], errors="coerce").fillna(1).astype(int)
    return out

def build_whitelist_lookup(df_wl: pd.DataFrame) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Map normalized player name -> list of (player_name, team_abbr, team_abbr_alt).
    We keep a list because you could (rarely) have duplicates.
    """
    lookup: Dict[str, List[Tuple[str, str, str]]] = {}
    for _, r in df_wl.iterrows():
        lookup.setdefault(r["player_name_norm"], []).append((r["player_name"], r["team_abbr"], r["team_abbr_alt"]))
    return lookup

def load_dim(path: str) -> pd.DataFrame:
    if not pd.io.common.file_exists(path):
        return pd.DataFrame(columns=[
            "player_id","player_name_canonical","player_name_norm",
            "last_seen_team_abbrev","first_seen_date","last_seen_date"
        ])
    df = pd.read_csv(path)
    for c in ["player_id","player_name_canonical","player_name_norm","last_seen_team_abbrev","first_seen_date","last_seen_date"]:
        if c not in df.columns:
            df[c] = ""
    return df

def upsert_dim(df_dim: pd.DataFrame, updates: List[Dict[str, Any]]) -> pd.DataFrame:
    if not updates:
        return df_dim
    df_up = pd.DataFrame(updates)
    if df_dim.empty:
        out = df_up.copy()
    else:
        out = pd.concat([df_dim, df_up], ignore_index=True)

    # Keep earliest first_seen_date per player_id, latest last_seen_date, latest team/name
    out["first_seen_date"] = pd.to_datetime(out["first_seen_date"], errors="coerce")
    out["last_seen_date"] = pd.to_datetime(out["last_seen_date"], errors="coerce")

    out = out.sort_values(["player_id","last_seen_date"]).drop_duplicates(subset=["player_id"], keep="last")
    # fill first_seen_date from min
    mins = out.groupby("player_id")["first_seen_date"].transform("min")
    out["first_seen_date"] = mins.fillna(out["first_seen_date"])

    out["first_seen_date"] = out["first_seen_date"].dt.date.astype(str)
    out["last_seen_date"] = out["last_seen_date"].dt.date.astype(str)
    out = out.reset_index(drop=True)
    return out

def load_game_log(path: str) -> pd.DataFrame:
    if not pd.io.common.file_exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def upsert_game_log(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    if df_old is None or df_old.empty:
        merged = df_new.copy()
    else:
        merged = pd.concat([df_old, df_new], ignore_index=True)

    # ensure key columns exist
    for col in ["season_end_year","game_id","player_id"]:
        if col not in merged.columns:
            raise ValueError(f"Missing required column in merged game log: {col}")

    merged["ingested_at"] = merged["ingested_at"].fillna("")
    merged = merged.sort_values("ingested_at").drop_duplicates(
        subset=["season_end_year","game_id","player_id"], keep="last"
    ).reset_index(drop=True)
    return merged

def read_master_game_ids(master_path: str, season_end_year: int) -> List[str]:
    df = pd.read_csv(master_path)
    required = {"game_id","game_date","home_score","away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{master_path} is missing columns required for backfill: {sorted(missing)}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df = df[df["game_date"].notna()].copy()

    df["season_end_year"] = df["game_date"].map(season_end_year_for_date)
    df = df[df["season_end_year"] == season_end_year].copy()

    # completed games only (scores present)
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    game_ids = df["game_id"].dropna().astype(str).unique().tolist()
    return sorted(game_ids)

def infer_home_away(team_abbrev: str, team_map: Dict[str, Dict[str, Any]]) -> str:
    # team_map keyed by team_id; we might only have abbrev here
    for _, meta in team_map.items():
        if str(meta.get("abbrev","")).upper() == str(team_abbrev).upper():
            ha = meta.get("homeAway","")
            return "H" if ha == "home" else ("A" if ha == "away" else "")
    return ""

def opponent_for_team(team_abbrev: str, team_map: Dict[str, Dict[str, Any]]) -> Tuple[str,str]:
    """Return (opp_abbrev, opp_team_id) given team_abbrev, using the two competitors."""
    abbrs = [(tid, meta.get("abbrev","")) for tid, meta in team_map.items()]
    # locate current
    team_abbrev_u = str(team_abbrev).upper()
    team_tid = None
    for tid, ab in abbrs:
        if str(ab).upper() == team_abbrev_u:
            team_tid = tid
            break
    # pick the other
    for tid, ab in abbrs:
        if team_tid is None:
            # unknown; just return first other if any
            continue
        if tid != team_tid:
            return (ab or "", tid)
    # fallback
    if len(abbrs) == 2:
        return (abbrs[1][1] if str(abbrs[0][1]).upper()==team_abbrev_u else abbrs[0][1], abbrs[1][0])
    return ("","")

# --------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------

def ingest_events(
    session: requests.Session,
    event_ids: List[str],
    whitelist_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    sleep_s: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (df_new_game_log_rows, df_new_dim, df_match_audit)
    """
    wl_lookup = build_whitelist_lookup(whitelist_df)

    # Fast allowlist by resolved IDs if any exist
    resolved_ids = set(dim_df["player_id"].dropna().astype(str).tolist()) if not dim_df.empty else set()

    new_rows: List[Dict[str, Any]] = []
    dim_updates: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for i, event_id in enumerate(event_ids, start=1):
        try:
            summary = fetch_summary(session, event_id)
        except Exception as e:
            print(f"[WARN] event {event_id}: summary fetch failed: {e}")
            continue

        game_date, team_map = infer_game_meta_from_summary(summary)
        season_end = season_end_year_for_date(game_date)

        parsed = parse_boxscore_players(summary)
        if not parsed:
            print(f"[WARN] event {event_id}: no boxscore players parsed")
            continue

        for r in parsed:
            athlete_id = str(r["athlete_id"])
            athlete_name = r["athlete_name"]
            athlete_name_norm = normalize_name(athlete_name)
            team_abbrev = str(r["team_abbrev"] or "").upper().strip()

            # Match logic:
            matched = False
            matched_name = ""
            matched_team_hint = ""
            matched_team_alt = ""
            team_hint_ok = None

            if athlete_id in resolved_ids:
                matched = True
            elif athlete_name_norm in wl_lookup:
                # If team hint exists, prefer those matching team, but do not hard reject.
                candidates = wl_lookup[athlete_name_norm]
                # take first candidate (rarely multiple)
                matched_name, matched_team_hint, matched_team_alt = candidates[0]
                matched = True
                hints = {matched_team_hint.upper(), matched_team_alt.upper()} - {""}
                team_hint_ok = (team_abbrev in hints) if hints else None

            if not matched:
                continue

            # Build meta fields that aren't in boxscore player block
            home_away = infer_home_away(team_abbrev, team_map)
            opp_abbrev, opp_id = opponent_for_team(team_abbrev, team_map)

            # Dimension update
            dim_updates.append({
                "player_id": athlete_id,
                "player_name_canonical": athlete_name,
                "player_name_norm": athlete_name_norm,
                "last_seen_team_abbrev": team_abbrev,
                "first_seen_date": str(game_date),
                "last_seen_date": str(game_date),
            })

            # Game log row
            ingested_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            new_rows.append({
                "season_end_year": season_end,
                "game_id": str(event_id),
                "game_date": str(game_date),
                "team_abbrev": team_abbrev,
                "opp_abbrev": (opp_abbrev or "").upper(),
                "home_away": home_away,
                "player_id": athlete_id,
                "player_name": athlete_name,
                "started": int(r["started"]),
                "minutes": float(r["minutes"]),
                "minutes_raw": str(r["minutes_raw"]),
                "pts": int(r["pts"]),
                "reb": int(r["reb"]),
                "ast": int(r["ast"]),
                "dnp": int(r["dnp"]),
                "ingested_at": ingested_at,
                "team_hint_ok": "" if team_hint_ok is None else int(team_hint_ok),
            })

            audit_rows.append({
                "game_id": str(event_id),
                "player_id": athlete_id,
                "athlete_name": athlete_name,
                "team_abbrev": team_abbrev,
                "matched_by": "id" if athlete_id in resolved_ids else "name",
                "team_hint_ok": "" if team_hint_ok is None else int(team_hint_ok),
            })

        if sleep_s:
            time.sleep(sleep_s)

        if i % 25 == 0:
            print(f"Processed {i}/{len(event_ids)} events...")

    df_new = pd.DataFrame(new_rows)
    df_audit = pd.DataFrame(audit_rows)
    df_dim_new = upsert_dim(dim_df, dim_updates)

    return df_new, df_dim_new, df_audit

def write_unresolved(whitelist_df: pd.DataFrame, dim_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Report whitelist names that have not yet been resolved to an ESPN player_id.
    This is a practical "fix the typos" list.
    """
    resolved_norms = set(dim_df["player_name_norm"].dropna().astype(str).tolist()) if not dim_df.empty else set()
    df = whitelist_df.copy()
    df["resolved"] = df["player_name_norm"].isin(resolved_norms).astype(int)
    unresolved = df[df["resolved"] == 0][["team_abbr","team_abbr_alt","player_name"]].copy()
    unresolved.to_csv(out_path, index=False)
    return unresolved

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backfill","date","yesterday"], default="backfill",
                        help="backfill: use nba_master to enumerate completed games; date: ingest a single date via scoreboard; yesterday: ingest yesterday's finals")
    parser.add_argument("--master", default="nba_master.csv", help="nba_master.csv path (used in backfill mode)")
    parser.add_argument("--whitelist", default="player_whitelist.csv", help="Whitelist CSV path (names+teams)")
    parser.add_argument("--out", default="player_game_log.csv", help="Output player game log CSV path")
    parser.add_argument("--dim", default="player_dim.csv", help="Output player dim CSV path")
    parser.add_argument("--unresolved", default="player_unresolved.csv", help="Output unresolved whitelist CSV path")
    parser.add_argument("--audit", default="player_match_audit.csv", help="Output match audit CSV path (optional)")
    parser.add_argument("--season-end-year", type=int, default=None, help="Season end year to filter (e.g., 2026). Default: derived from today")
    parser.add_argument("--date", default=None, help="Date for mode=date in YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between summary calls (rate limiting)")

    args = parser.parse_args()

    # Determine season
    today = dt.date.today()
    season_end_year = args.season_end_year or season_end_year_for_date(today)

    # Load inputs/outputs
    wl = load_whitelist(args.whitelist)
    dim = load_dim(args.dim)
    df_old = load_game_log(args.out)

    session = build_session()

    event_ids: List[str] = []

    if args.mode == "backfill":
        if not pd.io.common.file_exists(args.master):
            raise SystemExit(f"Backfill mode requires --master file, but not found: {args.master}")
        event_ids = read_master_game_ids(args.master, season_end_year)
        print(f"Backfill: {len(event_ids)} completed games found in {args.master} for season_end_year={season_end_year}")

    elif args.mode == "date":
        if not args.date:
            raise SystemExit("mode=date requires --date YYYY-MM-DD")
        d = parse_date(args.date)
        sb = fetch_scoreboard(session, d)
        event_ids = extract_event_ids_from_scoreboard(sb, finals_only=True)
        print(f"Date ingest {d}: {len(event_ids)} FINAL events found")

    elif args.mode == "yesterday":
        d = today - dt.timedelta(days=1)
        sb = fetch_scoreboard(session, d)
        event_ids = extract_event_ids_from_scoreboard(sb, finals_only=True)
        print(f"Yesterday ingest {d}: {len(event_ids)} FINAL events found")

    if not event_ids:
        print("No events to ingest. Exiting.")
        return

    df_new, dim_new, df_audit = ingest_events(session, event_ids, wl, dim, sleep_s=args.sleep)
    dim_new = apply_whitelist_active_to_dim(dim_new, wl)

    if df_new.empty:
        print("No whitelist players matched in these events. Writing unresolved report and exiting.")
        unresolved = write_unresolved(wl, dim_new, args.unresolved)
        print(f"Unresolved players: {len(unresolved)} → {args.unresolved}")
        dim_new.to_csv(args.dim, index=False)
        return

    # Upsert game log
    df_merged = upsert_game_log(df_old, df_new)

    # Write outputs
    # Stable column order
    col_order = [
        "season_end_year","game_id","game_date","team_abbrev","opp_abbrev","home_away",
        "player_id","player_name","started","minutes","minutes_raw","pts","reb","ast","tpm","dnp",
        "team_hint_ok","ingested_at"
    ]
    for c in col_order:
        if c not in df_merged.columns:
            df_merged[c] = ""

    df_merged = df_merged[col_order].copy()
    df_merged.to_csv(args.out, index=False)
    dim_new.to_csv(args.dim, index=False)
    df_audit.to_csv(args.audit, index=False)

    unresolved = write_unresolved(wl, dim_new, args.unresolved)

    try:
        tpm_vals = pd.to_numeric(df_merged.get("tpm"), errors="coerce").fillna(0)
        tpm_nonzero = int((tpm_vals > 0).sum())
        sample = df_merged.loc[tpm_vals > 0].head(1)
        if not sample.empty:
            sample_name = sample["player_name"].iloc[0]
            sample_tpm = sample["tpm"].iloc[0]
            print(f"[ingest] rows={len(df_merged)} tpm_nonzero={tpm_nonzero} sample_tpm={sample_name}:{sample_tpm}")
        else:
            print(f"[ingest] rows={len(df_merged)} tpm_nonzero={tpm_nonzero}")
    except Exception:
        pass

    print(f"Upserted {len(df_new)} new rows → {args.out} (total now {len(df_merged)})")
    print(f"Updated dim → {args.dim} (total now {len(dim_new)})")
    print(f"Match audit → {args.audit} (rows {len(df_audit)})")
    print(f"Unresolved whitelist → {args.unresolved} (rows {len(unresolved)})")

if __name__ == "__main__":
    main()
