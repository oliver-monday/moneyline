#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
LOG_PATH = Path("player_game_log.csv")

SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def last_name(full: str) -> str:
    parts = [p for p in re.split(r"\s+", str(full or "").strip()) if p]
    while parts and parts[-1].lower().strip(".") in SUFFIXES:
        parts.pop()
    if not parts:
        return ""
    return re.sub(r"[^A-Za-z'-]", "", parts[-1])


def build_matchups(df: pd.DataFrame) -> tuple[dict, dict]:
    needed = {"game_id", "team_abbrev", "pts", "reb", "ast", "player_name"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"player_game_log.csv missing columns: {sorted(missing)}")

    df = df.copy()
    df["pts"] = pd.to_numeric(df["pts"], errors="coerce")
    df["reb"] = pd.to_numeric(df["reb"], errors="coerce")
    df["ast"] = pd.to_numeric(df["ast"], errors="coerce")

    team_game = (
        df.groupby(["game_id", "team_abbrev"], as_index=False)[["pts", "reb", "ast"]]
        .sum()
    )

    merged = team_game.merge(team_game, on="game_id", suffixes=("", "_opp"))
    merged = merged[merged["team_abbrev"] != merged["team_abbrev_opp"]].copy()
    if merged.empty:
        return {}, {}

    merged["pts_allowed"] = merged["pts_opp"]
    merged["reb_allowed"] = merged["reb_opp"]
    merged["ast_allowed"] = merged["ast_opp"]
    merged["reb_diff"] = merged["reb"] - merged["reb_opp"]

    team_allowed = (
        merged.groupby("team_abbrev")[["pts_allowed", "reb_allowed", "ast_allowed", "reb_diff"]]
        .mean()
        .reset_index()
    )

    team_allowed["reb_allowed_rank"] = team_allowed["reb_allowed"].rank(
        method="min", ascending=True
    ).astype(int)
    team_allowed["pts_allowed_rank"] = team_allowed["pts_allowed"].rank(
        method="min", ascending=True
    ).astype(int)
    team_allowed["ast_allowed_rank"] = team_allowed["ast_allowed"].rank(
        method="min", ascending=True
    ).astype(int)
    team_allowed["reb_diff_rank"] = team_allowed["reb_diff"].rank(
        method="min", ascending=False
    ).astype(int)

    team_opp_allowed = {}
    for row in team_allowed.itertuples(index=False):
        team = row.team_abbrev
        team_opp_allowed[team] = {
            "reb_allowed": float(row.reb_allowed),
            "reb_allowed_rank": int(row.reb_allowed_rank),
            "reb_diff": float(row.reb_diff),
            "reb_diff_rank": int(row.reb_diff_rank),
            "pts_allowed": float(row.pts_allowed),
            "pts_allowed_rank": int(row.pts_allowed_rank),
            "ast_allowed": float(row.ast_allowed),
            "ast_allowed_rank": int(row.ast_allowed_rank),
        }

    player_team = (
        df.groupby(["player_name", "team_abbrev"], as_index=False)
        .agg(gp=("game_id", "nunique"), reb_sum=("reb", "sum"))
    )
    player_team["avg_reb"] = player_team["reb_sum"] / player_team["gp"]
    player_team = player_team[player_team["gp"] >= 10]
    player_team = player_team.dropna(subset=["avg_reb"])
    if player_team.empty:
        return team_opp_allowed, {}

    top10 = player_team.sort_values("avg_reb", ascending=False).head(10)
    opp_big = {}
    for row in top10.itertuples(index=False):
        team = row.team_abbrev
        existing = opp_big.get(team)
        if existing and existing.get("avg_reb", 0) >= row.avg_reb:
            continue
        opp_big[team] = {
            "last": last_name(row.player_name),
            "player": row.player_name,
            "avg_reb": float(row.avg_reb),
            "gp": int(row.gp),
        }

    return team_opp_allowed, opp_big


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"{LOG_PATH} not found")

    df = pd.read_csv(LOG_PATH, dtype=str)
    team_opp_allowed, opp_big = build_matchups(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "team_opp_allowed.json").write_text(
        json.dumps(team_opp_allowed, indent=2, sort_keys=True), encoding="utf-8"
    )
    (DATA_DIR / "opp_big.json").write_text(
        json.dumps(opp_big, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"[matchups] wrote team_opp_allowed.json teams={len(team_opp_allowed)}; "
        f"opp_big.json teams={len(opp_big)}"
    )


if __name__ == "__main__":
    main()
