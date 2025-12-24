#!/usr/bin/env python3
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd


def american_to_prob(odds):
    if odds is None:
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def load_master(path="nba_master.csv"):
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["home_ml"] = pd.to_numeric(df["home_ml"], errors="coerce")
    df["away_ml"] = pd.to_numeric(df["away_ml"], errors="coerce")
    return df


def empty_report(asof_date=None):
    return {
        "asof_date": asof_date,
        "built_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": {
            "games_total": 0,
            "favorites_won": 0,
            "upsets": 0,
            "coinflips_total": 0,
            "coinflips_fav_won": 0,
            "coinflips_ud_won": 0,
            "big_favs_total": 0,
            "big_favs_won": 0,
            "big_favs_upset": 0,
            "home_wins": 0,
            "away_wins": 0,
        },
        "biggest_upsets": [],
        "coinflips": [],
        "big_fav_failures": [],
    }


def main():
    df = load_master()
    completed = df[df["home_score"].notna() & df["away_score"].notna()]
    today_et = dt.datetime.now(ZoneInfo("America/New_York")).date()
    prior = completed[completed["game_date"] < today_et]
    asof_date = None
    if not prior.empty:
        asof_date = prior["game_date"].max()

    os.makedirs("data", exist_ok=True)

    if asof_date is None:
        report = empty_report(None)
        with open("data/moneyline_postmortem.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print("[moneyline_postmortem] No completed games before today; wrote empty report.")
        return

    day_games = prior[prior["game_date"] == asof_date].copy()

    records = []
    for row in day_games.itertuples(index=False):
        home_ml = row.home_ml
        away_ml = row.away_ml
        if pd.isna(home_ml) or pd.isna(away_ml):
            continue
        home_prob = american_to_prob(home_ml)
        away_prob = american_to_prob(away_ml)
        if home_prob is None or away_prob is None:
            continue

        home_score = row.home_score
        away_score = row.away_score
        if pd.isna(home_score) or pd.isna(away_score) or home_score == away_score:
            continue

        winner_side = "home" if home_score > away_score else "away"
        winner_team = row.home_team_name if winner_side == "home" else row.away_team_name
        winner_ml = home_ml if winner_side == "home" else away_ml
        winner_prob = home_prob if winner_side == "home" else away_prob

        favorite_side = "home" if home_prob >= away_prob else "away"
        favorite_team = row.home_team_name if favorite_side == "home" else row.away_team_name
        favorite_ml = home_ml if favorite_side == "home" else away_ml
        favorite_prob = home_prob if favorite_side == "home" else away_prob

        record = {
            "away_team": row.away_team_name,
            "home_team": row.home_team_name,
            "away_abbrev": row.away_team_abbrev,
            "home_abbrev": row.home_team_abbrev,
            "away_score": int(away_score),
            "home_score": int(home_score),
            "away_ml": float(away_ml),
            "home_ml": float(home_ml),
            "away_implied_pct": round(away_prob * 100, 1),
            "home_implied_pct": round(home_prob * 100, 1),
            "favorite_side": favorite_side,
            "favorite_team": favorite_team,
            "favorite_ml": float(favorite_ml),
            "favorite_implied_pct": round(favorite_prob * 100, 1),
            "winner_side": winner_side,
            "winner_team": winner_team,
            "winner_ml": float(winner_ml),
            "winner_implied_pct": round(winner_prob * 100, 1),
            "final_score": f"{row.away_team_abbrev} {int(away_score)} - {row.home_team_abbrev} {int(home_score)}",
            "surprise": round(1.0 - winner_prob, 4),
        }
        record["coinflip"] = abs(home_prob - away_prob) <= 0.07
        record["big_favorite"] = favorite_prob >= 0.70
        record["upset"] = winner_side != favorite_side
        records.append(record)

    if not records:
        report = empty_report(str(asof_date))
        with open("data/moneyline_postmortem.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print("[moneyline_postmortem] No eligible games with odds; wrote empty report.")
        return

    favorites_won = sum(1 for r in records if not r["upset"])
    upsets = sum(1 for r in records if r["upset"])
    coinflips = [r for r in records if r["coinflip"]]
    coinflips_fav_won = sum(1 for r in coinflips if not r["upset"])
    coinflips_ud_won = sum(1 for r in coinflips if r["upset"])
    big_favs = [r for r in records if r["big_favorite"]]
    big_favs_won = sum(1 for r in big_favs if not r["upset"])
    big_favs_upset = sum(1 for r in big_favs if r["upset"])
    home_wins = sum(1 for r in records if r["winner_side"] == "home")
    away_wins = sum(1 for r in records if r["winner_side"] == "away")

    biggest_upsets = sorted(
        [r for r in records if r["upset"]],
        key=lambda r: r["surprise"],
        reverse=True,
    )[:5]

    big_fav_failures = sorted(
        [r for r in records if r["big_favorite"] and r["upset"]],
        key=lambda r: r["favorite_implied_pct"],
        reverse=True,
    )[:5]

    coinflip_list = sorted(coinflips, key=lambda r: r["surprise"], reverse=True)[:8]

    report = {
        "asof_date": str(asof_date),
        "built_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": {
            "games_total": len(records),
            "favorites_won": favorites_won,
            "upsets": upsets,
            "coinflips_total": len(coinflips),
            "coinflips_fav_won": coinflips_fav_won,
            "coinflips_ud_won": coinflips_ud_won,
            "big_favs_total": len(big_favs),
            "big_favs_won": big_favs_won,
            "big_favs_upset": big_favs_upset,
            "home_wins": home_wins,
            "away_wins": away_wins,
        },
        "biggest_upsets": biggest_upsets,
        "coinflips": coinflip_list,
        "big_fav_failures": big_fav_failures,
    }

    with open("data/moneyline_postmortem.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"[moneyline_postmortem] asof_date={asof_date} games={len(records)}")


if __name__ == "__main__":
    main()
