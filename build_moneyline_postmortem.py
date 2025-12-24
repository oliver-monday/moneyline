#!/usr/bin/env python3
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd


def implied_prob_from_ml(odds):
    if odds is None:
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


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
            "favorites_lost": 0,
            "coinflips_total": 0,
        },
        "favorites": [],
        "underdogs": [],
        "coinflips": [],
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

    def pick_winner(home_score, away_score):
        if pd.isna(home_score) or pd.isna(away_score) or home_score == away_score:
            return None
        return "HOME" if home_score > away_score else "AWAY"

    def classify_game(home_ml, away_ml):
        home_prob = implied_prob_from_ml(home_ml)
        away_prob = implied_prob_from_ml(away_ml)
        if home_prob is None or away_prob is None:
            return {
                "home_prob": home_prob,
                "away_prob": away_prob,
                "fav_side": None,
                "dog_side": None,
                "is_coinflip": True,
                "abs_diff": None,
            }
        diff = abs(home_prob - away_prob)
        if diff <= 0.05:
            return {
                "home_prob": home_prob,
                "away_prob": away_prob,
                "fav_side": None,
                "dog_side": None,
                "is_coinflip": True,
                "abs_diff": diff,
            }
        fav_side = "HOME" if home_prob > away_prob else "AWAY"
        dog_side = "AWAY" if fav_side == "HOME" else "HOME"
        return {
            "home_prob": home_prob,
            "away_prob": away_prob,
            "fav_side": fav_side,
            "dog_side": dog_side,
            "is_coinflip": False,
            "abs_diff": diff,
        }

    records = []
    for row in day_games.itertuples(index=False):
        home_score = row.home_score
        away_score = row.away_score
        winner_side = pick_winner(home_score, away_score)
        if winner_side is None:
            continue

        winner_team = row.home_team_name if winner_side == "HOME" else row.away_team_name
        loser_team = row.away_team_name if winner_side == "HOME" else row.home_team_name
        if winner_team == loser_team:
            print(f"[moneyline_postmortem] bad mapping game_id={row.game_id}")
            continue

        winner_abbrev = row.home_team_abbrev if winner_side == "HOME" else row.away_team_abbrev
        loser_abbrev = row.away_team_abbrev if winner_side == "HOME" else row.home_team_abbrev
        winner_score = int(home_score) if winner_side == "HOME" else int(away_score)
        loser_score = int(away_score) if winner_side == "HOME" else int(home_score)

        classification = classify_game(row.home_ml, row.away_ml)
        home_prob = classification["home_prob"]
        away_prob = classification["away_prob"]

        winner_ml = row.home_ml if winner_side == "HOME" else row.away_ml
        loser_ml = row.away_ml if winner_side == "HOME" else row.home_ml

        record = {
            "game_id": row.game_id,
            "home_team": row.home_team_name,
            "away_team": row.away_team_name,
            "home_abbrev": row.home_team_abbrev,
            "away_abbrev": row.away_team_abbrev,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "home_ml": float(row.home_ml) if pd.notna(row.home_ml) else None,
            "away_ml": float(row.away_ml) if pd.notna(row.away_ml) else None,
            "home_prob": home_prob,
            "away_prob": away_prob,
            "winner_side": winner_side,
            "loser_side": "AWAY" if winner_side == "HOME" else "HOME",
            "winner_team": winner_team,
            "loser_team": loser_team,
            "winner_abbrev": winner_abbrev,
            "loser_abbrev": loser_abbrev,
            "winner_score": winner_score,
            "loser_score": loser_score,
            "winner_ml": float(winner_ml) if pd.notna(winner_ml) else None,
            "loser_ml": float(loser_ml) if pd.notna(loser_ml) else None,
            "abs_prob_diff": classification["abs_diff"],
        }

        record["is_coinflip"] = classification["is_coinflip"]
        record["fav_side"] = classification["fav_side"]
        record["dog_side"] = classification["dog_side"]

        if not record["is_coinflip"] and record["fav_side"]:
            fav_prob = home_prob if record["fav_side"] == "HOME" else away_prob
            dog_prob = away_prob if record["fav_side"] == "HOME" else home_prob
            record["favorite_prob"] = fav_prob
            record["underdog_prob"] = dog_prob
        else:
            record["favorite_prob"] = None
            record["underdog_prob"] = None

        records.append(record)

    if not records:
        report = empty_report(str(asof_date))
        with open("data/moneyline_postmortem.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print("[moneyline_postmortem] No eligible games with odds; wrote empty report.")
        return

    favorites = []
    underdogs = []
    coinflips = []
    favorites_won = 0
    favorites_lost = 0

    for r in records:
        if r["is_coinflip"] or not r["fav_side"]:
            coinflips.append(r)
            continue
        if r["winner_side"] == r["fav_side"]:
            favorites.append(r)
            favorites_won += 1
        else:
            underdogs.append(r)
            favorites_lost += 1

    favorites = sorted(favorites, key=lambda r: r["favorite_prob"], reverse=True)
    underdogs = sorted(underdogs, key=lambda r: r["underdog_prob"] or 1.0)
    coinflips = sorted(
        coinflips,
        key=lambda r: r["abs_prob_diff"] if r["abs_prob_diff"] is not None else 9.0,
    )

    report = {
        "asof_date": str(asof_date),
        "built_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": {
            "games_total": len(records),
            "favorites_won": favorites_won,
            "favorites_lost": favorites_lost,
            "coinflips_total": len(coinflips),
        },
        "favorites": favorites,
        "underdogs": underdogs,
        "coinflips": coinflips,
    }

    with open("data/moneyline_postmortem.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"[moneyline_postmortem] asof_date={asof_date} games={len(records)}")


if __name__ == "__main__":
    main()
