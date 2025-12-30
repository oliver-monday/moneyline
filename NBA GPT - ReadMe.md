# NBA GPT (Moneyline repo) — Canonical Overview

## What NBA GPT is
NBA GPT is a lightweight, GitHub Pages–hosted mini-app that turns raw NBA game and player data into a daily decision dashboard for two adjacent use-cases:

1) **Moneylines** — a compact slate view for game-level context and market signals.  
2) **Player Props** — the primary focus: a daily “targets” engine that surfaces a short list of player stat thresholds that are most likely to hit, with supporting context and historical performance tracking.

The project is built to be **robust, reproducible, and auditable over time**: every daily run produces artifacts that can be archived, re-read, and evaluated later—so improvements can be measured against real outcomes.

---

## Design principles
- **Reconstructable history:** Every day’s inputs/outputs can be preserved so you can replay “what did the system know then?”  
- **Minimal UI friction:** The app is meant to be fast to scan—chips/pills, compact summaries, expandable detail only when needed.  
- **Separation of concerns:** Data ingest → feature build → target selection → UI render. Each stage writes explicit artifacts.
- **Bias toward reliability:** If data is missing or stale, the system prefers graceful fallback over hard failure.

---

## How it works (end-to-end pipeline)
NBA GPT is powered by a daily workflow that produces a set of canonical data files used by the site.

### 1) Game + market backbone (nba_master.csv)
A central “master” table (nba_master.csv) acts as the daily schedule and results backbone. It contains:
- game identifiers + dates
- teams, scores
- odds/spreads
- additional metadata used by the site

This file anchors game grouping and daily state.

### 2) Player game logs (player_game_log.csv)
A canonical player historical game log is built from ESPN boxscore data and maintained incrementally. It stores per-player, per-game stat lines and identifiers needed for:
- hit-rate calculations at thresholds
- season and windowed averages
- post-game resolution (actual vs target)

This file is the foundation for all player-props analytics.

### 3) Feature build + snapshot (player_snapshot.csv + player_features.json)
A feature builder consumes the canonical player game log to produce:
- **player_snapshot.csv**: one row per player with derived features and stats used by the UI and target engine  
- **player_features.json**: a structured companion file for richer per-player computed fields

This layer is where “the system’s understanding” of each player is materialized into stable, versionable outputs.

### 4) Target selection + daily suggestions
From the snapshot + context signals, the system generates daily “targets”:
- Targets are stat thresholds (e.g., PTS 20+, AST 4+) with calculated hit rates over defined windows.
- Targets are ranked by a scoring heuristic and filtered by rules (injury, role volatility, etc.).
- The result is a curated set of Suggested Targets surfaced in the UI.

### 5) Postmortem + feedback artifacts (targets_postmortem.json + perf ledger)
After games complete, the system resolves suggested targets against actual outcomes and writes a daily postmortem:
- which targets hit/missed
- summary counts and rates
- “reaches” (outperforming beyond the suggested line)

Separately, a performance ledger persists daily and cumulative summaries, enabling season-to-date and all-time tracking.

---

## What the Player Props view does
The Player Props page is a daily interactive surface over the pipeline outputs.

Core behaviors:
- **Suggested Targets:** a ranked list of the day’s best target legs (with supporting context).
- **Player cards:** expandable per-player panels showing recent form, averages, and threshold hit rates.
- **Injury report:** integrates an external injuries feed to support stay-away logic and opportunity/context flags.
- **Recap (postmortem):** summarizes yesterday’s performance and provides a record of hit/miss outcomes.
- **Slip building (optional module):** allows selecting legs from existing target chips and exporting a plain-text “slip” list (no pricing/calcs required).

The UI is static-hosted and data-driven: it fetches the produced artifacts (CSV/JSON) and renders everything client-side.

---

## What the Moneylines view does
The Moneylines page is a game-level slate view that uses nba_master.csv and related daily artifacts to display:
- matchups
- moneyline/spread context
- basic classification and “confidence” presentation

It is intentionally simpler than Player Props and serves as a daily “favorites and stay-away” scan tool.

---

## Data integrity + reproducibility model
NBA GPT is structured around a small number of canonical files that serve as “single sources of truth”:

- **nba_master.csv** — canonical games/odds/results backbone  
- **player_game_log.csv** — canonical player boxscore-derived history  
- **player_snapshot.csv / player_features.json** — derived features used by the UI/engine  
- **targets_snapshot_YYYY-MM-DD.json** — frozen daily suggestions (“what we recommended that day”)  
- **targets_postmortem.json + history/** — daily resolution outputs (“what happened”)  
- **data/perf/** — durable performance ledger (daily + season + all-time)

This makes the system auditable: you can inspect exactly what the model surfaced and whether it worked, day by day.

---

## Operating model (how it’s run)
NBA GPT is designed to run automatically via GitHub Actions:
- daily ingestion/build jobs update data artifacts
- the site is updated by copying the latest outputs into the Pages-served directory
- history archives accumulate over time so analysis can grow without breaking the UI

The workflow is intentionally “artifact-first”: the website is just a renderer over the produced files.

---

## What this enables long-term
Because the system preserves daily decisions and outcomes, it can support:
- longitudinal performance measurement (season + multi-year)
- iterative improvement of selection heuristics
- future “learning loops” driven by postmortems and archived snapshots
- potential migration to local agents/LLMs without losing historical state

In short: NBA GPT is a daily NBA decision dashboard built on reproducible data artifacts, with a bias toward robustness, traceability, and continuous measurable iteration.