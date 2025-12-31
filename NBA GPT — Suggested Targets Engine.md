# NBA GPT — Suggested Targets Engine (Under the Hood)

*A sister doc to the canonical README: brief, readable, but detail-rich.*  
**Scope:** Player Props “Suggested Targets” engine: how targets are computed, tagged, and ranked.  
**Static doc:** update only by appending when features are deployed + proven useful.

---

## 1) Suggested Targets pipeline overview (end-to-end)

NBA GPT’s Player Props “Suggested Targets” is a deterministic pipeline that turns canonical stat history into a ranked daily list of legs (e.g., `PTS 20+`, `3PT 2+`).

### Canonical inputs
- **player_game_log.csv** (canonical stat history)
  - Built/updated by `espn_player_ingest.py`
  - Per player, per game: PTS/REB/AST + **TPM (tpm)** for 3PT, minutes, dates, IDs, etc.

### Feature layer (materialized “player understanding”)
- **player_snapshot.csv** + **data/player_features.json**
  - Built by `build_player_snapshot_v2.py`
  - Computes per-player, per-window features (L10/L20/Season), including:
    - Core averages
    - Consistency scores + tiers
    - Volatility proxies (IQR)
    - 3PT eligibility gates
    - Home/Away split metrics (sample-gated)

### Target construction + ranking (daily “what we recommend”)
- **players.html** (client-side engine)
  - Loads snapshot + features + daily context
  - Constructs threshold legs for each player/stat
  - Applies heuristics + gates + weighted adjustments
  - Produces ranked “Top Targets” and “All Targets” lists

### Post-game resolution + performance ledger (feedback loop scaffolding)
- **data/targets_postmortem.json** + archived history  
  - Built by `build_targets_postmortem.py`
  - Resolves actuals for recommended legs (PTS/REB/AST/3PT)
- **data/perf/** (daily + season + all-time summaries)
  - Built by `build_perf_daily.py`

---

## 2) Metric definitions (what the engine computes)

### Windows
Most metrics are computed for:
- **L10** (last 10 games)
- **L20** (last 20 games)
- **Season** (season-to-date)

The UI allows selecting a “Metric Window.” Some scoring may apply **season dampening** to reduce swinginess.

---

### Core Avg (per stat, per window)
**Core Avg = 50% median + 50% trimmed mean**

- Purpose: stabilize form/central tendency vs raw average.
- Used widely in:
  - Player card stat headers (`Core(L10)`, `Core(L20)`, etc.)
  - Suggested Targets ranking rubric

---

### Consistency score (0–100) + tiers
A normalized stability score that starts from a volatility proxy (IQR-based) and is enhanced with practical penalties/boosts:
- minutes stability
- DNP / zero-mins
- landmines
- injury risk / role instability (where available)

**Consistency tiers (current):**
- `< 50` → **Inconsistent**
- `50–64` → **Consistency OK**
- `65–72` → **Solid Consistency**
- `73–78` → **Consistent**
- `79–84` → **Very Consistent**
- `85–100` → **Elite Consistency**

Used in:
- Player card “Consistency pill”
- Suggested Targets ranking boosts (tier-weighted)

---

### Volatility (IQR)
IQR is used as a robust volatility proxy per stat/window.
- Used to flag **Trap Risk**
- Also used as a scale/normalizer for other derived signals

---

### Landmine rate (“near miss” pressure)
A heuristic “trap” feature: how often a player finishes **exactly 1 under** a given threshold line.
- Used to flag **Trap Risk**
- Also can influence ranking (depending on current weighting)

---

### 3PT (TPM) eligibility gate
3PT is computed from **TPM (tpm)**, but 3PT is structurally spikier than other stats.  
To avoid misleading “elite consistency” for low-volume shooters, 3PT includes eligibility gates per window based on:
- window games played
- TPM mean
- nonzero TPM rate

If ineligible:
- 3PT Core/Consistency may still be computed/stored,
- but display/weighting may show **Low 3PT volume** and reduce/disable boosts.

---

### Home/Away splits (sample-gated)
For each stat/window:
- Compute **Core(Home)** vs **Core(Away)**
- Compute split delta and a normalized split strength (`z`) using a volatility denominator
- Mark usable only if both splits have enough games (sample gating)

Used in:
- Suggested Targets flags (e.g., **HOME+**)
- Suggested Targets ranking adjustment (small boost/penalty when venue aligns/misaligns)
- Optional deep-card statline flags (HOME+/HOME- / AWAY+/AWAY-)

---

### Rest / Travel / Load context
The UI surfaces:
- **Rest** (days)
- **Travel** (pattern tags)
- **Load** (a numeric-ish pressure indicator)

This can appear as:
- a warning tag (e.g., **HIGH FATIGUE**)
- and/or a scoring adjustment (depending on current rubric wiring)

---

## 3) Tags / pills reference (meaning + wiring)

This section is the “dictionary” for what you see on the site and why it shows up.

### Suggested Targets / player context pills
- **ROLE / ↑ ROLE**
  - Meaning: role opportunity / role up signal (often injury-driven)
  - Source: daily injury context + role heuristics
  - Wiring: player cards + target rows

- **INJURY STATUS (OUT / QUES / PROB, etc.)**
  - Meaning: availability + risk
  - Source: injuries pipeline (`rotowire_injuries_only.py`) with Pages staleness fallback
  - Wiring: Injury drawer + player context + filters

- **HIGH FATIGUE**
  - Meaning: elevated fatigue risk from Rest/Travel/Load signals
  - Source: Rest/Travel/Load computations in UI context layer
  - Wiring: target flags + player card context

### Quality / risk pills
- **TRAP RISK**
  - Meaning: “near miss pressure” (landmine rate) and/or high volatility (IQR)
  - Source: Suggested Targets construction in `players.html`
  - Wiring: flag displayed on target rows; may influence ranking

- **HOME+ / AWAY+ (and optionally HOME- / AWAY-)**
  - Meaning: venue-aligned split advantage (or meaningful mismatch if `-`)
  - Source: Home/Away split features (sample-gated) + today’s venue
  - Wiring: target flags; optionally expanded player stat lines

### Consistency pill
- **Inconsistent / Consistency OK / Solid Consistency / Consistent / Very Consistent / Elite Consistency**
  - Meaning: stability tier derived from the 0–100 consistency score
  - Source: `build_player_snapshot_v2.py` → `player_features.json`
  - Wiring: player card header pill; also used by ranking rubric

> Note: exact thresholds, gates, and weights are intentionally encoded in code (mostly `players.html`) so the system remains deterministic and auditable.

---

## 4) Scoring / ranking rubric (how “Top Targets” is chosen)

Suggested Targets ranking is a heuristic score built from a base quality signal plus additive adjustments.  
**The guiding principle:** reward stable floors and context-aligned opportunity; downweight traps and low-signal edges.

### A) Candidate generation
For each player and stat (PTS/REB/AST/3PT):
- Enumerate threshold tiers (stat-specific)
- Compute hit rates over the chosen window(s)
- Filter out obvious invalids (injury/out, missing data, etc.)

### B) Base score (core target quality)
Base score generally increases with:
- higher hit rate
- adequate sample size
- stronger Core for the selected window (with season dampening where applicable)

### C) Additive adjustments (most common)
These apply as small boosts/penalties, often sample-gated:

- **Consistency tier boost**
  - Higher consistency tiers receive larger boosts
  - 3PT may be dampened, and **3PT ineligible** disables tier bonuses

- **Role / opportunity boosts**
  - “↑ ROLE” or injury-driven opportunity can boost targets

- **Fatigue / stress penalties**
  - High fatigue may reduce confidence and/or add a warning flag

- **Trap Risk penalties / warnings**
  - Elevated landmine rate and/or volatility can mark a target as TRAP RISK
  - May reduce ranking or simply warn (implementation-dependent)

- **Home/Away split alignment**
  - If venue aligns with meaningful split advantage: small boost + HOME+/AWAY+ tag
  - If meaningful mismatch: small penalty and/or HOME-/AWAY- (if enabled)

### D) Gating rules (avoid fake signal)
Common gates that suppress boosts/flags:
- minimum games in window
- stat-specific eligibility (notably 3PT)
- split sample requirements (home and away both must be adequate)

### Where to see the exact rubric in code
- Primary: `players.html` Suggested Targets construction and scoring section
- Feature inputs: `build_player_snapshot_v2.py` output fields in `player_features.json`

---

### Appendix: canonical artifacts referenced by the engine
- `data/player_game_log.csv`
- `data/player_snapshot.csv`
- `data/player_features.json`
- `data/targets_postmortem.json` (+ `data/history/...`)
- `data/perf/*`

---
**Last updated:** 2025-12-30