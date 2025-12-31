## NBA GPT v2 — Agentic Roadmap Slice (Draft)

> **Goal:** evolve NBA GPT from a deterministic v1 target engine into a **safe, testable, agent-assisted system** that can analyze daily performance, propose improvements, validate them via backtests, and (eventually) open PRs—without risking regressions or repo chaos.

---

### North Star Outcomes

**Product**
- Targets feel **“alive”**: recommendations respond to season drift (roles, injuries, rotations, trends), not just a fixed rubric.
- User can steer results via **risk appetite + constraints + notes** (“blacklist”, “team focus”, “avoid B2B”, etc.).
- UI becomes “LLM-style”: user gives direction → “Generate Targets” → outputs + rationale.

**Engineering**
- The “brain” becomes **configurable & versioned** (not hardcoded).
- All changes are measurable via an **evaluation harness** (replay/backtest + metrics).
- The agent is constrained: **analysis + proposals → PRs → CI gates** (no uncontrolled commits to main).

---

## v2 Milestones

### M0 — Repo hygiene prerequisite: reduce conflict / single-writer strategy (Recommended)
**Why:** As long as GitHub Actions and you both write to `main`, conflicts remain a tax—even with GitHub Desktop.

**Deliverables**
- Decide and implement one:
  1) **Automation writes to a `data` branch** (preferred), and `main` only updates via PR merges; OR  
  2) Automation publishes artifacts under Releases / a dedicated branch; OR  
  3) Automation creates PRs instead of committing directly.
- Add CI checks to prevent `main` drift from “silent automation commits”.

**Acceptance**
- You can work locally for a few hours and push without fighting “add/add JSON conflict” daily.
- Workflows run without colliding with your manual commits.

---

### M1 — Parameterize v1 engine into a tunable config (Config-first brain)
**Why:** “Agentic learning” starts by making the rubric tunable without touching code.

**Deliverables**
- Create `engine_config.json` (or similar) containing:
  - weights (Core/Consistency, stat weights, window weights)
  - gates (min minutes, min games, DNP thresholds, injury exclusions)
  - tier thresholds (how you map Core/Confidence to tiers)
  - penalties/bonuses (role trend, volatility, matchup flags if any)
  - stat-specific behavior (PTS vs REB/AST vs 3PT special casing)
  - default “risk modes”: Conservative / Neutral / Degen presets
- Modify `build_player_snapshot_v2.py` and/or selection layer to:
  - read config
  - emit outputs identical to v1 when using `engine_config_v1_equivalent.json`

**Artifacts**
- `configs/engine_config_v2.json`
- `configs/presets/{conservative,neutral,degen}.json` (optional structure)

**Acceptance**
- Running the pipeline with “v1 preset” yields effectively the same target output (within expected rounding/ties).
- Config changes alter outputs without code changes.

---

### M2 — Formalize “Directives”: bridge user intent → deterministic engine
**Why:** This is the core of “LLM UI” without letting the LLM freestyle pick bets.

**Deliverables**
- Define a JSON schema for “directives” (what user tells the system):
  - `risk_mode`: Conservative/Neutral/Degen
  - `include_players` / `exclude_players` (blacklist)
  - `include_teams` / `exclude_teams`
  - `max_legs`, `max_players`, or “top N”
  - `avoid_b2b`, `avoid_minutes_risk`, `avoid_injury_tags`
  - optionally: per-stat preference knobs (prioritize PTS vs 3PT, etc.)
- Update engine to accept `directives.json` at runtime:
  - merge preset config + directives into an effective run config
- Update UI:
  - a small “Generate Targets” panel (text + toggles)
  - persist directives locally (like Slip Stash)

**Acceptance**
- Given the same data + same directives, outputs are deterministic.
- A user can type: “Only show OKC + BOS players, conservative, blacklist Player X” and it reliably applies.

---

### M3 — Evaluation harness v1: replay + metrics (“truth machine”)
**Why:** Without evaluation, v2 becomes vibes and regressions.

**Deliverables**
- Create a backtest runner script (e.g., `backtest_targets.py`) that:
  - replays a date range
  - generates targets per date using a given config/preset
  - compares recommended thresholds to actuals from postmortem/perf artifacts
- Define v2 metrics (pick a minimal set you trust):
  - Hit Rate by stat/tier/window (L10/L20/Season)
  - “Quality” buckets: clean hits, near misses, landmines (DNP, injury scratch)
  - Stability: day-to-day churn (how often targets flip)
  - Coverage: how many “eligible” players produce targets
  - Optional ROI proxy if/when you model odds (but not required early)
- Write results to:
  - `data/eval/` (date-stamped)
  - summary JSON for dashboard display (tiny widget)

**Acceptance**
- You can run: “compare v1 preset vs v2-tuned preset over last 30 days” and see metrics.
- Harness is fast enough to run locally + in CI for PR checks.

---

### M4 — “Agentic analyst” v1: daily performance reasoning & recommendations
**Why:** Immediate value; low risk. This is your first “agentic” feature.

**Deliverables**
- Agent reads:
  - latest perf summaries (`data/perf/…`)
  - yesterday targets + actuals
  - injury outputs + notable DNP/role landmines
- Agent produces **structured report**:
  - “What worked / what failed”
  - “Top 3 recurring failure modes”
  - “Suggested config changes” (as a patch-like diff against config)
  - “Suggested blacklists / watchlists”
- Output stored as:
  - `data/agent/daily_report_YYYY-MM-DD.md`
  - `data/agent/recommendations_YYYY-MM-DD.json` (machine-usable)

**Acceptance**
- Daily agent report is readable and consistently references your own metrics/artifacts.
- Recommendations are expressed as config/directive changes (not fuzzy advice only).

---

### M5 — Agent proposes config changes safely (no code edits yet)
**Why:** Tight feedback loop without letting an LLM refactor your codebase.

**Deliverables**
- Build a small “apply recommendations” tool:
  - takes `recommendations.json`
  - generates a candidate config preset (`configs/candidates/...`)
- Add CI job:
  - run evaluation harness on baseline preset vs candidate preset
  - produce comparison report artifact
- Optional: have agent open a PR that contains:
  - the candidate preset
  - generated eval results
  - a plain-English rationale

**Acceptance**
- Every agent suggestion is testable.
- Candidate presets are only promoted if they improve defined metrics and don’t increase landmine rate.

---

### M6 — Agent opens PRs for small code changes (guardrailed)
**Why:** This is “real agentic dev,” but must be constrained.

**Deliverables**
- Define boundaries:
  - Allowed file paths (e.g., `configs/`, `docs/`, `data/agent/`, maybe `build_targets_postmortem.py` *later*)
  - Forbidden: workflows, deployment, core ingest scripts (initially)
- Agent can:
  - open PR
  - include a “why” summary + eval artifact link
- CI must pass:
  - unit tests (add some)
  - eval harness threshold
  - lint/format

**Acceptance**
- PRs are consistently reviewable and don’t introduce breakage.
- You can merge with confidence.

---

### M7 — UI v2: “LLM-style interface” over deterministic engine
**Why:** Product leap. This is where it feels like NBA GPT v2.

**Deliverables**
- UI panel:
  - text input (notes/instructions)
  - risk mode toggle
  - optional quick filters (teams/players)
  - “Generate Targets” button
- Under the hood:
  - parse notes into directives (LLM)
  - pass directives to deterministic engine
  - display “Why these picks” explanations (LLM-generated, but grounded)
- Persist session state:
  - last directives
  - last run outputs
  - slip stash still intact

**Acceptance**
- User experience: “I talk to it like a coach” + outputs are still reproducible.

---

### M8 — Advanced (later): canary + rollback + auto-merge
**Why:** Only after harness is trustworthy and failures are rare.

**Deliverables**
- Maintain “last known good” preset
- Auto-merge only when:
  - improvements persist over multiple evaluation windows
  - landmine rate doesn’t increase
- Rollback triggers:
  - performance drop over last N days
  - excessive churn / instability

**Acceptance**
- System can self-correct without bricking quality.

---

## Core Design Principles (Non-negotiables)

1) **Deterministic executor, LLM orchestrator.**  
LLM suggests + explains; engine computes.

2) **Everything versioned.**  
Configs, directives, eval results, daily reports.

3) **No silent self-modification.**  
All “learning” surfaces as a PR + eval delta.

4) **Single-writer strategy.**  
Stop daily JSON conflicts by design, not heroics.

---

## Suggested File/Folder Structure (v2-ready)
- `configs/`
  - `engine_config.json`
  - `presets/neutral.json`
  - `presets/conservative.json`
  - `presets/degen.json`
  - `candidates/…`
- `data/agent/`
  - `daily_report_YYYY-MM-DD.md`
  - `recommendations_YYYY-MM-DD.json`
- `data/eval/`
  - `eval_YYYY-MM-DD.json`
  - `compare_baseline_vs_candidate_YYYY-MM-DD.json`
- `scripts/`
  - `backtest_targets.py`
  - `apply_recommendations.py`

---

## Definition of Done for “v2 Agentic MVP”
You can:
1) Generate targets via **preset + directives**,  
2) Run a **backtest/eval** on recent history,  
3) Have an **agent produce a daily report + suggested config tweaks**,  
4) Apply those tweaks into a candidate config,  
5) Validate via CI,  
6) Optionally open a PR.

That’s “agentic” in a way that’s *real* and won’t blow up your repo.