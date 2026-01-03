# NBA GPT v1 - ROADMAP

## FEATURES:
 — Immediate ROI (strengthen Suggested Targets signal)["bundles" remaining...]
	• Opponent pace + implied totals environment (derive implied team totals from spread/ML/OU(over/under); bump PTS/AST/REB confidence) [parked until spreads/OU ingest/backfill work is done]

 - Trajectory Engine v1 [Parked - med. priority]
	• Basic Trend v0 overlay failed to produce much/any signal.
	• Dedicate time to build a robust Trajectory feature

 - Postmortem Learning v0.1
	• Rolling last 7/14 days + “best/worst tags” summary
	• Add sample-size gating styling (e.g., fade chips when total < 3), so you don’t overreact to T 0/1.
	• Add a tiny “lift vs baseline” later (e.g., UP: +22pp vs All) once you’ve got a week+ of data.

 — Minutes & game-script realism
	• Blowout sensitivity / minutes-stability 2.0 (discount/adjust confidence when big-fav blowouts clip minutes)

 — Wire in actual odds
	• this will replace/upgrade our proxy rubric that boosts stat tiers closer to/above players' avgs, given baseline assumption being that high conf tier too far below avg = miniscule odds/consistency priced in/not great R/R
	• not out to disprove assumption, just build towards surfacing true edge where it may exist / we can find it.
	• https://the-odds-api.com ?

 — Cross-cutting overlay
	• Model-free “market alignment” on props day-of (stability tags: big favorite + healthy rotation; clustering in high implied totals)

 — Defensive matchup depth (optional complexity)
	• Defense-by-position lite (proxy-based; stepping stone before true positional tables)

 — Reach Mode (make “Reached” meaningful) [Parked - med. priority, until baseline selection engine = maximum robustness/repeatable success/profitability]
	• Next-tier increments spec:
		•	PTS: +5
		•	REB/AST: +2 and +4
		•	3PT: +1 and +2

## HOUSEKEEPING:

 - Continued Documentation (as needed):
 		- - Reference dict/list/table for all tags etc. what they mean, how they’re computed, where they are wired to, + current weights/expected impact; "tags/weights schema + where computed."

⸻

## Resolved (implemented + functioning live)
	•	Injury-aware intelligence (most “betting-real” swing)
		 - Injury impact tag baked into Suggested Targets (“next man up” boosts; opp big OUT logic)
		Role change detector (injury-driven) (same goal as #6, but formalized as a ruleset + small boosts by archetype)
	•	Market-facing UX (you’ll use it more, faster feedback loop)
	•	One-click “Copy slip” parlay string (Suggested Targets → copyable text)
	•	Parlay Builder mode (2–6 legs) (pin legs + naive hit prob)
	•Suggested Targets signal strengthening:
		•	3PT wired into Core/Consistency
		-3PT is integrated everywhere props exist (thresholds 1–6).
		•	Trend overlay v0: Core/Consistency trends 
		 (L10–L20,L10–Season) → future “TREND ↑/↓” (no signal, move on to Trajectory Engine)
		•	Trap Risk: Volatility + landmine rate (IQR/std + “finished within 1 of line” %) → trap warnings + ranking boost for stable floors
		•	Home/Away splits (floor + hit-rate split with sample gating)
		•	Stress score (0–100) from rest/travel/load (single numeric penalty/boost used everywhere). [v0 -> v1.1 recalibration]	
		•	Usage proxy via team-share trends (player share of team PTS/REB/AST; last 5 vs last 20 trend)
			- Role Confirmation: Usage proxy via team-share trends … + ROLE✓ / ROLE? confirmation layer
		•	MINs-adjusted everything v0
		•	B2B calibrated to actual player performance (not just conventional wisdom ie B2B=underperformance risk)

	• Postmortem learning (turn daily results into tuning)
		•	Postmortem learning loop: “what worked yesterday” (hit-rate by stat, threshold, chip conditions)
		•	Postmortem-driven auto-tune (bias thresholds down under stress; promote archetypes that “reach” often)

	•	Create Documentation for internal use:
		• ReadMe.md - holistic project overview
		• Suggested Targets Engine.md - feature logic/wiring/architecture explainer 

## Parked (low priority context only)
	•	Durability upgrades (low priority): data contracts + validation
	•	Durability upgrades (low priority): “single source of truth” mapping table for stat aliases / keys

### v2+ Goals / Longterm Vision
	•	OM rough idea / prompt:
			•	NBA GPT goes agentic: auto-self-iterating-learning feedback loop. Reason over fresh daily data ingest & performance postmortems -> recalibrate weights and/or Target selection criteria and/or make changes and/or design and implement new features, on an as-desired/as-needed basis.
				• Data-based Backtesting + Feature Testing/Experimental dev environment.
			•	Selection Target generation becomes a result of daily Reasoning over entire/discrete dataset(s), not just a hardcoded rubric that returns outputs -> more "alive" and tapped into the flow of the shifting sands/trajectories of an NBA season.
				• UI becomes a simple LLM-style interface; perhaps user can adjust some directional parameters (ex: risk appetite: Conservative/Neutral/Degen) and/or Chat-input any eyeball/gut feeling observations / ad-hoc instructions (ex: "Player X looks like he's playing through an undiagnosed injury" -> stay away/temporary blacklist; "I'm only interested in X,Y,Z Players / Players on XYZ Teams", etc) -> CTA="Generate Targets" -> Think/Reason -> agent makes Custom Selections.
	•	ChatGPT (Dojo) roadmap design [condensed - see "NBA GPT v2 - "Agentic" Roadmap.md" for details]:
			• **v2 North Star:** NBA GPT becomes *agent-assisted + data-driven* (targets feel “alive”), with a “Generate Targets” UI steered by user directives (risk appetite + filters + notes), while keeping execution deterministic:
				• **Pre-req (reduce repo chaos):** implement a **single-writer strategy** so automation doesn’t collide with your manual work (automation writes to `data` branch / PRs / releases—`main` updates via PR merge).
				• **Config-first engine:** convert the v1 hardcoded rubric into a **versioned, tunable config** (weights, gates, thresholds, stat rules) + presets (**Conservative / Neutral / Degen**) that reproduce v1 outputs under a “v1-equivalent” preset.
				• **Directives layer:** define a **directives schema** (blacklists, include/exclude teams/players, max legs, avoid B2B/injury tags, etc.) and make target generation deterministic from *(data + preset + directives)*.
				• **Guardrail:** no “agentic” tuning ships without an **eval/backtest harness** showing improvement (tracked + reproducible).