1 — Immediate ROI (strengthen Suggested Targets signal)
	•	Opponent pace + implied totals environment (derive implied team totals from spread/ML/OU(over/under); bump PTS/AST/REB confidence) [parked until spreads/OU ingest/backfill work is done]
	•	Stress score (0–100) from rest/travel/load (single numeric penalty/boost used everywhere)
	•	Usage proxy via team-share trends (player share of team PTS/REB/AST; last 5 vs last 20 trend)


2 - Trajectory Engine v1 (Parked - med. priority)
	• Basic Trend v0 overlay failed to produce much/any signal.
	• Dedicate time to build a robust Trajectory feature

3 — Reach Mode (make “Reached” meaningful)
	•	Reach Mode (med priority): pre-game flag “reach” candidates (meaningful “Reached” tag)
	•	Next-tier increments spec:
	•	PTS: +5
	•	REB/AST: +2 and +4
	•	3PT: +1 and +2

4 — Minutes & game-script realism
	•	Blowout sensitivity / minutes-stability 2.0 (discount/adjust confidence when big-fav blowouts clip minutes)

5 — Cross-cutting overlay
	•	Model-free “market alignment” on props day-of (stability tags: big favorite + healthy rotation; clustering in high implied totals)

6 — Postmortem learning (turn daily results into tuning)
	•	Postmortem learning loop: “what worked yesterday” (hit-rate by stat, threshold, chip conditions)
	•	Postmortem-driven auto-tune (bias thresholds down under stress; promote archetypes that “reach” often)

7 — Defensive matchup depth (optional complexity)
	•	Defense-by-position lite (proxy-based; stepping stone before true positional tables)


⸻

Resolved (implemented + functioning live)
	•	Injury-aware intelligence (most “betting-real” swing)
		 - Injury impact tag baked into Suggested Targets (“next man up” boosts; opp big OUT logic)
		Role change detector (injury-driven) (same goal as #6, but formalized as a ruleset + small boosts by archetype)
	•	Market-facing UX (you’ll use it more, faster feedback loop)
	•	One-click “Copy slip” parlay string (Suggested Targets → copyable text)
	•	Parlay Builder mode (2–6 legs) (pin legs + naive hit prob)
	•	Correlation guardrails (warn on bad combos; tag correlated legs)
	•	3PT wired into Core/Consistency
		-3PT is integrated everywhere props exist (thresholds 1–6).
	•	Trend overlay v0: Core/Consistency trends 
		 (L10–L20,L10–Season) → future “TREND ↑/↓” (no signal, move on to Trajectory Engine)
	•	Trap Risk: Volatility + landmine rate (IQR/std + “finished within 1 of line” %) → trap warnings + ranking boost for stable floors
	•	Home/Away splits (floor + hit-rate split with sample gating)

Parked (low priority context only)
	•	Durability upgrades (low priority): data contracts + validation
	•	Durability upgrades (low priority): “single source of truth” mapping table for stat aliases / keys