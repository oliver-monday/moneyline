Phase 1 — Reach Mode (make “Reached” meaningful)
	•	Reach Mode (med priority): pre-game flag “reach” candidates (meaningful “Reached” tag)
	•	Next-tier increments spec:
	•	PTS: +5
	•	REB/AST: +2 and +4
	•	3PT: +1 and +2

Phase 2 — Trajectories / Trend overlay
	•	Trajectories (keep open): Core/Consistency trends (L10–L20, L10–Season) → future “TREND ↑/↓”

Phase 3 — Immediate ROI (strengthen Suggested Targets signal)
	1.	Opponent pace + implied totals environment (derive implied team totals from spread/ML; bump PTS/AST/REB confidence)
	2.	Stress score (0–100) from rest/travel/load (single numeric penalty/boost used everywhere)
	3.	Volatility + landmine rate (IQR/std + “finished within 1 of line” %) → trap warnings + ranking boost for stable floors
	4.	Usage proxy via team-share trends (player share of team PTS/REB/AST; last 5 vs last 20 trend)
	5.	Home/Away splits (floor + hit-rate split with sample gating)

Phase 4 — Minutes & game-script realism
	8.	Blowout sensitivity / minutes-stability 2.0 (discount/adjust confidence when big-fav blowouts clip minutes)

Phase 5 — 3PT wired into Core/Consistency
	•	3PT is integrated everywhere props exist (thresholds 1–6), but explicitly excluded from Core/Consistency for now (planned later).

Phase 6 — Cross-cutting overlay
	15.	Model-free “market alignment” on props day-of (stability tags: big favorite + healthy rotation; clustering in high implied totals)

Phase 7 — Postmortem learning (turn daily results into tuning)
	12.	Postmortem learning loop: “what worked yesterday” (hit-rate by stat, threshold, chip conditions)
	13.	Postmortem-driven auto-tune (bias thresholds down under stress; promote archetypes that “reach” often)

Phase 8 — Defensive matchup depth (optional complexity)
	14.	Defense-by-position lite (proxy-based; stepping stone before true positional tables)

⸻

Resolved (implemented + functioning live)
	•	Phase 2 — Injury-aware intelligence (most “betting-real” swing)
6. Injury impact tag baked into Suggested Targets (“next man up” boosts; opp big OUT logic)
7. Role change detector (injury-driven) (same goal as #6, but formalized as a ruleset + small boosts by archetype)
	•	Phase 4 — Market-facing UX (you’ll use it more, faster feedback loop)
9. One-click “Copy slip” parlay string (Suggested Targets → copyable text)
10. Parlay Builder mode (2–6 legs) (pin legs + naive hit prob)
11. Correlation guardrails (warn on bad combos; tag correlated legs)

Parked (low priority context only)
	•	Durability upgrades (low priority): data contracts + validation
	•	Durability upgrades (low priority): “single source of truth” mapping table for stat aliases / keys