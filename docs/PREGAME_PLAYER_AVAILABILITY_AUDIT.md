# Pregame QB-availability evidence — October 2026

This change fixes a meaningful status-parser bug in `harbin/context.py`: previously an injury designation of **Inactive** could match the substring **active** and be treated as healthy. Definitive negative statuses now take priority. Regression tests cover Inactive, Unavailable, and Not cleared.

Roster summary no longer counts a QB with unknown active status as **confirmed** active; `qb_status_unknown_count` and `qb_status_verified_count` expose the distinction. Injury summaries additionally expose `qb_injury_evidence_status` (FRESH, STALE, UNKNOWN_TIMESTAMP, NO_QB_REPORT) and timestamp coverage. Live context carries both per-team values and `qb_starter_verified=False` because the current free roster and injury feeds **do not establish a named, confirmed game-day starter**. These fields are diagnostic and are not model score adjustments or validation of the injury model's betting effect.

Historical-season context is disabled under the existing engine rules. Do not infer starter confirmations from these fields; that requires an independent named-starter feed or first-seen, pregame manual observations with immutable timestamps. Actual betting readiness remains governed by existing release gates; no EA ratings or postgame information are imported.
