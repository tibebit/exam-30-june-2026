# Diagnostics

Utilities for inspecting policy decisions from legal observations.

## Files

- `decision_log.py`: records one complete game as a sequence of decisions.
- `views.py`: filters decision logs into readable subsets.

## Decision Log

`record_decision_log` stores, for each decision:

- player id;
- policy name;
- legal observation before the action;
- legal actions;
- action probabilities;
- chosen action;
- action-selection mode;
- public outcome after the action.

The log stores public outcomes and legal observations only.

## Views

`views.py` provides filters for:

- player;
- policy name;
- trick position;
- partner currently leading;
- opponent currently leading;
- rich tricks;
- chosen actions below a probability threshold.
