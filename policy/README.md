# policy

This folder contains policy interfaces and baseline policies for four-player
Briscola.

Policies act only from a legal `Osservazione` and return legal cards from the
current hand.

## Interface

All policies implement the minimal `Policy` protocol:

- `name`: policy identifier;
- `action_probabilities(osservazione)`: probability assigned to each legal card;
- `select_action(osservazione, rng, greedy=False)`: selected legal card.

## Policies

- `RandomPolicy`: uniform random choice among legal cards.
- `GreedyPolicy`: myopic baseline; takes with the least costly sufficient card,
  otherwise discards the least costly card.
- `HeuristicPolicy`: minimal team-aware baseline; when the partner is taking,
  avoids spending valuable cards.
- `AdvancedHeuristicPolicy`: explicit rule-based heuristic for richer team-aware
  play. It separates cases by current winner, player position in the trick, and
  trick value.
- `PerfectHeuristicPolicy`: highly detailed rule-based heuristic. It classifies
  cards into specific categories (e.g., liscio, punticini, carico, taglietto)
  and evaluates the best card to play using strict priority lists. The decision
  tree considers 6 main scenarios, branching further based on who is currently
  winning, the player's position in the trick (1st, 2nd, 3rd, or 4th), and the
  points or specific cards already on the table.
- `LinearSoftmaxPolicy`: learnable policy with linear action preferences and a
  stable softmax over legal cards. It supports stochastic sampling and greedy
  argmax selection.

## Features

- `BriscolaFeatureExtractor`: builds numeric features from a legal
  `Osservazione` and a legal candidate card for learnable policies, without
  using hidden game state.

## Tests

Run policy tests from the repository root:

```bash
python3 -B -m unittest discover -s policy/tests
```
