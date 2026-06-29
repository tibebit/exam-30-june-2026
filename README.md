# Briscola RL 4 Players

This repository contains a 4-player Briscola engine and a reinforcement
learning framework for training and evaluating learnable card-playing policies.

## Python Version

The project was developed and tested with Python 3.11.

Recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

Runtime and notebook dependencies are listed in:

```text
requirements.txt
```

Current packages:

```text
numpy
torch
notebook
nbconvert
ipykernel
```

## Repository Structure

```text
game/
  Core 4-player Briscola engine, cards, observations, legal moves, and scoring.

policy/
  Fixed and learnable policies.

training/
  Episode collection, reward definitions, REINFORCE updates, self-play,
  snapshot pool, and neural training utilities.

evaluation/
  Match execution, evaluation metrics, and fixed evaluation suites.

diagnostics/
  Decision logs and diagnostic views for inspecting learner behavior.

diagnostics_ui/
  Lightweight visual interface for browsing diagnostic logs.

scripts/
  Command-line entry points for training, evaluation, diagnostics, graphs,
  and experiment protocols.

notebooks/
  Demonstration notebook

models/
  Trained checkpoints, logs, evaluation outputs, and archived snapshot runs.
```

## Main Workflows

### Run Tests

```bash
python3.11 -m unittest discover
```

### Evaluate a Checkpoint

```bash
python3.11 -B scripts/evaluate.py \
  --checkpoint path/to/checkpoint.json \
  --games 1000
```

### Use the Demo Notebook

```bash
jupyter notebook notebooks/01_training_evaluation_diagnostics_demo.ipynb
```

The notebook loads archived checkpoints, runs evaluation tables, generates
graphs, and opens diagnostic views.

## AI Usage Disclosure

AI tools were used only for auxiliary development and presentation tasks,
including:

- support in formatting comments and explanatory text;
- assistance in generating plotting utilities and visualization scripts;
- support in building the diagnostics workflow and the HTML
  diagnostics interface.

The implementation, experimental setup, and reported results were manually
reviewed and verified.
