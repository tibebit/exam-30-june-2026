#!/usr/bin/env python3
"""Print a decision diagnostics JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a readable trace from a Briscola decision diagnostics JSON.",
    )
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument(
        "--player",
        type=int,
        default=None,
        help="Show only decisions made by this player id.",
    )
    parser.add_argument(
        "--our-player",
        type=int,
        default=None,
        help="Player id to mark as ours. Defaults to learner_giocatore_id in the report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of decisions to print.",
    )
    parser.add_argument(
        "--show-probs",
        action="store_true",
        help="Print action probabilities for the cards in hand.",
    )
    args = parser.parse_args()
    if args.player is not None and args.player not in range(4):
        parser.error("--player deve essere tra 0 e 3")
    if args.our_player is not None and args.our_player not in range(4):
        parser.error("--our-player deve essere tra 0 e 3")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit deve essere positivo")
    return args


def main() -> None:
    args = parse_args()
    report = load_report(args.diagnostics)
    decision_log = report["decision_log"]
    our_player = args.our_player
    if our_player is None:
        our_player = report.get("learner_giocatore_id")
    records = filtered_records(
        decision_log["records"],
        player=args.player,
        limit=args.limit,
    )

    print_header(
        report,
        decision_log,
        shown_records=len(records),
        our_player=our_player,
    )
    for record in records:
        print_record(record, show_probs=args.show_probs, our_player=our_player)
    print_final_points(decision_log)


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def filtered_records(
    records: list[dict[str, Any]],
    *,
    player: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if player is None or record["giocatore_id"] == player
    ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def print_header(
    report: dict[str, Any],
    decision_log: dict[str, Any],
    *,
    shown_records: int,
    our_player: int | None,
) -> None:
    records = decision_log["records"]
    briscola = records[0]["briscola_esposta"] if records else None
    seme_briscola = records[0]["seme_briscola"] if records else "?"
    print("Decision diagnostics")
    print(f"  file kind: {report.get('kind', 'unknown')}")
    print(f"  checkpoint: {report.get('checkpoint_path', '-')}")
    print(f"  seed ambiente: {decision_log['seed_ambiente']}")
    print(f"  seed policy: {decision_log['seed_policy']}")
    print(f"  greedy: {decision_log['greedy']}")
    print(f"  our player: {player_label(our_player)}")
    print(f"  briscola: {seme_briscola} ({card_label(briscola)})")
    print(f"  shown decisions: {shown_records}/{len(records)}")
    print()


def print_record(
    record: dict[str, Any],
    *,
    show_probs: bool,
    our_player: int | None,
) -> None:
    osservazione = record["osservazione"]
    outcome = record["outcome"]
    is_ours = record["giocatore_id"] == our_player
    turn_label = decision_owner_label(record, our_player)
    print(
        f"step {record['step_index']:02d} | "
        f"trick {osservazione['indice_presa']} "
        f"card {osservazione['posizione_nella_presa'] + 1}/4 | "
        f"{turn_label}"
    )
    print(
        f"  acting player: P{record['giocatore_id']} "
        f"team={osservazione['squadra']} "
        f"policy={record['policy_name']}"
    )
    print(f"  briscola: {record['seme_briscola']} ({card_label(record['briscola_esposta'])})")
    print(f"  score: {osservazione['punteggio_squadra']}-{osservazione['punteggio_avversari']}")
    print(f"  table: {format_played_cards(record['carte_sul_campo'])}")
    print(f"  hand: {format_cards(record['mano'])}")
    print(f"  chose: {card_label(record['azione'])}")
    if show_probs:
        print(f"  probs: {format_probabilities(record)}")
    if outcome["presa_completata"]:
        print(
            "  trick: "
            f"winner={outcome['vincitore_presa']} "
            f"points={outcome['punti_presa']} "
            f"cards={format_played_cards(outcome['carte_presa_completata'])}"
        )
    print()


def format_score(score: dict[str, int]) -> str:
    return ", ".join(f"{team}={points}" for team, points in score.items())


def format_played_cards(played_cards: list[dict[str, Any]]) -> str:
    if not played_cards:
        return "-"
    return ", ".join(
        f"P{played['giocatore_id']}:{card_label(played['carta'])}"
        for played in played_cards
    )


def format_cards(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "-"
    return ", ".join(card_label(card) for card in cards)


def format_probabilities(record: dict[str, Any]) -> str:
    probabilities = record["action_probabilities"]
    parts = []
    for card in record["azioni_legali"]:
        probability = probabilities[card["id"]]
        parts.append(f"{card_label(card)}={probability:.3f}")
    return ", ".join(parts)


def card_label(card: dict[str, Any] | None) -> str:
    if card is None:
        return "?"
    return card["id"]


def player_label(player: int | None) -> str:
    if player is None:
        return "unknown"
    return f"P{player}"


def print_final_points(decision_log: dict[str, Any]) -> None:
    print("Final points")
    print(f"  {format_score(decision_log['punteggi_finali'])}")
    print(f"  winner={decision_log['squadra_vincitrice']}")


def decision_owner_label(record: dict[str, Any], our_player: int | None) -> str:
    if our_player is None:
        return "turn"
    acting_player = record["giocatore_id"]
    if acting_player == our_player:
        return "OUR TURN"
    if acting_player % 2 == our_player % 2:
        return "partner turn"
    return "opponent turn"


if __name__ == "__main__":
    main()
