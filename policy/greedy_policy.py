"""Greedy baseline policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from game.rules import vincitore_presa


@dataclass
class GreedyPolicy:
    """Myopic policy: takes with the least costly card, otherwise discards."""

    name: str = "greedy"

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        carte_migliori = self._carte_migliori(osservazione)
        probabilita = 1.0 / len(carte_migliori)
        return {
            carta: probabilita if carta in carte_migliori else 0.0
            for carta in osservazione.azioni_legali
        }

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        return rng.choice(self._carte_migliori(osservazione))

    def _carte_migliori(self, osservazione: Osservazione) -> list[Carta]:
        azioni_legali = osservazione.azioni_legali
        if not azioni_legali:
            raise ValueError("No legal actions available")

        carte_che_prendono = [
            carta for carta in azioni_legali if self._carta_prende(osservazione, carta)
        ]
        candidate = carte_che_prendono or list(azioni_legali)
        costo_minimo = min(self._costo_carta(osservazione, carta) for carta in candidate)
        return [
            carta
            for carta in candidate
            if self._costo_carta(osservazione, carta) == costo_minimo
        ]

    def _carta_prende(self, osservazione: Osservazione, carta: Carta) -> bool:
        presa_candidata = tuple(osservazione.carte_sul_campo) + (
            CartaGiocata(giocatore_id=osservazione.giocatore_id, carta=carta),
        )
        vincitore = vincitore_presa(
            presa_candidata,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id == osservazione.giocatore_id

    def _costo_carta(self, osservazione: Osservazione, carta: Carta) -> tuple[int, bool, int]:
        return (
            carta.punti,
            carta.seme == osservazione.seme_briscola,
            carta.forza,
        )
