"""Advanced heuristic policy based on explicit rules."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from game.rules import punti_presa, vincitore_presa


@dataclass
class AdvancedHeuristicPolicy:
    """Team-aware policy for explicit presa cases."""

    name: str = "advanced_heuristic"

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Assign uniform probability to the best cards."""

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
        """Randomly choose among the best cards to preserve ties."""

        return rng.choice(self._carte_migliori(osservazione))

    def _carte_migliori(self, osservazione: Osservazione) -> list[Carta]:
        """Route the observation to the rule branch suited to the presa."""

        azioni_legali = list(osservazione.azioni_legali)
        if not azioni_legali:
            raise ValueError("No legal actions available")

        if not osservazione.carte_sul_campo:
            return self._minime(
                azioni_legali,
                lambda carta: self._costo_apertura(osservazione, carta),
            )

        vincitore = self._vincitore_corrente(osservazione)
        if vincitore == osservazione.compagno_id:
            if self._ultimo_di_mano(osservazione):
                return self._carte_compagno_prende_ultimo(osservazione, azioni_legali)
            return self._carte_compagno_prende_non_ultimo(osservazione, azioni_legali)

        if vincitore in osservazione.avversari:
            if self._ultimo_di_mano(osservazione):
                return self._carte_avversario_prende_ultimo(osservazione, azioni_legali)
            return self._carte_avversario_prende_non_ultimo(osservazione, azioni_legali)

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_compagno_prende_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """Load safe points when the team cannot be overtaken."""

        carte_che_salvano_presa = [
            carta
            for carta in azioni_legali
            if self._team_prende_dopo_carta(osservazione, carta)
        ]
        candidate = carte_che_salvano_presa or azioni_legali

        carichi_non_briscola = [
            carta
            for carta in candidate
            if self._carico(carta) and not self._briscola(osservazione, carta)
        ]
        if carichi_non_briscola:
            return self._massime(
                carichi_non_briscola,
                lambda carta: (carta.punti, -carta.forza),
            )

        non_briscola = [
            carta for carta in candidate if not self._briscola(osservazione, carta)
        ]
        if non_briscola:
            return self._minime(
                non_briscola,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        return self._minime(
            candidate,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_compagno_prende_non_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """Keep the partner ahead while spending as little as possible."""

        carte_che_lasciano_compagno = [
            carta
            for carta in azioni_legali
            if self._vincitore_dopo_carta(osservazione, carta)
            == osservazione.compagno_id
        ]
        if carte_che_lasciano_compagno:
            return self._minime(
                carte_che_lasciano_compagno,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_avversario_prende_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """When last to play, seek points using non-briscola cards when possible."""

        carte_che_prendono = self._carte_che_prendono(osservazione, azioni_legali)
        if not carte_che_prendono:
            return self._minime(
                azioni_legali,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        carichi_non_briscola = [
            carta
            for carta in carte_che_prendono
            if self._carico(carta) and not self._briscola(osservazione, carta)
        ]
        if carichi_non_briscola:
            return self._massime(
                carichi_non_briscola,
                lambda carta: (carta.punti, -carta.forza),
            )

        if self._presa_ricca(osservazione):
            return self._minime(
                carte_che_prendono,
                lambda carta: self._costo_presa_ultimo(osservazione, carta),
            )

        non_briscola = [
            carta
            for carta in carte_che_prendono
            if not self._briscola(osservazione, carta)
        ]
        if non_briscola:
            return self._minime(
                non_briscola,
                lambda carta: self._costo_presa_ultimo(osservazione, carta),
            )

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_avversario_prende_non_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """Take cautiously because an opponent can still overtake."""

        carte_che_prendono = self._carte_che_prendono(osservazione, azioni_legali)
        non_briscola_non_carico = [
            carta
            for carta in carte_che_prendono
            if not self._briscola(osservazione, carta) and not self._carico(carta)
        ]

        if self._presa_ricca(osservazione):
            briscole = [
                carta
                for carta in carte_che_prendono
                if self._briscola(osservazione, carta)
            ]
            if briscole:
                return self._minime(briscole, self._costo_briscola_bassa)

            if non_briscola_non_carico:
                return self._minime(non_briscola_non_carico, self._costo_presa_povera)

            return self._minime(
                azioni_legali,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        if non_briscola_non_carico:
            return self._minime(non_briscola_non_carico, self._costo_presa_povera)

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_che_prendono(
        self,
        osservazione: Osservazione,
        carte: list[Carta],
    ) -> list[Carta]:
        """Filter cards that make the current player the winner."""

        return [
            carta
            for carta in carte
            if self._vincitore_dopo_carta(osservazione, carta)
            == osservazione.giocatore_id
        ]

    def _team_prende_dopo_carta(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> bool:
        """Check whether the presa stays with the current team after the card."""

        return self._vincitore_dopo_carta(osservazione, carta) in (
            osservazione.giocatore_id,
            osservazione.compagno_id,
        )

    def _vincitore_corrente(self, osservazione: Osservazione) -> int:
        """Compute the provisional winner before the current card."""

        vincitore = vincitore_presa(
            osservazione.carte_sul_campo,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id

    def _vincitore_dopo_carta(self, osservazione: Osservazione, carta: Carta) -> int:
        """Compute the provisional winner after a candidate card."""

        presa_candidata = tuple(osservazione.carte_sul_campo) + (
            CartaGiocata(giocatore_id=osservazione.giocatore_id, carta=carta),
        )
        vincitore = vincitore_presa(
            presa_candidata,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id

    def _ultimo_di_mano(self, osservazione: Osservazione) -> bool:
        """Detect when no one else will play after this card."""

        return osservazione.posizione_nella_presa == 3

    def _presa_ricca(self, osservazione: Osservazione) -> bool:
        """Treat a presa with at least ten points on the table as rich."""

        return punti_presa(osservazione.carte_sul_campo) >= 10

    def _briscola(self, osservazione: Osservazione, carta: Carta) -> bool:
        """Check whether the card belongs to the briscola suit."""

        return carta.seme == osservazione.seme_briscola

    def _carico(self, carta: Carta) -> bool:
        """Recognize asso and tre through their point value."""

        return carta.punti >= 10

    def _costo_apertura(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> tuple[int, bool, int]:
        """Rank opening discards: few points, non-briscola, low strength."""

        return (
            carta.punti,
            self._briscola(osservazione, carta),
            carta.forza,
        )

    def _costo_danno(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> tuple[int, bool, bool, int]:
        """Rank damage: few points, non-carico, non-briscola, low strength."""

        return (
            carta.punti,
            self._carico(carta),
            self._briscola(osservazione, carta),
            carta.forza,
        )

    def _costo_presa_ultimo(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> tuple[bool, int, int]:
        """When last, prefer taking without briscola and at low cost."""

        return (
            self._briscola(osservazione, carta),
            carta.punti,
            carta.forza,
        )

    def _costo_briscola_bassa(self, carta: Carta) -> tuple[int, int]:
        """Choose the cheapest briscola to protect a rich presa."""

        return (carta.punti, carta.forza)

    def _costo_presa_povera(self, carta: Carta) -> tuple[int, int]:
        """Choose the cheapest poor presa among already admissible cards."""

        return (carta.punti, carta.forza)

    def _minime(self, carte: list[Carta], key) -> list[Carta]:
        """Return all best cards according to a priority order."""

        valore_minimo = min(key(carta) for carta in carte)
        return [carta for carta in carte if key(carta) == valore_minimo]

    def _massime(self, carte: list[Carta], key) -> list[Carta]:
        """Return all worst cards according to a priority order."""

        valore_massimo = max(key(carta) for carta in carte)
        return [carta for carta in carte if key(carta) == valore_massimo]
