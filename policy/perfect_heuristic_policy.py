"""Policy euristica perfetta basata su regole esplicite."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from game.rules import punti_presa, vincitore_presa


@dataclass
class PerfectHeuristicPolicy:
    """Policy basata su euristiche perfette."""

    name: str = "perfect_heuristic"

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Distribuisce probabilita' uniforme sulle carte migliori."""
        carte_migliori = self._carte_migliori(osservazione)
        if not carte_migliori:
            carte_migliori = list(osservazione.azioni_legali)
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
        """Sceglie casualmente tra le carte migliori."""
        carte_migliori = self._carte_migliori(osservazione)
        if not carte_migliori:
            carte_migliori = list(osservazione.azioni_legali)
        return rng.choice(carte_migliori)

    def _carte_migliori(self, osservazione: Osservazione) -> list[Carta]:
        """Smista l'osservazione nel ramo di regole adatto alla presa."""
        azioni_legali = list(osservazione.azioni_legali)
        if not azioni_legali:
            raise ValueError("No legal actions available")

        if len(azioni_legali) == 1:
            return azioni_legali

        presa_vuota = not osservazione.carte_sul_campo
        pos = osservazione.posizione_nella_presa  # 0, 1, 2, 3

        if presa_vuota and pos == 0:
            return self._caso_1(osservazione, azioni_legali)
        
        # We need a fallback if not presa_vuota but carte_sul_campo somehow empty? That's contradictory.
        if presa_vuota:
            # According to rules, only learner = 1st is covered for empty trick.
            return self._select(azioni_legali, "-")

        vincitore = self._vincitore_corrente(osservazione)
        compagno_vince = vincitore == osservazione.compagno_id
        avversario_vince = vincitore in osservazione.avversari

        if compagno_vince:
            if pos == 3:
                return self._caso_2(osservazione, azioni_legali)
            elif pos == 2:
                return self._caso_3(osservazione, azioni_legali)
        elif avversario_vince:
            if pos == 3:
                return self._caso_4(osservazione, azioni_legali)
            elif pos == 1:
                return self._caso_5(osservazione, azioni_legali)
            elif pos == 2:
                return self._caso_6(osservazione, azioni_legali)

        # Fallback di sicurezza
        return self._select(azioni_legali, "-")

    # --- CATEGORIZZAZIONE CARTE ---

    def _is_liscio(self, c: Carta, sb: str) -> bool:
        return c.punti == 0 and c.seme != sb

    def _is_punticino(self, c: Carta, sb: str) -> bool:
        return c.punti in (2, 3, 4) and c.seme != sb

    def _is_carico(self, c: Carta, sb: str) -> bool:
        return c.punti >= 10 and c.seme != sb

    def _is_taglietto(self, c: Carta, sb: str) -> bool:
        return c.punti == 0 and c.seme == sb

    def _is_briscola(self, c: Carta, sb: str) -> bool:
        return c.seme == sb

    def _is_briscola_alta(self, c: Carta, sb: str) -> bool:
        return c.punti > 0 and c.seme == sb

    def _is_3_di_briscola(self, c: Carta, sb: str) -> bool:
        return c.rango == "tre" and c.seme == sb

    # --- RISOLUZIONE PRESA ---

    def _carta_vincente_corrente(self, o: Osservazione) -> Carta:
        return vincitore_presa(o.carte_sul_campo, o.seme_briscola).carta

    def _vincitore_corrente(self, o: Osservazione) -> int:
        return vincitore_presa(o.carte_sul_campo, o.seme_briscola).giocatore_id

    def _vincitore_dopo_carta(self, o: Osservazione, c: Carta) -> int:
        presa_candidata = tuple(o.carte_sul_campo) + (CartaGiocata(o.giocatore_id, c),)
        return vincitore_presa(presa_candidata, o.seme_briscola).giocatore_id

    def _prende(self, o: Osservazione, c: Carta) -> bool:
        return self._vincitore_dopo_carta(o, c) == o.giocatore_id

    def _non_prende(self, o: Osservazione, c: Carta) -> bool:
        return not self._prende(o, c)

    # --- SELETTORI (+), (-), * ---

    def _applica_asterisco(self, carte: list[Carta]) -> list[Carta]:
        if len(carte) >= 3:
            counts = Counter(c.seme for c in carte)
            semi_maggiori = {s for s, count in counts.items() if count >= 2}
            if semi_maggiori:
                carte = [c for c in carte if c.seme in semi_maggiori]
        return carte

    def _select(self, carte: list[Carta], sign: str, asterisco: bool = False) -> list[Carta]:
        if not carte:
            return []
        if asterisco:
            carte = self._applica_asterisco(carte)
        
        if sign == "+":
            val = max(c.forza for c in carte)
        elif sign == "-":
            val = min(c.forza for c in carte)
        else:
            raise ValueError(f"Sign sconosciuto {sign}")
            
        return [c for c in carte if c.forza == val]

    # --- CASISTICHE ---

    def _caso_1(self, o: Osservazione, azioni: list[Carta]) -> list[Carta]:
        sb = o.seme_briscola

        lisci = [c for c in azioni if self._is_liscio(c, sb)]
        if res := self._select(lisci, "-", asterisco=True): return res

        taglietti = [c for c in azioni if self._is_taglietto(c, sb)]
        if len(taglietti) >= 2:
            if res := self._select(taglietti, "-"): return res

        punticini = [c for c in azioni if self._is_punticino(c, sb)]
        if res := self._select(punticini, "-", asterisco=True): return res

        carichi = [c for c in azioni if self._is_carico(c, sb)]
        briscole_alte = [c for c in azioni if self._is_briscola_alta(c, sb)]
        if len(carichi) == 2 and len(briscole_alte) == 1:
            if res := self._select(carichi, "-"): return res

        briscole = [c for c in azioni if self._is_briscola(c, sb)]
        if res := self._select(briscole, "-"): return res

        if res := self._select(carichi, "-"): return res

        return self._select(azioni, "-")

    def _caso_2(self, o: Osservazione, azioni: list[Carta]) -> list[Carta]:
        sb = o.seme_briscola
        
        carichi = [c for c in azioni if self._is_carico(c, sb)]
        if res := self._select(carichi, "+"): return res

        punticini = [c for c in azioni if self._is_punticino(c, sb)]
        if res := self._select(punticini, "+"): return res

        lisci = [c for c in azioni if self._is_liscio(c, sb)]
        if res := self._select(lisci, "-"): return res

        briscole = [c for c in azioni if self._is_briscola(c, sb)]
        if res := self._select(briscole, "-"): return res

        return self._select(azioni, "-")

    def _caso_3(self, o: Osservazione, azioni: list[Carta]) -> list[Carta]:
        sb = o.seme_briscola
        carta_vincente = self._carta_vincente_corrente(o)
        
        if self._is_briscola_alta(carta_vincente, sb):
            carichi = [c for c in azioni if self._is_carico(c, sb)]
            if res := self._select(carichi, "+"): return res

        if self._is_taglietto(carta_vincente, sb):
            punticini = [c for c in azioni if self._is_punticino(c, sb)]
            if res := self._select(punticini, "+"): return res

        punti = punti_presa(o.carte_sul_campo)
        if punti >= 15:
            ba = [c for c in azioni if self._is_briscola_alta(c, sb)]
            if res := self._select(ba, "-"): return res
        
        if 8 <= punti <= 14:
            taglietti = [c for c in azioni if self._is_taglietto(c, sb)]
            if res := self._select(taglietti, "+"): return res

        lisci = [c for c in azioni if self._is_liscio(c, sb)]
        if res := self._select(lisci, "-"): return res

        return self._select(azioni, "-")

    def _caso_4(self, o: Osservazione, azioni: list[Carta]) -> list[Carta]:
        sb = o.seme_briscola
        punti = punti_presa(o.carte_sul_campo)
        
        if 0 <= punti <= 4:
            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "+"): return res

            punticini_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            punticini_p_5 = [c for c in punticini_p if punti + c.punti >= 5]
            if res := self._select(punticini_p_5, "+"): return res

            l_np = [c for c in azioni if self._is_liscio(c, sb) and self._non_prende(o, c)]
            if res := self._select(l_np, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            if res := self._select(punticini_p, "+"): return res

            l_p = [c for c in azioni if self._is_liscio(c, sb) and self._prende(o, c)]
            if res := self._select(l_p, "-"): return res

            b_np = [c for c in azioni if self._is_briscola(c, sb) and self._non_prende(o, c)]
            if res := self._select(b_np, "-"): return res

            b_p = [c for c in azioni if self._is_briscola(c, sb) and self._prende(o, c)]
            if res := self._select(b_p, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res
            
        elif 5 <= punti <= 10:
            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "+"): return res

            p_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            if res := self._select(p_p, "+"): return res

            l_p = [c for c in azioni if self._is_liscio(c, sb) and self._prende(o, c)]
            if res := self._select(l_p, "-"): return res

            t_p = [c for c in azioni if self._is_taglietto(c, sb) and self._prende(o, c)]
            if res := self._select(t_p, "-"): return res

            l_np = [c for c in azioni if self._is_liscio(c, sb) and self._non_prende(o, c)]
            if res := self._select(l_np, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            t_np = [c for c in azioni if self._is_taglietto(c, sb) and self._non_prende(o, c)]
            if res := self._select(t_np, "-"): return res

            ba_p = [c for c in azioni if self._is_briscola_alta(c, sb) and self._prende(o, c)]
            if res := self._select(ba_p, "-"): return res

            ba_np = [c for c in azioni if self._is_briscola_alta(c, sb) and self._non_prende(o, c)]
            if res := self._select(ba_np, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

        else: # maggiori di 10
            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "+"): return res

            p_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            if res := self._select(p_p, "+"): return res

            l_p = [c for c in azioni if self._is_liscio(c, sb) and self._prende(o, c)]
            if res := self._select(l_p, "-"): return res

            b_p = [c for c in azioni if self._is_briscola(c, sb) and self._prende(o, c)]
            if res := self._select(b_p, "-"): return res

            l_np = [c for c in azioni if self._is_liscio(c, sb) and self._non_prende(o, c)]
            if res := self._select(l_np, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            b_np_no3 = [c for c in azioni if self._is_briscola(c, sb) and self._non_prende(o, c) and not self._is_3_di_briscola(c, sb)]
            if res := self._select(b_np_no3, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

            tre_briscola = [c for c in azioni if self._is_3_di_briscola(c, sb)]
            if res := self._select(tre_briscola, "-"): return res

        return self._select(azioni, "-")

    def _caso_5(self, o: Osservazione, azioni: list[Carta]) -> list[Carta]:
        sb = o.seme_briscola
        carta_vincente = self._carta_vincente_corrente(o)
        
        # Caso A: sul tavolo c'è un carico
        if self._is_carico(carta_vincente, sb):
            ba = [c for c in azioni if self._is_briscola_alta(c, sb)]
            if res := self._select(ba, "-"): return res

            t = [c for c in azioni if self._is_taglietto(c, sb)]
            if res := self._select(t, "-"): return res

            l = [c for c in azioni if self._is_liscio(c, sb)]
            if res := self._select(l, "-"): return res

            p = [c for c in azioni if self._is_punticino(c, sb)]
            if res := self._select(p, "-"): return res

            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := c_p: return res  # no (+)/(-) in rule

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

        # Caso B: sul tavolo c'è una briscola
        elif self._is_briscola(carta_vincente, sb):
            l = [c for c in azioni if self._is_liscio(c, sb)]
            if res := self._select(l, "-"): return res

            t_p = [c for c in azioni if self._is_taglietto(c, sb) and self._prende(o, c)]
            if res := self._select(t_p, "-"): return res

            p = [c for c in azioni if self._is_punticino(c, sb)]
            if res := self._select(p, "-"): return res

            t_np = [c for c in azioni if self._is_taglietto(c, sb) and self._non_prende(o, c)]
            if res := self._select(t_np, "-"): return res

            ba_p = [c for c in azioni if self._is_briscola_alta(c, sb) and self._prende(o, c)]
            if res := self._select(ba_p, "-"): return res

            ba_np_no3 = [c for c in azioni if self._is_briscola_alta(c, sb) and self._non_prende(o, c) and not self._is_3_di_briscola(c, sb)]
            if res := self._select(ba_np_no3, "-"): return res

            carichi = [c for c in azioni if self._is_carico(c, sb)]
            if res := self._select(carichi, "-"): return res

            tre_briscola = [c for c in azioni if self._is_3_di_briscola(c, sb)]
            if res := self._select(tre_briscola, "-"): return res

        # Caso C: sul tavolo ci sono punticini
        elif self._is_punticino(carta_vincente, sb):
            p_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            if res := self._select(p_p, "+"): return res

            l = [c for c in azioni if self._is_liscio(c, sb)]
            if res := self._select(l, "-"): return res

            t = [c for c in azioni if self._is_taglietto(c, sb)]
            if res := self._select(t, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "-"): return res

            ba = [c for c in azioni if self._is_briscola_alta(c, sb)]
            if res := self._select(ba, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

        # Caso D: sul tavolo c'è un liscio
        elif self._is_liscio(carta_vincente, sb):
            p_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            if res := self._select(p_p, "+"): return res

            l_p = [c for c in azioni if self._is_liscio(c, sb) and self._prende(o, c)]
            if res := self._select(l_p, "+"): return res

            l_np = [c for c in azioni if self._is_liscio(c, sb) and self._non_prende(o, c)]
            if res := self._select(l_np, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            t = [c for c in azioni if self._is_taglietto(c, sb)]
            if res := self._select(t, "-"): return res

            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "-"): return res

            ba = [c for c in azioni if self._is_briscola_alta(c, sb)]
            if res := self._select(ba, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

        return self._select(azioni, "-")

    def _caso_6(self, o: Osservazione, azioni: list[Carta]) -> list[Carta]:
        sb = o.seme_briscola
        punti = punti_presa(o.carte_sul_campo)
        
        if 0 <= punti <= 4:
            p_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            if res := self._select(p_p, "+"): return res

            l_np = [c for c in azioni if self._is_liscio(c, sb) and self._non_prende(o, c)]
            if res := self._select(l_np, "-"): return res

            l_p = [c for c in azioni if self._is_liscio(c, sb) and self._prende(o, c)]
            if res := self._select(l_p, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            t_np = [c for c in azioni if self._is_taglietto(c, sb) and self._non_prende(o, c)]
            if res := self._select(t_np, "-"): return res

            t_p = [c for c in azioni if self._is_taglietto(c, sb) and self._prende(o, c)]
            if res := self._select(t_p, "-"): return res

            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "-"): return res

            ba_p = [c for c in azioni if self._is_briscola_alta(c, sb) and self._prende(o, c)]
            if res := self._select(ba_p, "-"): return res

            ba_np_no3 = [c for c in azioni if self._is_briscola_alta(c, sb) and self._non_prende(o, c) and not self._is_3_di_briscola(c, sb)]
            if res := self._select(ba_np_no3, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

            tre_briscola = [c for c in azioni if self._is_3_di_briscola(c, sb)]
            if res := self._select(tre_briscola, "-"): return res

        elif 5 <= punti <= 8:
            p_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            if res := self._select(p_p, "+"): return res

            l_p = [c for c in azioni if self._is_liscio(c, sb) and self._prende(o, c)]
            if res := self._select(l_p, "+"): return res

            l_np = [c for c in azioni if self._is_liscio(c, sb) and self._non_prende(o, c)]
            if res := self._select(l_np, "-"): return res

            t_p = [c for c in azioni if self._is_taglietto(c, sb) and self._prende(o, c)]
            if res := self._select(t_p, "-"): return res

            t_np = [c for c in azioni if self._is_taglietto(c, sb) and self._non_prende(o, c)]
            if res := self._select(t_np, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            ba_p = [c for c in azioni if self._is_briscola_alta(c, sb) and self._prende(o, c)]
            if res := self._select(ba_p, "-"): return res

            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "-"): return res

            ba_np_no3 = [c for c in azioni if self._is_briscola_alta(c, sb) and self._non_prende(o, c) and not self._is_3_di_briscola(c, sb)]
            if res := self._select(ba_np_no3, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

            tre_briscola = [c for c in azioni if self._is_3_di_briscola(c, sb)]
            if res := self._select(tre_briscola, "-"): return res

        else: # maggiori di 9
            ba_p = [c for c in azioni if self._is_briscola_alta(c, sb) and self._prende(o, c)]
            if res := self._select(ba_p, "-"): return res

            t_p = [c for c in azioni if self._is_taglietto(c, sb) and self._prende(o, c)]
            if res := self._select(t_p, "+"): return res

            p_p = [c for c in azioni if self._is_punticino(c, sb) and self._prende(o, c)]
            if res := self._select(p_p, "+"): return res

            l_p = [c for c in azioni if self._is_liscio(c, sb) and self._prende(o, c)]
            if res := self._select(l_p, "+"): return res

            l_np = [c for c in azioni if self._is_liscio(c, sb) and self._non_prende(o, c)]
            if res := self._select(l_np, "-"): return res

            p_np = [c for c in azioni if self._is_punticino(c, sb) and self._non_prende(o, c)]
            if res := self._select(p_np, "-"): return res

            t_np = [c for c in azioni if self._is_taglietto(c, sb) and self._non_prende(o, c)]
            if res := self._select(t_np, "-"): return res

            c_p = [c for c in azioni if self._is_carico(c, sb) and self._prende(o, c)]
            if res := self._select(c_p, "-"): return res

            ba_np_no3 = [c for c in azioni if self._is_briscola_alta(c, sb) and self._non_prende(o, c) and not self._is_3_di_briscola(c, sb)]
            if res := self._select(ba_np_no3, "-"): return res

            c_np = [c for c in azioni if self._is_carico(c, sb) and self._non_prende(o, c)]
            if res := self._select(c_np, "-"): return res

            tre_briscola = [c for c in azioni if self._is_3_di_briscola(c, sb)]
            if res := self._select(tre_briscola, "-"): return res

        return self._select(azioni, "-")
