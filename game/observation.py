"""Legal observation for a player."""

from __future__ import annotations

from dataclasses import dataclass

from .cards import Carta, CartaGiocata


@dataclass(frozen=True)
class Osservazione:
    """Information legally available to a player."""

    giocatore_id: int
    compagno_id: int
    avversario_sinistro_id: int
    avversario_destro_id: int
    mano: tuple[Carta, ...]
    mano_compagno_visibile: bool
    mano_compagno: tuple[Carta, ...]
    seme_briscola: str
    briscola_esposta: Carta
    proprietario_briscola_esposta: int | None
    carte_sul_campo: tuple[CartaGiocata, ...]
    carte_giocate: tuple[CartaGiocata, ...]
    vincitori_prese: tuple[int, ...]
    squadra: str
    squadra_avversaria: str
    punteggio_squadra: int
    punteggio_avversari: int
    primo_giocatore_presa: int
    giocatore_corrente: int
    carte_nel_mazzo: int
    indice_presa: int
    posizione_nella_presa: int

    @property
    def azioni_legali(self) -> tuple[Carta, ...]:
        return self.mano

    @property
    def numero_carte_giocate(self) -> int:
        return len(self.carte_giocate)

    @property
    def avversari(self) -> tuple[int, int]:
        return (self.avversario_sinistro_id, self.avversario_destro_id)

    @property
    def punteggi(self) -> dict[str, int]:
        return {
            self.squadra: self.punteggio_squadra,
            self.squadra_avversaria: self.punteggio_avversari,
        }
