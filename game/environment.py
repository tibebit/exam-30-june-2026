"""Game environment for four-player Briscola."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .cards import Carta, CartaGiocata, crea_mazzo
from .observation import Osservazione
from .rules import (
    NUMERO_GIOCATORI,
    SQUADRA_DISPARI,
    SQUADRA_PARI,
    avversario_destro_di,
    avversario_sinistro_di,
    compagno_di,
    giocatore_successivo,
    ordine_giocatori_da,
    punti_presa,
    squadra_avversaria_di,
    squadra_di,
    valida_giocatore_id,
    vincitore_presa,
)


@dataclass(frozen=True)
class CartaPescata:
    """Observable pescata event that does not reveal hidden cards."""

    giocatore_id: int
    carta_visibile: Carta | None


@dataclass(frozen=True)
class EventoPubblico:
    """Public event useful for replays, logs, and diagnostics."""

    tipo: str
    giocatore_id: int | None = None
    carta: Carta | None = None
    punti: int | None = None


@dataclass(frozen=True)
class EsitoMossa:
    """Result produced by a legal move."""

    osservazione: Osservazione | None
    partita_finita: bool
    presa_completata: bool
    carte_presa_completata: tuple[CartaGiocata, ...]
    vincitore_presa: int | None
    punti_presa: int
    carte_pescate: tuple[CartaPescata, ...]
    prossimo_giocatore: int | None
    punteggi: dict[str, int]
    eventi_pubblici: tuple[EventoPubblico, ...]


@dataclass
class Ambiente:
    """Internal state and progression of a four-player Briscola game."""

    seed: int | None = None
    primo_giocatore_id: int = 0
    rng: random.Random = field(init=False)
    mani: list[list[Carta]] = field(init=False)
    mazzo: list[Carta] = field(init=False)
    seme_briscola: str = field(init=False)
    briscola_esposta: Carta = field(init=False)
    proprietario_briscola_esposta: int | None = field(init=False)
    carte_sul_campo: list[CartaGiocata] = field(init=False)
    carte_giocate: list[CartaGiocata] = field(init=False)
    vincitori_prese: list[int] = field(init=False)
    punteggi: dict[str, int] = field(init=False)
    primo_giocatore_presa: int = field(init=False)
    giocatore_corrente: int = field(init=False)
    indice_presa: int = field(init=False)
    finita: bool = field(init=False)

    def __post_init__(self) -> None:
        self.reset(seed=self.seed, primo_giocatore_id=self.primo_giocatore_id)

    def reset(
        self,
        seed: int | None = None,
        primo_giocatore_id: int | None = None,
    ) -> Osservazione:
        if seed is not None:
            self.seed = seed
        if primo_giocatore_id is not None:
            valida_giocatore_id(primo_giocatore_id)
            self.primo_giocatore_id = primo_giocatore_id

        self.rng = random.Random(self.seed)
        self.mani = [[] for _ in range(NUMERO_GIOCATORI)]
        self.mazzo = crea_mazzo()
        self.rng.shuffle(self.mazzo)

        for _ in range(3):
            for giocatore_id in ordine_giocatori_da(self.primo_giocatore_id):
                self.mani[giocatore_id].append(self.mazzo.pop(0))

        self.briscola_esposta = self.mazzo.pop(0)
        self.seme_briscola = self.briscola_esposta.seme
        self.mazzo.append(self.briscola_esposta)
        self.proprietario_briscola_esposta = None

        self.carte_sul_campo = []
        self.carte_giocate = []
        self.vincitori_prese = []
        self.punteggi = {SQUADRA_PARI: 0, SQUADRA_DISPARI: 0}
        self.primo_giocatore_presa = self.primo_giocatore_id
        self.giocatore_corrente = self.primo_giocatore_id
        self.indice_presa = 0
        self.finita = False
        return self.osserva(self.giocatore_corrente)

    def osserva(self, giocatore_id: int) -> Osservazione:
        valida_giocatore_id(giocatore_id)

        squadra = squadra_di(giocatore_id)
        squadra_avversaria = squadra_avversaria_di(squadra)
        compagno_id = compagno_di(giocatore_id)
        mano_compagno_visibile = self.mano_compagno_visibile()

        return Osservazione(
            giocatore_id=giocatore_id,
            compagno_id=compagno_id,
            avversario_sinistro_id=avversario_sinistro_di(giocatore_id),
            avversario_destro_id=avversario_destro_di(giocatore_id),
            mano=tuple(self.mani[giocatore_id]),
            mano_compagno_visibile=mano_compagno_visibile,
            mano_compagno=(
                tuple(self.mani[compagno_id]) if mano_compagno_visibile else ()
            ),
            seme_briscola=self.seme_briscola,
            briscola_esposta=self.briscola_esposta,
            proprietario_briscola_esposta=self.proprietario_briscola_esposta,
            carte_sul_campo=tuple(self.carte_sul_campo),
            carte_giocate=tuple(self.carte_giocate),
            vincitori_prese=tuple(self.vincitori_prese),
            squadra=squadra,
            squadra_avversaria=squadra_avversaria,
            punteggio_squadra=self.punteggi[squadra],
            punteggio_avversari=self.punteggi[squadra_avversaria],
            primo_giocatore_presa=self.primo_giocatore_presa,
            giocatore_corrente=self.giocatore_corrente,
            carte_nel_mazzo=len(self.mazzo),
            indice_presa=self.indice_presa,
            posizione_nella_presa=len(self.carte_sul_campo),
        )

    def mano_compagno_visibile(self) -> bool:
        return len(self.mazzo) == 0

    def azioni_legali(self, giocatore_id: int | None = None) -> tuple[Carta, ...]:
        if giocatore_id is None:
            giocatore_id = self.giocatore_corrente
        valida_giocatore_id(giocatore_id)
        return tuple(self.mani[giocatore_id])

    def gioca(self, carta: Carta) -> EsitoMossa:
        if self.finita:
            raise RuntimeError("La partita e' gia' finita")

        giocatore_id = self.giocatore_corrente
        if carta not in self.mani[giocatore_id]:
            raise ValueError(
                f"Mossa illegale: il giocatore {giocatore_id} non ha {carta}"
            )

        self.mani[giocatore_id].remove(carta)
        giocata = CartaGiocata(giocatore_id=giocatore_id, carta=carta)
        self.carte_sul_campo.append(giocata)
        self.carte_giocate.append(giocata)

        eventi: list[EventoPubblico] = [
            EventoPubblico(
                tipo="carta_giocata",
                giocatore_id=giocatore_id,
                carta=carta,
            )
        ]
        presa_completata = len(self.carte_sul_campo) == NUMERO_GIOCATORI
        carte_presa_completata: tuple[CartaGiocata, ...] = ()
        vincitore: int | None = None
        punti = 0
        carte_pescate: tuple[CartaPescata, ...] = ()

        if presa_completata:
            carte_presa_completata = tuple(self.carte_sul_campo)
            giocata_vincente = vincitore_presa(
                carte_presa_completata,
                seme_briscola=self.seme_briscola,
            )
            vincitore = giocata_vincente.giocatore_id
            punti = punti_presa(carte_presa_completata)
            self.punteggi[squadra_di(vincitore)] += punti
            self.vincitori_prese.append(vincitore)
            eventi.append(
                EventoPubblico(
                    tipo="presa_completata",
                    giocatore_id=vincitore,
                    punti=punti,
                )
            )

            carte_pescate, eventi_pescata = self._pesca_dopo_presa(vincitore)
            eventi.extend(eventi_pescata)

            self.carte_sul_campo = []
            self.indice_presa += 1
            self.primo_giocatore_presa = vincitore
            self.giocatore_corrente = vincitore
        else:
            self.giocatore_corrente = giocatore_successivo(giocatore_id)

        self.finita = self._partita_finita()
        prossimo_giocatore = None if self.finita else self.giocatore_corrente
        osservazione = None if self.finita else self.osserva(self.giocatore_corrente)

        return EsitoMossa(
            osservazione=osservazione,
            partita_finita=self.finita,
            presa_completata=presa_completata,
            carte_presa_completata=carte_presa_completata,
            vincitore_presa=vincitore,
            punti_presa=punti,
            carte_pescate=carte_pescate,
            prossimo_giocatore=prossimo_giocatore,
            punteggi=dict(self.punteggi),
            eventi_pubblici=tuple(eventi),
        )

    def _pesca_dopo_presa(
        self,
        vincitore: int,
    ) -> tuple[tuple[CartaPescata, ...], tuple[EventoPubblico, ...]]:
        if not self.mazzo:
            return (), ()

        carte_pescate: list[CartaPescata] = []
        eventi: list[EventoPubblico] = []

        for giocatore_id in ordine_giocatori_da(vincitore):
            if not self.mazzo:
                break

            carta = self.mazzo.pop(0)
            self.mani[giocatore_id].append(carta)
            carta_visibile = None
            tipo_evento = "carta_pescata"

            if carta == self.briscola_esposta:
                self.proprietario_briscola_esposta = giocatore_id
                carta_visibile = carta
                tipo_evento = "briscola_esposta_pescata"

            carte_pescate.append(
                CartaPescata(
                    giocatore_id=giocatore_id,
                    carta_visibile=carta_visibile,
                )
            )
            eventi.append(
                EventoPubblico(
                    tipo=tipo_evento,
                    giocatore_id=giocatore_id,
                    carta=carta_visibile,
                )
            )

        return tuple(carte_pescate), tuple(eventi)

    def _partita_finita(self) -> bool:
        return (
            not self.mazzo
            and not self.carte_sul_campo
            and all(len(mano) == 0 for mano in self.mani)
            and self.indice_presa == 10
        )

    def squadra_vincitrice(self) -> str | None:
        if self.punteggi[SQUADRA_PARI] == self.punteggi[SQUADRA_DISPARI]:
            return None
        if self.punteggi[SQUADRA_PARI] > self.punteggi[SQUADRA_DISPARI]:
            return SQUADRA_PARI
        return SQUADRA_DISPARI

    def carte_giocate_per_giocatore(self) -> list[int]:
        conteggi = [0 for _ in range(NUMERO_GIOCATORI)]
        for giocata in self.carte_giocate:
            conteggi[giocata.giocatore_id] += 1
        return conteggi

    def verifica_integrita_stato(self) -> None:
        tutte_le_carte: list[Carta] = []
        for mano in self.mani:
            tutte_le_carte.extend(mano)
        tutte_le_carte.extend(self.mazzo)
        tutte_le_carte.extend(giocata.carta for giocata in self.carte_giocate)

        if len(tutte_le_carte) != 40:
            raise AssertionError(
                f"Attese 40 carte totali, trovate {len(tutte_le_carte)}"
            )
        if len(set(tutte_le_carte)) != 40:
            raise AssertionError("Trovata carta duplicata")
        if self.finita and sum(self.punteggi.values()) != 120:
            raise AssertionError(
                f"Il punteggio finale deve sommare 120, ottenuto {self.punteggi}"
            )
        if self.finita and self.carte_giocate_per_giocatore() != [10, 10, 10, 10]:
            raise AssertionError("Ogni giocatore deve giocare esattamente 10 carte")
