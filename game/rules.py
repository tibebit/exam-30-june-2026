"""Pure rules for four-player Briscola."""

from __future__ import annotations

from collections.abc import Sequence

from .cards import Carta, CartaGiocata, SEMI


NUMERO_GIOCATORI = 4
SQUADRA_PARI = "pari"
SQUADRA_DISPARI = "dispari"


def valida_giocatore_id(giocatore_id: int) -> None:
    if giocatore_id not in range(NUMERO_GIOCATORI):
        raise ValueError(f"Id giocatore non valido: {giocatore_id}")


def valida_seme(seme: str) -> None:
    if seme not in SEMI:
        raise ValueError(f"Seme sconosciuto: {seme}")


def valida_squadra(squadra: str) -> None:
    if squadra not in (SQUADRA_PARI, SQUADRA_DISPARI):
        raise ValueError(f"Squadra sconosciuta: {squadra}")


def squadra_di(giocatore_id: int) -> str:
    valida_giocatore_id(giocatore_id)
    return SQUADRA_PARI if giocatore_id % 2 == 0 else SQUADRA_DISPARI


def squadra_avversaria_di(squadra: str) -> str:
    valida_squadra(squadra)
    return SQUADRA_DISPARI if squadra == SQUADRA_PARI else SQUADRA_PARI


def compagno_di(giocatore_id: int) -> int:
    valida_giocatore_id(giocatore_id)
    return (giocatore_id + 2) % NUMERO_GIOCATORI


def avversario_sinistro_di(giocatore_id: int) -> int:
    valida_giocatore_id(giocatore_id)
    return (giocatore_id + 1) % NUMERO_GIOCATORI


def avversario_destro_di(giocatore_id: int) -> int:
    valida_giocatore_id(giocatore_id)
    return (giocatore_id - 1) % NUMERO_GIOCATORI


def ordine_giocatori_da(primo_giocatore_id: int) -> list[int]:
    valida_giocatore_id(primo_giocatore_id)
    return [
        (primo_giocatore_id + offset) % NUMERO_GIOCATORI
        for offset in range(NUMERO_GIOCATORI)
    ]


def giocatore_successivo(giocatore_id: int) -> int:
    valida_giocatore_id(giocatore_id)
    return (giocatore_id + 1) % NUMERO_GIOCATORI


def punti_presa(carte_sul_campo: Sequence[CartaGiocata]) -> int:
    return sum(giocata.carta.punti for giocata in carte_sul_campo)


def vincitore_presa(
    carte_sul_campo: Sequence[CartaGiocata],
    seme_briscola: str,
) -> CartaGiocata:
    if not carte_sul_campo:
        raise ValueError("Non si puo risolvere una presa vuota")

    valida_seme(seme_briscola)
    seme_apertura = carte_sul_campo[0].carta.seme
    vincitore = carte_sul_campo[0]

    for giocata in carte_sul_campo[1:]:
        if _carta_batte(
            sfidante=giocata.carta,
            attuale=vincitore.carta,
            seme_apertura=seme_apertura,
            seme_briscola=seme_briscola,
        ):
            vincitore = giocata

    return vincitore


def _carta_batte(
    sfidante: Carta,
    attuale: Carta,
    seme_apertura: str,
    seme_briscola: str,
) -> bool:
    sfidante_briscola = sfidante.seme == seme_briscola
    attuale_briscola = attuale.seme == seme_briscola

    if sfidante_briscola and not attuale_briscola:
        return True
    if attuale_briscola and not sfidante_briscola:
        return False

    if sfidante_briscola and attuale_briscola:
        return sfidante.forza > attuale.forza

    if sfidante.seme == seme_apertura and attuale.seme != seme_apertura:
        return True
    if sfidante.seme != seme_apertura:
        return False

    return sfidante.forza > attuale.forza
