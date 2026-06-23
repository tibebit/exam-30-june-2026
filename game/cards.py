"""Cards and deck for four-player Briscola."""

from __future__ import annotations

from dataclasses import dataclass


SEMI: tuple[str, ...] = ("coppe", "denari", "bastoni", "spade")

RANGHI: tuple[str, ...] = (
    "asso",
    "tre",
    "re",
    "cavallo",
    "fante",
    "sette",
    "sei",
    "cinque",
    "quattro",
    "due",
)

PUNTI: dict[str, int] = {
    "asso": 11,
    "tre": 10,
    "re": 4,
    "cavallo": 3,
    "fante": 2,
    "sette": 0,
    "sei": 0,
    "cinque": 0,
    "quattro": 0,
    "due": 0,
}

FORZA_RANGO: dict[str, int] = {
    rango: len(RANGHI) - indice for indice, rango in enumerate(RANGHI)
}


@dataclass(frozen=True, order=True)
class Carta:
    """One card from the Briscola deck."""

    seme: str
    rango: str

    def __post_init__(self) -> None:
        if self.seme not in SEMI:
            raise ValueError(f"Seme sconosciuto: {self.seme}")
        if self.rango not in RANGHI:
            raise ValueError(f"Rango sconosciuto: {self.rango}")

    @property
    def punti(self) -> int:
        return PUNTI[self.rango]

    @property
    def forza(self) -> int:
        return FORZA_RANGO[self.rango]

    @property
    def id(self) -> str:
        return f"{self.rango}_di_{self.seme}"


@dataclass(frozen=True)
class CartaGiocata:
    """One card publicly played by a player."""

    giocatore_id: int
    carta: Carta


def crea_mazzo() -> list[Carta]:
    """Create an ordered 40-card deck."""

    return [Carta(seme=seme, rango=rango) for seme in SEMI for rango in RANGHI]


def carta_da_id(carta_id: str) -> Carta:
    """Rebuild a card from an id produced by ``Carta.id``."""

    rango, separatore, seme = carta_id.partition("_di_")
    if not separatore or not seme:
        raise ValueError(f"Id carta non valido: {carta_id}")
    return Carta(seme=seme, rango=rango)


def punti_totali_mazzo() -> int:
    return sum(carta.punti for carta in crea_mazzo())
