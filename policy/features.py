"""Feature extraction for Briscola policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from game.cards import Carta, CartaGiocata, crea_mazzo
from game.observation import Osservazione
from game.rules import NUMERO_GIOCATORI, punti_presa, vincitore_presa


MAX_PUNTI_CARTA = 11
MAX_PUNTI_PRESA = 44
MAX_CARTE_NEL_MAZZO = 28
PUNTI_TOTALI = 120
TOTALE_BRISCOLE = 10
MAX_PUNTI_MANO = 33
TOTALE_CARTE = 40
TOTALE_CARICHI = 8
TOTALE_FIGURE = 12
MAX_FORZA_CARTA = 10
MAX_CARTE_SUPERIORI = 9
MAX_CARTE_IN_MANO = 3

FIGURE = {"re", "cavallo", "fante"}
CARICHI = {"asso", "tre"}

DEFAULT_ATOMIC_FEATURE_NAMES: tuple[str, ...] = (
    "punti_carta",
    "forza_carta",
    "carta_briscola",
    "carta_asso",
    "carta_tre",
    "carta_figura",
    "carta_carico",
    "carta_liscia",
    "carta_rischiosa",
    "posizione_primo",
    "posizione_secondo",
    "posizione_terzo",
    "posizione_quarto",
    "carte_nella_presa",
    "punti_presa",
    "compagno_sta_prendendo",
    "avversario_sta_prendendo",
    "carta_prende",
    "carta_supera_compagno",
    "carta_supera_avversario",
    "giocatori_dopo",
    "avversari_dopo",
    "compagno_deve_giocare",
    "avversario_deve_giocare",
    "mano_compagno_visibile",
    "punti_mano_compagno",
    "briscole_mano_compagno",
    "carichi_mano_compagno",
    "compagno_ha_briscola",
    "compagno_ha_carico",
    "compagno_puo_prendere",
    "carte_giocate",
    "briscole_giocate",
    "briscole_non_osservate",
    "briscola_esposta_pescata",
    "briscola_esposta_non_giocata",
    "briscola_esposta_mia",
    "briscola_esposta_compagno",
    "briscola_esposta_avversario",
    "assi_giocati",
    "tre_giocati",
    "carichi_giocati",
    "figure_giocate",
    "superiori_stesso_seme_non_osservate",
    "briscole_che_battono_non_osservate",
    "punteggio_squadra",
    "punteggio_avversari",
    "differenza_punteggio",
    "squadra_avanti",
    "squadra_indietro",
    "carte_nel_mazzo",
    "fase_iniziale",
    "fase_media",
    "fase_finale",
    "mazzo_vuoto",
    "ultime_prese",
)

DEFAULT_INTERACTION_FEATURE_NAMES: tuple[str, ...] = (
    "briscola_x_punti_presa",
    "briscola_x_avversario_sta_prendendo",
    "briscola_x_compagno_sta_prendendo",
    "compagno_sta_prendendo_x_punti_carta",
    "carta_prende_x_punti_presa",
    "vantaggio_x_fase_finale",
    "svantaggio_x_fase_finale",
    "avversari_dopo_x_carta_rischiosa",
    "compagno_puo_prendere_x_punti_carta",
    "compagno_ha_briscola_x_avversario_sta_prendendo",
    "mazzo_vuoto_x_carico",
    "mazzo_vuoto_x_briscola",
)

DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    DEFAULT_ATOMIC_FEATURE_NAMES + DEFAULT_INTERACTION_FEATURE_NAMES
)


@dataclass
class BriscolaFeatureExtractor:
    """Build ``phi(osservazione, carta)`` for a learnable policy."""

    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = self._default_feature_names()

    def size(self) -> int:
        """Return the feature vector size."""

        return len(self.feature_names)

    @property
    def atomic_feature_names(self) -> tuple[str, ...]:
        """Return the active non-interaction feature names."""

        atomic_names = set(DEFAULT_ATOMIC_FEATURE_NAMES)
        return tuple(name for name in self.feature_names if name in atomic_names)

    @property
    def interaction_feature_names(self) -> tuple[str, ...]:
        """Return the active engineered interaction feature names."""

        interaction_names = set(DEFAULT_INTERACTION_FEATURE_NAMES)
        return tuple(name for name in self.feature_names if name in interaction_names)

    def extract(self, osservazione: Osservazione, carta: Carta) -> list[float]:
        """Extract legal-observation features for one candidate card."""

        if carta not in osservazione.azioni_legali:
            raise ValueError("Features can only be extracted for legal hand cards")

        vincitore_corrente = self._vincitore_corrente(osservazione)
        vincitore_candidato = self._vincitore_dopo_carta(
            osservazione=osservazione,
            carta=carta,
            giocatore_id=osservazione.giocatore_id,
        )
        carte_osservate = self._carte_osservate(osservazione)

        compagno_sta_prendendo = vincitore_corrente == osservazione.compagno_id
        avversario_sta_prendendo = vincitore_corrente in osservazione.avversari
        carta_prende = vincitore_candidato == osservazione.giocatore_id
        carta_supera_compagno = compagno_sta_prendendo and carta_prende
        carta_supera_avversario = avversario_sta_prendendo and carta_prende

        giocatori_dopo = max(0, 3 - osservazione.posizione_nella_presa)
        ordine_dopo = [
            (osservazione.giocatore_id + offset) % NUMERO_GIOCATORI
            for offset in range(1, giocatori_dopo + 1)
        ]
        avversari_dopo = sum(
            1 for giocatore in ordine_dopo if giocatore in osservazione.avversari
        )
        compagno_deve_giocare = osservazione.compagno_id in ordine_dopo
        avversario_deve_giocare = avversari_dopo > 0

        carta_briscola = self._briscola(osservazione, carta)
        carta_carico = self._carico(carta)
        carta_liscia = carta.punti == 0 and not carta_briscola
        carta_rischiosa = carta_carico or carta_briscola

        punti_carta = carta.punti / MAX_PUNTI_CARTA
        forza_carta = carta.forza / MAX_FORZA_CARTA
        punti_presa_corrente = punti_presa(osservazione.carte_sul_campo)
        punti_presa_norm = punti_presa_corrente / MAX_PUNTI_PRESA

        mano_compagno = (
            osservazione.mano_compagno if osservazione.mano_compagno_visibile else ()
        )
        punti_mano_compagno = sum(carta.punti for carta in mano_compagno)
        briscole_mano_compagno = sum(
            1
            for carta_compagno in mano_compagno
            if self._briscola(osservazione, carta_compagno)
        )
        carichi_mano_compagno = sum(
            1 for carta_compagno in mano_compagno if self._carico(carta_compagno)
        )
        compagno_ha_briscola = briscole_mano_compagno > 0
        compagno_ha_carico = carichi_mano_compagno > 0
        compagno_puo_prendere = self._compagno_puo_prendere(osservazione)

        briscole_giocate = self._conta_briscole_giocate(osservazione)
        briscole_osservate = self._conta_briscole_osservate(
            osservazione=osservazione,
            carte_osservate=carte_osservate,
        )
        assi_giocati = self._conta_rango_giocato(osservazione, "asso")
        tre_giocati = self._conta_rango_giocato(osservazione, "tre")
        carichi_giocati = sum(
            1 for giocata in osservazione.carte_giocate if self._carico(giocata.carta)
        )
        figure_giocate = sum(
            1 for giocata in osservazione.carte_giocate if giocata.carta.rango in FIGURE
        )
        superiori_stesso_seme = self._superiori_stesso_seme_non_osservate(
            carta=carta,
            carte_osservate=carte_osservate,
        )
        briscole_che_battono = self._briscole_che_battono_non_osservate(
            osservazione=osservazione,
            carta=carta,
            carte_osservate=carte_osservate,
        )

        briscola_esposta_pescata = osservazione.proprietario_briscola_esposta is not None
        briscola_esposta_giocata = self._briscola_esposta_giocata(osservazione)
        briscola_esposta_non_giocata = (
            briscola_esposta_pescata and not briscola_esposta_giocata
        )
        briscola_esposta_mia = (
            briscola_esposta_non_giocata
            and osservazione.proprietario_briscola_esposta == osservazione.giocatore_id
        )
        briscola_esposta_compagno = (
            briscola_esposta_non_giocata
            and osservazione.proprietario_briscola_esposta == osservazione.compagno_id
        )
        briscola_esposta_avversario = (
            briscola_esposta_non_giocata
            and osservazione.proprietario_briscola_esposta in osservazione.avversari
        )

        differenza_punteggio = (
            osservazione.punteggio_squadra - osservazione.punteggio_avversari
        ) / PUNTI_TOTALI
        carte_nel_mazzo = osservazione.carte_nel_mazzo / MAX_CARTE_NEL_MAZZO
        fase_iniziale = osservazione.indice_presa <= 2
        fase_media = 3 <= osservazione.indice_presa <= 6
        fase_finale = osservazione.indice_presa >= 7
        mazzo_vuoto = osservazione.carte_nel_mazzo == 0
        ultime_prese = osservazione.indice_presa >= 8

        values = {
            "punti_carta": punti_carta,
            "forza_carta": forza_carta,
            "carta_briscola": float(carta_briscola),
            "carta_asso": float(carta.rango == "asso"),
            "carta_tre": float(carta.rango == "tre"),
            "carta_figura": float(carta.rango in FIGURE),
            "carta_carico": float(carta_carico),
            "carta_liscia": float(carta_liscia),
            "carta_rischiosa": float(carta_rischiosa),
            "posizione_primo": float(osservazione.posizione_nella_presa == 0),
            "posizione_secondo": float(osservazione.posizione_nella_presa == 1),
            "posizione_terzo": float(osservazione.posizione_nella_presa == 2),
            "posizione_quarto": float(osservazione.posizione_nella_presa == 3),
            "carte_nella_presa": osservazione.posizione_nella_presa / 3,
            "punti_presa": punti_presa_norm,
            "compagno_sta_prendendo": float(compagno_sta_prendendo),
            "avversario_sta_prendendo": float(avversario_sta_prendendo),
            "carta_prende": float(carta_prende),
            "carta_supera_compagno": float(carta_supera_compagno),
            "carta_supera_avversario": float(carta_supera_avversario),
            "giocatori_dopo": giocatori_dopo / 3,
            "avversari_dopo": avversari_dopo / 2,
            "compagno_deve_giocare": float(compagno_deve_giocare),
            "avversario_deve_giocare": float(avversario_deve_giocare),
            "mano_compagno_visibile": float(osservazione.mano_compagno_visibile),
            "punti_mano_compagno": punti_mano_compagno / MAX_PUNTI_MANO,
            "briscole_mano_compagno": briscole_mano_compagno / MAX_CARTE_IN_MANO,
            "carichi_mano_compagno": carichi_mano_compagno / MAX_CARTE_IN_MANO,
            "compagno_ha_briscola": float(compagno_ha_briscola),
            "compagno_ha_carico": float(compagno_ha_carico),
            "compagno_puo_prendere": float(compagno_puo_prendere),
            "carte_giocate": osservazione.numero_carte_giocate / TOTALE_CARTE,
            "briscole_giocate": briscole_giocate / TOTALE_BRISCOLE,
            "briscole_non_osservate": (
                TOTALE_BRISCOLE - briscole_osservate
            ) / TOTALE_BRISCOLE,
            "briscola_esposta_pescata": float(briscola_esposta_pescata),
            "briscola_esposta_non_giocata": float(briscola_esposta_non_giocata),
            "briscola_esposta_mia": float(briscola_esposta_mia),
            "briscola_esposta_compagno": float(briscola_esposta_compagno),
            "briscola_esposta_avversario": float(briscola_esposta_avversario),
            "assi_giocati": assi_giocati / 4,
            "tre_giocati": tre_giocati / 4,
            "carichi_giocati": carichi_giocati / TOTALE_CARICHI,
            "figure_giocate": figure_giocate / TOTALE_FIGURE,
            "superiori_stesso_seme_non_osservate": (
                superiori_stesso_seme / MAX_CARTE_SUPERIORI
            ),
            "briscole_che_battono_non_osservate": (
                briscole_che_battono / TOTALE_BRISCOLE
            ),
            "punteggio_squadra": osservazione.punteggio_squadra / PUNTI_TOTALI,
            "punteggio_avversari": osservazione.punteggio_avversari / PUNTI_TOTALI,
            "differenza_punteggio": differenza_punteggio,
            "squadra_avanti": float(differenza_punteggio > 0),
            "squadra_indietro": float(differenza_punteggio < 0),
            "carte_nel_mazzo": carte_nel_mazzo,
            "fase_iniziale": float(fase_iniziale),
            "fase_media": float(fase_media),
            "fase_finale": float(fase_finale),
            "mazzo_vuoto": float(mazzo_vuoto),
            "ultime_prese": float(ultime_prese),
            "briscola_x_punti_presa": float(carta_briscola) * punti_presa_norm,
            "briscola_x_avversario_sta_prendendo": (
                float(carta_briscola) * float(avversario_sta_prendendo)
            ),
            "briscola_x_compagno_sta_prendendo": (
                float(carta_briscola) * float(compagno_sta_prendendo)
            ),
            "compagno_sta_prendendo_x_punti_carta": (
                float(compagno_sta_prendendo) * punti_carta
            ),
            "carta_prende_x_punti_presa": float(carta_prende) * punti_presa_norm,
            "vantaggio_x_fase_finale": max(differenza_punteggio, 0.0)
            * float(fase_finale),
            "svantaggio_x_fase_finale": max(-differenza_punteggio, 0.0)
            * float(fase_finale),
            "avversari_dopo_x_carta_rischiosa": (avversari_dopo / 2)
            * float(carta_rischiosa),
            "compagno_puo_prendere_x_punti_carta": (
                float(compagno_puo_prendere) * punti_carta
            ),
            "compagno_ha_briscola_x_avversario_sta_prendendo": (
                float(compagno_ha_briscola) * float(avversario_sta_prendendo)
            ),
            "mazzo_vuoto_x_carico": float(mazzo_vuoto) * float(carta_carico),
            "mazzo_vuoto_x_briscola": float(mazzo_vuoto) * float(carta_briscola),
        }

        return [float(values[name]) for name in self.feature_names]

    def _default_feature_names(self) -> list[str]:
        return list(DEFAULT_FEATURE_NAMES)

    def _vincitore_corrente(self, osservazione: Osservazione) -> int | None:
        if not osservazione.carte_sul_campo:
            return None
        vincitore = vincitore_presa(
            osservazione.carte_sul_campo,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id

    def _vincitore_dopo_carta(
        self,
        osservazione: Osservazione,
        carta: Carta,
        giocatore_id: int,
    ) -> int:
        presa_candidata = tuple(osservazione.carte_sul_campo) + (
            CartaGiocata(giocatore_id=giocatore_id, carta=carta),
        )
        vincitore = vincitore_presa(
            presa_candidata,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id

    def _carte_osservate(self, osservazione: Osservazione) -> set[Carta]:
        osservate = set(osservazione.mano)
        osservate.update(giocata.carta for giocata in osservazione.carte_sul_campo)
        osservate.update(giocata.carta for giocata in osservazione.carte_giocate)
        osservate.add(osservazione.briscola_esposta)
        if osservazione.mano_compagno_visibile:
            osservate.update(osservazione.mano_compagno)
        return osservate

    def _compagno_puo_prendere(self, osservazione: Osservazione) -> bool:
        if not osservazione.mano_compagno_visibile:
            return False
        return any(
            self._vincitore_dopo_carta(
                osservazione=osservazione,
                carta=carta,
                giocatore_id=osservazione.compagno_id,
            )
            == osservazione.compagno_id
            for carta in osservazione.mano_compagno
        )

    def _conta_briscole_giocate(self, osservazione: Osservazione) -> int:
        return sum(
            1
            for giocata in osservazione.carte_giocate
            if self._briscola(osservazione, giocata.carta)
        )

    def _conta_briscole_osservate(
        self,
        osservazione: Osservazione,
        carte_osservate: set[Carta],
    ) -> int:
        return sum(
            1 for carta in carte_osservate if self._briscola(osservazione, carta)
        )

    def _conta_rango_giocato(self, osservazione: Osservazione, rango: str) -> int:
        return sum(
            1
            for giocata in osservazione.carte_giocate
            if giocata.carta.rango == rango
        )

    def _superiori_stesso_seme_non_osservate(
        self,
        carta: Carta,
        carte_osservate: set[Carta],
    ) -> int:
        return sum(
            1
            for altra in crea_mazzo()
            if altra.seme == carta.seme
            and altra.forza > carta.forza
            and altra not in carte_osservate
        )

    def _briscole_che_battono_non_osservate(
        self,
        osservazione: Osservazione,
        carta: Carta,
        carte_osservate: set[Carta],
    ) -> int:
        if not self._briscola(osservazione, carta):
            return sum(
                1
                for altra in crea_mazzo()
                if altra.seme == osservazione.seme_briscola
                and altra not in carte_osservate
            )

        return sum(
            1
            for altra in crea_mazzo()
            if altra.seme == osservazione.seme_briscola
            and altra.forza > carta.forza
            and altra not in carte_osservate
        )

    def _briscola_esposta_giocata(self, osservazione: Osservazione) -> bool:
        return any(
            giocata.carta == osservazione.briscola_esposta
            for giocata in osservazione.carte_giocate
        )

    def _briscola(self, osservazione: Osservazione, carta: Carta) -> bool:
        return carta.seme == osservazione.seme_briscola

    def _carico(self, carta: Carta) -> bool:
        return carta.rango in CARICHI
