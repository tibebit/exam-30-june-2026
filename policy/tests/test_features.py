from __future__ import annotations

import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from policy import BriscolaFeatureExtractor
from policy.features import (
    DEFAULT_ATOMIC_FEATURE_NAMES,
    DEFAULT_FEATURE_NAMES,
    DEFAULT_INTERACTION_FEATURE_NAMES,
)


def osservazione(
    *,
    mano: tuple[Carta, ...] = (
        Carta("coppe", "asso"),
        Carta("denari", "due"),
        Carta("bastoni", "sette"),
    ),
    mano_compagno_visibile: bool = False,
    mano_compagno: tuple[Carta, ...] = (),
    seme_briscola: str = "denari",
    briscola_esposta: Carta = Carta("denari", "asso"),
    proprietario_briscola_esposta: int | None = None,
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    carte_giocate: tuple[CartaGiocata, ...] = (),
    punteggio_squadra: int = 0,
    punteggio_avversari: int = 0,
    carte_nel_mazzo: int = 28,
    indice_presa: int = 0,
) -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
        mano_compagno_visibile=mano_compagno_visibile,
        mano_compagno=mano_compagno,
        seme_briscola=seme_briscola,
        briscola_esposta=briscola_esposta,
        proprietario_briscola_esposta=proprietario_briscola_esposta,
        carte_sul_campo=carte_sul_campo,
        carte_giocate=carte_giocate,
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=punteggio_squadra,
        punteggio_avversari=punteggio_avversari,
        primo_giocatore_presa=1,
        giocatore_corrente=0,
        carte_nel_mazzo=carte_nel_mazzo,
        indice_presa=indice_presa,
        posizione_nella_presa=len(carte_sul_campo),
    )


def feature(extractor: BriscolaFeatureExtractor, values: list[float], name: str) -> float:
    return values[extractor.feature_names.index(name)]


class TestBriscolaFeatureExtractor(unittest.TestCase):
    def test_feature_vector_has_expected_size(self):
        # The extracted vector must match the declared size.
        extractor = BriscolaFeatureExtractor()
        obs = osservazione()

        values = extractor.extract(obs, obs.mano[0])

        self.assertEqual(len(values), extractor.size())

    def test_features_are_floats(self):
        # The linear policy works on a uniform numeric vector.
        extractor = BriscolaFeatureExtractor()
        obs = osservazione()

        values = extractor.extract(obs, obs.mano[0])

        self.assertTrue(all(isinstance(value, float) for value in values))

    def test_feature_groups_are_explicit_and_cover_default_vector(self):
        # Atomiche e interazioni devono coprire tutto il vettore senza duplicati.
        extractor = BriscolaFeatureExtractor()

        self.assertEqual(extractor.feature_names, list(DEFAULT_FEATURE_NAMES))
        self.assertEqual(extractor.atomic_feature_names, DEFAULT_ATOMIC_FEATURE_NAMES)
        self.assertEqual(
            extractor.interaction_feature_names,
            DEFAULT_INTERACTION_FEATURE_NAMES,
        )
        self.assertEqual(len(extractor.atomic_feature_names), 56)
        self.assertEqual(len(extractor.interaction_feature_names), 12)
        self.assertEqual(
            len(extractor.feature_names),
            len(set(extractor.feature_names)),
        )
        self.assertEqual(
            extractor.feature_names,
            list(extractor.atomic_feature_names + extractor.interaction_feature_names),
        )

    def test_carta_non_legale_solleva_value_error(self):
        # Features are defined only for legal actions in the current hand.
        extractor = BriscolaFeatureExtractor()
        obs = osservazione()

        with self.assertRaises(ValueError):
            extractor.extract(obs, Carta("spade", "asso"))

    def test_feature_di_carta(self):
        # Points, strength, and card type must be encoded directly.
        extractor = BriscolaFeatureExtractor()
        carta = Carta("denari", "tre")
        obs = osservazione(mano=(carta,))

        values = extractor.extract(obs, carta)

        self.assertAlmostEqual(feature(extractor, values, "punti_carta"), 10 / 11)
        self.assertAlmostEqual(
            feature(extractor, values, "forza_carta"),
            carta.forza / 10,
        )
        self.assertEqual(feature(extractor, values, "carta_briscola"), 1.0)
        self.assertEqual(feature(extractor, values, "carta_tre"), 1.0)
        self.assertEqual(feature(extractor, values, "carta_carico"), 1.0)
        self.assertEqual(feature(extractor, values, "carta_liscia"), 0.0)

    def test_feature_di_presa_quando_avversario_prende_e_carta_supera(self):
        # If a card wins over an opponent, the candidate trick must be recognized.
        extractor = BriscolaFeatureExtractor()
        carta = Carta("coppe", "asso")
        obs = osservazione(
            mano=(carta,),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "re")),
            ),
        )

        values = extractor.extract(obs, carta)

        self.assertEqual(feature(extractor, values, "avversario_sta_prendendo"), 1.0)
        self.assertEqual(feature(extractor, values, "carta_prende"), 1.0)
        self.assertEqual(feature(extractor, values, "carta_supera_avversario"), 1.0)

    def test_mano_compagno_nascosta_non_alimenta_feature_derivate(self):
        # A hidden teammate hand must not produce derived information.
        extractor = BriscolaFeatureExtractor()
        obs = osservazione(
            mano_compagno_visibile=False,
            mano_compagno=(Carta("denari", "asso"), Carta("coppe", "tre")),
        )

        values = extractor.extract(obs, obs.mano[0])

        self.assertEqual(feature(extractor, values, "mano_compagno_visibile"), 0.0)
        self.assertEqual(feature(extractor, values, "punti_mano_compagno"), 0.0)
        self.assertEqual(feature(extractor, values, "briscole_mano_compagno"), 0.0)
        self.assertEqual(feature(extractor, values, "compagno_ha_briscola"), 0.0)

    def test_mano_compagno_visibile_alimenta_feature_derivate(self):
        # The teammate hand becomes usable only when it is legally visible.
        extractor = BriscolaFeatureExtractor()
        obs = osservazione(
            mano_compagno_visibile=True,
            mano_compagno=(Carta("denari", "asso"), Carta("coppe", "tre")),
        )

        values = extractor.extract(obs, obs.mano[0])

        self.assertEqual(feature(extractor, values, "mano_compagno_visibile"), 1.0)
        self.assertAlmostEqual(
            feature(extractor, values, "punti_mano_compagno"),
            21 / 33,
        )
        self.assertAlmostEqual(
            feature(extractor, values, "briscole_mano_compagno"),
            1 / 3,
        )
        self.assertEqual(feature(extractor, values, "compagno_ha_briscola"), 1.0)
        self.assertEqual(feature(extractor, values, "compagno_ha_carico"), 1.0)

    def test_briscola_esposta_avversario_se_pescata_e_non_giocata(self):
        # A drawn and unplayed exposed briscola remains public information.
        extractor = BriscolaFeatureExtractor()
        briscola_esposta = Carta("denari", "asso")
        obs = osservazione(
            briscola_esposta=briscola_esposta,
            proprietario_briscola_esposta=1,
        )

        values = extractor.extract(obs, obs.mano[0])

        self.assertEqual(feature(extractor, values, "briscola_esposta_pescata"), 1.0)
        self.assertEqual(
            feature(extractor, values, "briscola_esposta_non_giocata"),
            1.0,
        )
        self.assertEqual(
            feature(extractor, values, "briscola_esposta_avversario"),
            1.0,
        )

    def test_briscola_esposta_giocata_non_viene_attribuita_al_proprietario(self):
        # An already played exposed briscola must no longer be attributed to its owner.
        extractor = BriscolaFeatureExtractor()
        briscola_esposta = Carta("denari", "asso")
        obs = osservazione(
            briscola_esposta=briscola_esposta,
            proprietario_briscola_esposta=1,
            carte_giocate=(CartaGiocata(giocatore_id=1, carta=briscola_esposta),),
        )

        values = extractor.extract(obs, obs.mano[0])

        self.assertEqual(feature(extractor, values, "briscola_esposta_pescata"), 1.0)
        self.assertEqual(
            feature(extractor, values, "briscola_esposta_non_giocata"),
            0.0,
        )
        self.assertEqual(
            feature(extractor, values, "briscola_esposta_avversario"),
            0.0,
        )

    def test_briscole_che_battono_non_osservate_per_carta_non_briscola(self):
        # A non-briscola card can be beaten by any unobserved briscola.
        extractor = BriscolaFeatureExtractor()
        carta = Carta("coppe", "due")
        obs = osservazione(mano=(carta,))

        values = extractor.extract(obs, carta)

        self.assertGreater(
            feature(extractor, values, "briscole_che_battono_non_osservate"),
            0.0,
        )

    def test_briscole_che_battono_non_osservate_per_briscola_conta_solo_superiori(self):
        # A briscola card is threatened only by stronger unobserved briscola cards.
        extractor = BriscolaFeatureExtractor()
        carta = Carta("denari", "re")
        obs = osservazione(mano=(carta,))

        values = extractor.extract(obs, carta)

        self.assertAlmostEqual(
            feature(extractor, values, "briscole_che_battono_non_osservate"),
            1 / 10,
        )

    def test_feature_di_punteggio_e_fase(self):
        # Quantitative context features must use a consistent scale.
        extractor = BriscolaFeatureExtractor()
        obs = osservazione(
            punteggio_squadra=70,
            punteggio_avversari=40,
            carte_nel_mazzo=0,
            indice_presa=8,
        )

        values = extractor.extract(obs, obs.mano[0])

        self.assertAlmostEqual(
            feature(extractor, values, "punteggio_squadra"),
            70 / 120,
        )
        self.assertAlmostEqual(
            feature(extractor, values, "punteggio_avversari"),
            40 / 120,
        )
        self.assertAlmostEqual(
            feature(extractor, values, "differenza_punteggio"),
            30 / 120,
        )
        self.assertEqual(feature(extractor, values, "squadra_avanti"), 1.0)
        self.assertEqual(feature(extractor, values, "mazzo_vuoto"), 1.0)
        self.assertEqual(feature(extractor, values, "fase_finale"), 1.0)
        self.assertEqual(feature(extractor, values, "ultime_prese"), 1.0)

    def test_feature_interazione_e_prodotto_delle_componenti(self):
        # Le interazioni engineered devono restare prodotti numerici espliciti.
        extractor = BriscolaFeatureExtractor()
        carta = Carta("denari", "due")
        obs = osservazione(
            mano=(carta,),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "asso")),
                CartaGiocata(giocatore_id=2, carta=Carta("bastoni", "tre")),
            ),
        )

        values = extractor.extract(obs, carta)

        self.assertAlmostEqual(
            feature(extractor, values, "briscola_x_punti_presa"),
            feature(extractor, values, "carta_briscola")
            * feature(extractor, values, "punti_presa"),
        )


if __name__ == "__main__":
    unittest.main()
