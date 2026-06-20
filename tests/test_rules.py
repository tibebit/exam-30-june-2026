import unittest

from game.cards import Carta, CartaGiocata
from game.rules import (
    compagno_di,
    giocatore_successivo,
    ordine_giocatori_da,
    punti_presa,
    squadra_di,
    vincitore_presa,
)


def giocata(giocatore_id: int, seme: str, rango: str) -> CartaGiocata:
    return CartaGiocata(giocatore_id=giocatore_id, carta=Carta(seme=seme, rango=rango))


class TestRegole(unittest.TestCase):
    def test_briscola_batte_non_briscola(self):
        carte_sul_campo = (
            giocata(0, "coppe", "asso"),
            giocata(1, "denari", "due"),
            giocata(2, "coppe", "tre"),
            giocata(3, "spade", "re"),
        )

        self.assertEqual(
            vincitore_presa(carte_sul_campo, seme_briscola="denari").giocatore_id,
            1,
        )

    def test_briscola_piu_forte_batte_briscola_piu_debole(self):
        carte_sul_campo = (
            giocata(0, "coppe", "fante"),
            giocata(1, "denari", "due"),
            giocata(2, "denari", "asso"),
            giocata(3, "coppe", "tre"),
        )

        self.assertEqual(
            vincitore_presa(carte_sul_campo, seme_briscola="denari").giocatore_id,
            2,
        )

    def test_senza_briscole_vince_il_seme_di_apertura(self):
        carte_sul_campo = (
            giocata(0, "coppe", "re"),
            giocata(1, "bastoni", "asso"),
            giocata(2, "coppe", "fante"),
            giocata(3, "spade", "tre"),
        )

        self.assertEqual(
            vincitore_presa(carte_sul_campo, seme_briscola="denari").giocatore_id,
            0,
        )

    def test_punti_presa_somma_i_punti_delle_carte(self):
        carte_sul_campo = (
            giocata(0, "coppe", "asso"),
            giocata(1, "denari", "tre"),
            giocata(2, "bastoni", "fante"),
            giocata(3, "spade", "due"),
        )

        self.assertEqual(punti_presa(carte_sul_campo), 23)

    def test_compagno_e_squadra(self):
        self.assertEqual(compagno_di(0), 2)
        self.assertEqual(compagno_di(1), 3)
        self.assertEqual(squadra_di(0), squadra_di(2))
        self.assertEqual(squadra_di(1), squadra_di(3))
        self.assertNotEqual(squadra_di(0), squadra_di(1))

    def test_ordine_giocatori(self):
        self.assertEqual(giocatore_successivo(3), 0)
        self.assertEqual(ordine_giocatori_da(3), [3, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
