"""Szenario: Zufallszahlen (Gamma, Zuteilung, Verweilzeit) sind bestimmt, reproduzierbar und haben die gewünschten Momente; jeder Zweig der Welten wirkt."""

import numpy as np
import pytest

import hzv_core as K
import hzv_scenario as S
from hzv_scenario import FrailtyWorld, RandomWorld


def test_gamma_zufallszahlen_haben_mittel_eins_und_varianz_v():
    for v in (0.25, 1.0, 2.0, 3.0):
        z = S.frailty_values(K.SplitMix64(4), 200_000, v)
        assert z.min() > 0 and abs(z.mean() - 1.0) < 0.02 and abs(z.var() / v - 1.0) < 0.04, v


def test_gamma_zufallszahlen_sind_reproduzierbar_und_verbrauchen_den_strom_bestimmt():
    a, b = K.SplitMix64(8), K.SplitMix64(8)
    assert np.array_equal(S.gamma_variates(a, 1000, 0.5), S.gamma_variates(b, 1000, 0.5)) and a.state == b.state
    assert not np.array_equal(S.gamma_variates(K.SplitMix64(8), 1000, 0.5), S.gamma_variates(K.SplitMix64(9), 1000, 0.5))


def test_ohne_streuung_ist_die_anfaelligkeit_konstant_eins_und_verbraucht_keinen_zufall():
    rng = K.SplitMix64(3)
    z = S.frailty_values(rng, 10, 0.0)
    assert np.array_equal(z, np.ones(10)) and rng.state == 3
    with pytest.raises(ValueError):
        S.frailty_values(rng, 10, -0.1)
    with pytest.raises(ValueError):
        S.gamma_variates(rng, 10, 0.0)


def test_welt_von_experiment_2_ist_reproduzierbar_und_zuteilung_ist_etwa_haelftig():
    w = FrailtyWorld(n=2000)
    a, b = S.simulate_frailty(w, 0), S.simulate_frailty(w, 0)
    assert np.array_equal(a.time, b.time) and np.array_equal(a.x, b.x) and np.array_equal(a.z, b.z)
    assert not np.array_equal(a.time, S.simulate_frailty(w, 1).time)
    assert 0.45 < a.x.mean() < 0.55 and np.all(a.time > 0) and set(np.unique(a.x)) == {0.0, 1.0}


def test_massnahme_verkuerzt_die_verweilzeit_nach_dem_faktor():
    """Bei v = 0 ist die Verweilzeit exponentiell mit Rate 0,1 · f^x: Mittel 10 bzw. 5 Tage (Band: gut 6 Standardfehler bei n = 40 000)."""
    sm = S.simulate_frailty(FrailtyWorld(n=40_000, v=0.0), 0)
    assert abs(sm.time[sm.x == 0].mean() - 10.0) < 0.3 and abs(sm.time[sm.x == 1].mean() - 5.0) < 0.15


def test_welt_von_experiment_3_zweites_merkmal_wirkt_und_ist_von_der_zuteilung_unabhaengig():
    sm = S.simulate_random(RandomWorld(n=40_000, gamma=1.0), 0)
    assert abs(sm.z.mean()) < 0.03 and abs(sm.z.std() - 1.0) < 0.03
    assert abs(np.corrcoef(sm.x, sm.z)[0, 1]) < 0.03                                   # Zuteilung unabhängig von z
    assert np.corrcoef(sm.z, np.log(sm.time))[0, 1] < -0.3                              # hohes z: kurze Verweilzeit
    flat = S.simulate_random(RandomWorld(n=40_000, gamma=0.0), 0)
    assert abs(np.corrcoef(flat.z, np.log(flat.time))[0, 1]) < 0.03                     # γ = 0: z ohne Wirkung


def test_lauf_seeds_unterscheiden_sich_je_lauf_und_je_zufalls_seed():
    assert len({S.run_seed(12, 0), S.run_seed(12, 1), S.run_seed(13, 1)}) == 3 and S.run_seed(12, 5) == S.run_seed(12, 5)
