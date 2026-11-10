"""Tests der kopierten Kernfunktionen (aus ueberlebenszeit-planung/kern/test_uz_core.py übernommen, soweit die Demo die Funktionen braucht)."""

from fractions import Fraction as F

import numpy as np
import pytest

import hzv_core as K

SIX_T = [2, 3, 4, 5, 5, 8]
SIX_E = [1, 0, 1, 1, 0, 1]


def test_km_handbeispiel_sechs_container():
    km = K.kaplan_meier(SIX_T, SIX_E)
    assert km["t"].tolist() == [2, 4, 5, 8]
    assert km["n"].tolist() == [6, 4, 3, 1]
    for got, want in zip(km["s"], [F(5, 6), F(5, 8), F(5, 12), F(0)]):
        assert got == pytest.approx(float(want), abs=1e-15)


def test_km_ohne_zensierung_ist_eins_minus_verteilungsfunktion():
    rng = K.SplitMix64(1)
    t = np.round(rng.exponential(50) * 3 + 0.5, 1)  # mit Gleichständen
    km = K.kaplan_meier(t, np.ones(50, bool))
    for x in (0.5, 1.0, 2.0, 4.0, 8.0):
        assert K.survival_at(km, x) == pytest.approx(float(np.mean(t > x)), abs=1e-12)


def test_km_alle_zensiert():
    km = K.kaplan_meier([1, 2, 3], [0, 0, 0])
    assert km["t"].size == 0 and K.survival_at(km, 5.0) == 1.0


def test_eingabepruefung():
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([3, 3, 5], [1, 1, 1], [3, 3, 0])  # Eintritt = Zeit
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([0, 1], [1, 1])  # Zeit 0
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1, 2])  # Ereignis 2
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1])  # Länge
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([], [])


def test_cox_handbeispiel_vier_container():
    x = [1, 0, 1, 0]
    r = K.cox_fit([1, 2, 3, 4], [1, 1, 1, 1], x)
    assert r["ok"] and r["beta"][0] == pytest.approx(0.9406, abs=5e-4)

    def score(beta):  # unabhängige Schleife über Risikomengen
        import math

        s = 0.0
        for i in range(4):
            w = [math.exp(beta * v) for v in x[i:]]
            m = sum(wi * v for wi, v in zip(w, x[i:])) / sum(w)
            s += x[i] - m
        return s

    assert abs(score(r["beta"][0])) < 1e-8
    assert score(0.0) == pytest.approx(2 / 3, abs=1e-12)


def test_cox_eintritt_gleich_verschobener_stichprobe():
    rng = K.SplitMix64(3)
    n = 300
    x = rng.normal(n)
    t = 8 * (rng.exponential(n) / np.exp(0.7 * x)) ** (1 / 1.5)
    c = rng.uniform(n) * 12
    obs, ev = np.minimum(t, c), t <= c
    r = K.cox_fit(obs, ev, x)
    assert r["ok"] and 0.4 < r["beta"][0] < 1.0 and r["se"][0] > 0
    m = obs > 3.0
    a = K.cox_fit(obs[m], ev[m], x[m], entry=np.full(m.sum(), 3.0))
    b = K.cox_fit(obs[m] - 3.0, ev[m], x[m])
    assert a["beta"][0] == pytest.approx(b["beta"][0], abs=1e-9)


def test_cox_trennung_singularitaet_und_ohne_ereignis_melden_status():
    sep = K.cox_fit([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0])  # alle x = 1 werden zuerst abgeholt
    assert sep["ok"] is False and "Trennung" in sep["message"]
    const = K.cox_fit([1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1])
    assert const["ok"] is False and "singulär" in const["message"]
    none = K.cox_fit([1, 2], [0, 0], [0, 1])
    assert none["ok"] is False


def test_cox_vorzeichen_und_skalierung():
    x = np.array([1, 0, 1, 0, 1, 0, 0, 1], float)
    t = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
    e = np.ones(8, bool)
    a = K.cox_fit(t, e, x)["beta"][0]
    b = K.cox_fit(t, e, 1 - x)["beta"][0]
    c = K.cox_fit(t, e, 2 * x)["beta"][0]
    assert a == pytest.approx(-b, abs=1e-8) and a == pytest.approx(2 * c, abs=1e-8)


def test_splitmix_referenzwert_und_determinismus():
    # erster Ausgabewert von SplitMix64 bei Seed 0 (Referenz: Vigna, splitmix64.c)
    assert int(K.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    a, b = K.SplitMix64(5), K.SplitMix64(5)
    assert np.array_equal(a.uniform(10), b.uniform(10))
    u = K.SplitMix64(9).uniform(50000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    x = K.SplitMix64(9).exponential(50000)
    assert abs(x.mean() - 1.0) < 0.03
    z = K.SplitMix64(9).normal(50000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_kopie_stimmt_mit_der_quelle_ueberein_falls_vorhanden():
    """Die Funktionen dieser Datei sind wörtlich aus ueberlebenszeit-planung/kern/uz_core.py kopiert; gibt es die Quelle, stimmen die Texte der Funktionen überein."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "ueberlebenszeit-planung" / "kern" / "uz_core.py"
    if not src.exists():
        pytest.skip("ueberlebenszeit-planung/kern/uz_core.py nicht vorhanden")
    mine = Path(K.__file__).read_text(encoding="utf-8")
    theirs = src.read_text(encoding="utf-8")

    def definitions(text):
        return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    a, b = definitions(mine), definitions(theirs)
    assert set(a) <= set(b)
    assert {name for name in a if a[name] != b[name]} == set()


# --- Ergänzungen dieser Demo: Greenwood-Band, mittlere Lebenszeit bis τ, Trennung bei einem einzigen Ereignis -------------------------------------------


def test_greenwood_varianz_und_band_handbeispiel():
    km = K.kaplan_meier(SIX_T, SIX_E)
    i = 2   # Ereigniszeit 5
    assert km["s"][i] ** 2 * km["gw"][i] == pytest.approx(85 / 1728, abs=1e-12)     # Greenwood-Varianz bei t = 5 von Hand
    for method in ("loglog", "plain"):
        s, lo, hi = K.greenwood_band(km, 4.5, 0.95, method)
        assert s == pytest.approx(5 / 8) and 0 <= lo < s < hi <= 1
    s, lo, hi = K.greenwood_band(km, 9.0)
    assert s == 0.0 and np.isnan(lo)
    assert K._z_two_sided(0.95) == pytest.approx(1.959964, abs=1e-6)


def test_mittlere_lebenszeit_handbeispiel_und_rmst_test():
    km = K.kaplan_meier(SIX_T, SIX_E)
    assert K.restricted_mean(km, 8.0) == pytest.approx(133 / 24, abs=1e-12)
    r = K.rmst_difference_test([2, 5], [1, 1], [0, 1], 4.0)
    assert r["status"] == "varianz_null" and np.isnan(r["z"]) and r["diff"] == pytest.approx(2.0, abs=1e-12) and r["se"] == 0.0
    t0, e0, t1, e1 = [2, 4, 6], [1, 1, 1], [3, 5, 7], [1, 1, 1]
    r = K.rmst_difference_test(t0 + t1, e0 + e1, [0] * 3 + [1] * 3, 6.0)
    assert r["diff"] == pytest.approx(K.restricted_mean(K.kaplan_meier(t1, e1), 6.0) - K.restricted_mean(K.kaplan_meier(t0, e0), 6.0), abs=1e-12)


def test_trennung_bei_einem_einzigen_ereignis_meldet_status():
    one = K.cox_fit([1, 2, 3], [1, 0, 0], [1, 0, 0])
    assert one["ok"] is False and "Trennung" in one["message"]
    assert K.cox_fit([1, 2, 3, 4], [1, 1, 1, 1], [1, 0, 1, 0])["ok"] is True   # kein Fehlalarm im Normalfall
