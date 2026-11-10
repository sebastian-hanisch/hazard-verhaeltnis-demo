"""Eigenschaften, die für jede zulässige Einstellung gelten müssen (nicht nur an Handwerten): über ein Raster der Regler gezählt."""

import math

import numpy as np
import pytest

import hzv_constants as C
import hzv_model as M
from hzv_scenario import ClassWorld, FrailtyWorld

TIMES = np.linspace(0.0, 400.0, 801)


def test_streuung_null_gibt_ein_konstantes_verhaeltnis():
    for f in np.arange(C.FACTOR_MIN, C.FACTOR_MAX + 1e-9, C.FACTOR_STEP):
        assert np.allclose(M.gamma_hr(0.0, C.BASE_RATE, f, TIMES), f, atol=1e-12)


def test_verhaeltnis_bei_streuung_faellt_monoton_von_f_gegen_eins():
    """Für jedes v > 0 und jeden Faktor > 1: HR(0) = f, HR fällt streng, bleibt über 1; Zählung über das ganze Raster der Regler."""
    cells = 0
    for v in np.arange(C.V_STEP, C.V_MAX + 1e-9, C.V_STEP):
        for f in np.arange(C.FACTOR_MIN + C.FACTOR_STEP, C.FACTOR_MAX + 1e-9, C.FACTOR_STEP):
            hr = M.gamma_hr(v, C.BASE_RATE, f, TIMES)
            assert hr[0] == pytest.approx(f, abs=1e-12) and np.all(np.diff(hr) < 0) and np.all(hr > 1.0), (v, f)
            assert hr[-1] < 1.0 + 0.5 * (f - 1.0), (v, f)       # nach 400 Tagen höchstens auf halbem Weg zwischen 1 und f
            cells += 1
    assert cells == 12 * 12


def test_ohne_wirkung_ist_das_verhaeltnis_eins_zu_jedem_zeitpunkt():
    for v in (0.0, 0.5, 3.0):
        assert np.allclose(M.gamma_hr(v, C.BASE_RATE, 1.0, TIMES), 1.0, atol=1e-15)


def test_ueberlebenskurven_kreuzen_sich_nicht_fuer_jeden_faktor_ueber_eins():
    for v in (0.0, 0.5, 1.0, 3.0):
        for f in (1.25, 2.0, 4.0):
            law0, law1 = M.frailty_laws(FrailtyWorld(v=v, factor=f))
            assert np.all(law1.surv(TIMES[1:]) < law0.surv(TIMES[1:])), (v, f)


def test_marginale_ueberlebensfunktion_ist_das_mittel_der_klassen():
    for world in (ClassWorld(), ClassWorld(rate_a=0.5, rate_b=7.0, share_b=0.3, factor=3.0)):
        c, t = M.class_laws(world)
        for x in (0.0, 0.7, 3.0):
            expect = (1 - world.share_b) * math.exp(-world.rate_a * x) + world.share_b * math.exp(-world.rate_b * x)
            assert float(c.surv(x)) == pytest.approx(expect, abs=1e-15)
            expect_t = (1 - world.share_b) * math.exp(-world.rate_a * world.factor * x) + world.share_b * math.exp(-world.rate_b * world.factor * x)
            assert float(t.surv(x)) == pytest.approx(expect_t, abs=1e-15)


def test_zweiklassen_verhaeltnis_beginnt_beim_faktor_fuer_jede_aufteilung_und_jede_rate():
    cases = 0
    for a in (0.2, 0.6, 1.0):
        for b in (1.5, 4.0, 10.0):
            for share in (0.1, 0.5, 0.9):
                for f in (1.0, 2.0, 4.0):
                    w = ClassWorld(a, b, share, f)
                    assert M.class_rows(w, [0.0])[0]["hr"] == pytest.approx(f, abs=1e-12)
                    assert M.class_zero_rates_exact(w)["hr"] == pytest.approx(f, abs=1e-9)
                    cases += 1
    assert cases == 81


def test_zweiklassen_verhaeltnis_bleibt_zwischen_eins_und_f_und_kehrt_zurueck():
    for a, b, share, f in ((1.0, 4.0, 0.5, 2.0), (0.4, 8.0, 0.2, 3.0), (0.8, 1.5, 0.7, 1.5)):
        w = ClassWorld(a, b, share, f)
        hr = M.class_hr_curve(w, np.linspace(0.0, 60.0, 3001))
        assert np.all(hr > 1.0 - 1e-12) and np.all(hr <= f + 1e-12)
        assert hr[-1] == pytest.approx(f, abs=1e-6)      # am Ende nur noch die langsamere Klasse


def test_grenzwert_der_phasen_faellt_von_frueh_ueber_mittel_nach_spaet():
    for v in (0.5, 1.0, 2.0, 3.0):
        b = M.population_windows(FrailtyWorld(v=v))
        assert b[0] > b[1] > b[2] > 0.0 and b[0] < math.log(2.0), v


def test_grenzwert_waechst_mit_dem_faktor_und_faellt_mit_der_streuung():
    by_f = [M.population_windows(FrailtyWorld(v=1.0, factor=f))[1] for f in (1.5, 2.0, 3.0)]
    assert by_f == sorted(by_f)
    by_v = [M.population_windows(FrailtyWorld(v=v))[1] for v in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert by_v == sorted(by_v, reverse=True)


def test_grenzwert_der_nachbeobachtung_faellt_mit_ihrer_laenge():
    b = M.population_follow_ups(FrailtyWorld(v=1.0))
    assert all(x > y for x, y in zip(b, b[1:]))
