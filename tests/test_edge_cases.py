"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste und größte Reglerwerte, leere Phasen, Status statt stiller Zahlen."""

import math

import numpy as np
import pytest

import hzv_constants as C
import hzv_core as K
import hzv_evaluation as ev
import hzv_model as M
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld, Sample


def test_kleinste_stichprobe_liefert_in_jeder_phase_ein_ergebnis_oder_einen_status():
    study = M.run_frailty_study(FrailtyWorld(n=C.N_MIN, v=C.V_MAX), 12)
    assert study.windows.shape == (12, 3) and study.follow_ups.shape == (12, len(C.FOLLOW_UPS))
    assert sum(n for _, n in study.status_counts) == 12 * 9
    text = ev.failure_text(study.status_counts, study.reps)
    bad = sum(n for k, n in study.status_counts if k != "ok")
    assert (text == "") == (bad == 0)


def test_groesste_streuung_und_groesster_faktor_laufen_ohne_fehler():
    for world in (FrailtyWorld(n=C.N_MIN, v=C.V_MAX, factor=C.FACTOR_MAX), FrailtyWorld(n=C.N_MIN, v=C.V_MIN, factor=C.FACTOR_MIN)):
        study = M.run_frailty_study(world, 5)
        assert study.windows.shape == (5, 3)
        assert M.analyse_stay(world, C.TAU_MAX).n == world.n and M.analyse_stay(world, C.TAU_MIN).n == world.n


def test_zweites_merkmal_an_den_reglerraendern():
    for gamma in (C.GAMMA_MIN, C.GAMMA_MAX):
        for factor in (C.FACTOR_MIN, C.FACTOR_MAX):
            study = M.run_random_study(RandomWorld(n=C.N_MIN, gamma=gamma, factor=factor), 4)
            assert study.only_x.size == 4
            assert math.isfinite(M.population_random(RandomWorld(gamma=gamma, factor=factor)))


def test_phase_ohne_container_gibt_nan_und_der_hinweis_nennt_den_grund():
    """Alle Container verlassen die Gruppen vor Tag 3: mittlere und späte Phase sind leer, kein Fehler, Status statt Zahl."""
    sample = Sample(time=np.array([0.5, 1.0, 1.5, 2.0, 0.7, 0.9]), x=np.array([1, 0, 1, 0, 1, 0], float), z=np.ones(6))
    betas, states = M.phase_betas(sample)
    assert math.isnan(betas[1]) and math.isnan(betas[2]) and states[1:] == ["ohne_ereignis"] * 2
    text = ev.failure_text((("ok", 7), ("ohne_ereignis", 2)), 3)
    assert "2 von 9" in text and "ohne Ereignis: 2" in text


def test_hinweis_ohne_fehlschlaege_ist_leer_und_nennt_sonst_alle_gruende():
    assert ev.failure_text((("ok", 9),), 1) == ""
    text = ev.failure_text((("keine_konvergenz", 3), ("ok", 6), ("singulaer", 1)), 2)
    assert "4 von 10" in text and "keine Konvergenz (häufig perfekte Trennung): 3" in text and "Information 0: 1" in text


def test_ungueltige_eingaben_werfen_eine_verstaendliche_meldung():
    with pytest.raises(K.SurvivalError, match="größer als 0"):
        M.cox_fast([0.0, 1.0], [1, 1], [0, 1])
    with pytest.raises(K.SurvivalError, match="Ereignis"):
        M.cox_fast([1.0, 2.0], [1, 2], [0, 1])


def test_zwei_klassen_mit_gleicher_rate_und_ohne_wirkung_haben_keinen_tiefpunkt():
    assert M.class_hr_minimum(ClassWorld(rate_a=1.0, rate_b=1.0))[1] == pytest.approx(2.0, abs=1e-9)
    assert M.class_hr_minimum(ClassWorld(factor=1.0))[1] == pytest.approx(1.0, abs=1e-12)


def test_stay_an_den_zeitraendern():
    w = FrailtyWorld(n=C.N_MIN, v=0.0)
    early = M.analyse_stay(w, 1.0)
    assert 0.8 < early.s0 <= 1.0 and early.rmst_status == "ok"
    late = M.analyse_stay(w, 40.0)
    assert 0.0 <= late.s1 < late.s0 <= 1.0 and late.rmst0 > late.rmst1 > 0
