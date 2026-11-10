"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import hzv_constants as C
import hzv_evaluation as ev
import hzv_model as M
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld


def test_formatierer():
    assert ev.num(-1.234, 2) == "−1,23" and ev.num(0.5, 1) == "0,5" and ev.pct(0.456) == "46 %" and ev.pct(0.0456, 1) == "4,6 %"
    assert ev.frac(F(5, 2)) == "5/2" and ev.frac(F(-1, 3)) == "−1/3" and ev.frac(F(4, 2)) == "2"
    assert ev.hr_text(math.log(2.0)) == "2,00" and ev.hr_text(float("nan")) == "–"


# ------------------------------------------------------------------------------------------------ Experiment 1


def _classes(world, t_min=0.445, hr_min=1.47, hr_end=2.0, share_end=0.0, cross=True):
    return ev.ClassAnalysis(world, {}, [], {}, t_min, hr_min, hr_end, share_end, cross)


def test_verdikt_klassen_nennt_start_tiefpunkt_ende_und_anteil():
    kind, text = ev.verdict_classes(_classes(ClassWorld(), share_end=0.0001))
    assert kind == "success"
    assert "startet bei 2,00" in text and "bis Tag 0,45 auf 1,47" in text and "(Tag 6) bei 2,00" in text and "noch 0,0 % der Übrigen" in text and "(zu Beginn 50 %)" in text
    assert "kreuzen sich nicht" in text and "Verhältnis 2,00" in text


def test_verdikt_klassen_ohne_wirkung_und_ohne_unterschied():
    kind, text = ev.verdict_classes(_classes(ClassWorld(factor=1.0)))
    assert kind == "info" and text.startswith("Faktor 1:") and "zu jedem Zeitpunkt 1" in text
    kind, text = ev.verdict_classes(_classes(ClassWorld(rate_a=1.0, rate_b=1.0, factor=3.0)))
    assert kind == "info" and "dieselbe Rate (1,0 je Tag)" in text and "bei 3,00" in text


def test_verdikt_klassen_meldet_eine_kreuzung_der_ueberlebenskurven_wenn_sie_vorkommt():
    assert "kreuzen sich." in ev.verdict_classes(_classes(ClassWorld(), cross=False))[1]


# ------------------------------------------------------------------------------------------------ Experiment 2


def _study(rows):
    arr = np.array(rows, dtype=float)
    return M.FrailtyStudy(arr, np.full((arr.shape[0], len(C.FOLLOW_UPS)), 0.5), (("ok", 9 * arr.shape[0]),))


def test_verdikt_streuung_erfolgsfall_nennt_phasen_grenzwerte_und_abfall():
    study = _study([[0.6, 0.4, 0.2]] * 10)
    kind, text = ev.verdict_frailty(FrailtyWorld(n=2000, v=1.0), study, [0.59, 0.41, 0.16])
    assert kind == "success"
    assert "Über 10 Läufe mit n = 2000" in text and "**0,600** (früh), **0,400** (mittel), **0,200** (spät)" in text
    assert "Hazard-Verhältnisse 1,82, 1,49 und 1,22" in text and "ln 2,00 = 0,693" in text
    assert "0,590, 0,410, 0,160" in text and "früh minus spät: 0,400" in text and "den Faktor 2,00" in text


def test_verdikt_streuung_ohne_streuung_nennt_den_abstand_der_phasen():
    kind, text = ev.verdict_frailty(FrailtyWorld(v=0.0), _study([[0.70, 0.69, 0.68]] * 5), [0.693] * 3)
    assert kind == "info" and "Ohne Streuung (v = 0) bleibt das Verhältnis in jeder Phase beim Faktor" in text and "ist 0,020" in text


def test_verdikt_streuung_ohne_wirkung_unterscheidet_rauschen_und_auffaelligkeit():
    noise = _study([[0.01, -0.01, 0.0], [-0.01, 0.01, 0.0]] * 5)
    kind, text = ev.verdict_frailty(FrailtyWorld(factor=1.0), noise, [0.0] * 3)
    assert kind == "info" and "Ohne Wirkung gibt es keinen Drift" in text and "innerhalb von 3 Standardfehlern" in text
    kind, text = ev.verdict_frailty(FrailtyWorld(factor=1.0), _study([[0.5, 0.0, 0.0]] * 6), [0.0] * 3)
    assert kind == "warning" and "mehr als 3 Standardfehler" in text


def test_verdikt_streuung_ohne_schaetzer_ist_eine_warnung():
    kind, text = ev.verdict_frailty(FrailtyWorld(), _study([[0.5, float("nan"), 0.2]] * 4), [0.5] * 3)
    assert kind == "warning" and "in keinem der 4 Läufe" in text


def test_nachbeobachtung_nennt_kuerzeste_und_laengste_laenge():
    arr = np.tile([0.62, 0.55, 0.49, 0.44, 0.39, 0.35], (5, 1))
    study = M.FrailtyStudy(np.zeros((5, 3)), arr, (("ok", 45),))
    text = ev.follow_up_text(FrailtyWorld(), study, [math.log(1.85), 0, 0, 0, 0, math.log(1.41)])
    assert "bis Tag 2 ist das Hazard-Verhältnis im Mittel 1,86" in text and "aller Container 1,42" in text and "Wahrheit je Container: 2,00" in text
    assert "Grenzwerte 1,85 und 1,41" in text
    empty = M.FrailtyStudy(np.zeros((2, 3)), np.full((2, 6), np.nan), ())
    assert "keinen Schätzer" in ev.follow_up_text(FrailtyWorld(), empty, [0.0] * 6)


def test_mittlere_anfaelligkeit_der_uebrigen_von_hand():
    """v = 1, t = 10: 1/(1 + 1 · 0,1 · 10) = 1/2 (Kontrolle), 1/(1 + 0,2 · 10) = 1/3 (Maßnahme), Verhältnis 2/3 = HR/f = (1 + 1)/(1 + 2) aus der Formel."""
    rows = {r["t"]: r for r in ev.survivor_frailty_rows(FrailtyWorld(n=2000, v=1.0), times=(0.0, 10.0))}
    assert rows[10.0]["formula0"] == pytest.approx(1 / 2) and rows[10.0]["formula1"] == pytest.approx(1 / 3) and rows[10.0]["hr_over_f"] == pytest.approx(2 / 3)
    assert rows[0.0]["formula0"] == 1.0 and rows[0.0]["hr_over_f"] == 1.0 and rows[0.0]["alive0"] + rows[0.0]["alive1"] == 2000
    assert rows[10.0]["ratio"] == pytest.approx(2 / 3, abs=0.06)


# ------------------------------------------------------------------------------------------------ Experiment 3


def _random(only, both):
    return M.RandomStudy(np.array(only, float), np.array(both, float), np.ones(len(only)), (("ok", 2 * len(only)),))


def test_verdikt_zweites_merkmal_erfolgsfall():
    kind, text = ev.verdict_random(RandomWorld(n=2000, gamma=1.0), _random([0.44, 0.45, 0.43, 0.44], [0.70, 0.69, 0.71, 0.70]), 0.438)
    assert kind == "success"
    assert "Über 4 Läufe mit n = 2000" in text and "**0,440** (± 0,004)" in text and "**0,700** (± 0,004)" in text and "ln 2,00 = 0,693" in text
    assert "beim Weglassen des zweiten Merkmals 0,438" in text and "ein Störfaktor ist hier ausgeschlossen" in text


def test_verdikt_zweites_merkmal_ohne_wirkung_von_gamma_oder_massnahme():
    kind, text = ev.verdict_random(RandomWorld(gamma=0.0), _random([0.69, 0.70], [0.69, 0.70]), 0.693)
    assert kind == "info" and "Ohne Wirkung des zweiten Merkmals (γ = 0,00, Faktor 2,00)" in text and "stimmen im Mittel überein" in text
    kind, text = ev.verdict_random(RandomWorld(factor=1.0), _random([0.0, 0.01], [0.0, 0.01]), 0.0)
    assert kind == "info" and "Ohne Wirkung der Maßnahme" in text
    assert ev.verdict_random(RandomWorld(), _random([float("nan")] * 2, [0.7, 0.7]), 0.4)[0] == "warning"


# ------------------------------------------------------------------------------------------------ Experiment 4


def _stay(**kw):
    base = dict(tau=10.0, n=2000, s0=0.458, s0_band=(0.43, 0.48), s1=0.318, s1_band=(0.29, 0.34), s_diff=-0.14, s_diff_band=(-0.183, -0.098),
                rmst0=6.55, rmst1=5.58, rmst_diff=-0.97, rmst_se=0.17, rmst_band=(-1.3, -0.64), rmst_status="ok", truth_s0=0.5, truth_s1=1 / 3,
                truth_rmst0=6.93, truth_rmst1=5.49, cox_hr_all=1.34, cox_ok=True, phase_hr=(1.35, 1.58, 1.21), km0={}, km1={})
    base.update(kw)
    return M.Stay(**base)


def test_verdikt_stay_nennt_anteile_differenzen_intervalle_und_wahrheit():
    kind, text = ev.verdict_stay(_stay())
    assert kind == "info"
    assert "Nach 10 Tagen" in text and "(n = 2000)" in text and "**45,8 %** der Kontrolle und **31,8 %**" in text and "(Wahrheit 50,0 % und 33,3 %)" in text
    assert "**−14,0 Prozentpunkte**" in text and "−18,3 bis −9,8" in text and "Wahrheit −16,7" in text
    assert "6,55 Tage in der Kontrolle und 5,58" in text and "(Wahrheit 6,93 und 5,49)" in text and "**−0,97 Tage**" in text and "−1,30 bis −0,64" in text and "Wahrheit −1,44" in text


def test_verdikt_stay_ohne_standardfehler_ist_eine_warnung():
    kind, text = ev.verdict_stay(_stay(rmst_status="varianz_null"))
    assert kind == "warning" and "keinen Standardfehler" in text


def test_text_der_phasen_derselben_daten():
    assert ev.phase_table_text(_stay()) == "Dieselben Daten als Hazard-Verhältnis gelesen: ein einziges über alle Zeiten 1,34, je Phase (früh, mittel, spät) 1,35, 1,58, 1,21."
    assert "–" in ev.phase_table_text(_stay(phase_hr=(1.35, float("nan"), 1.21)))
    assert "keinen Cox-Schätzer" in ev.phase_table_text(_stay(cox_ok=False))
