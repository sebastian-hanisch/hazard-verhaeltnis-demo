"""Einheiten des Rechenkerns an Mini-Instanzen von Hand (DEMO-PLAYBOOK §1b): jede Regel einzeln, das Handbeispiel von Experiment 1 mit Brüchen."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import hzv_constants as C
import hzv_core as K
import hzv_model as M
from hzv_scenario import ClassWorld, FrailtyWorld, Sample

# ------------------------------------------------------------------------------------------------ Experiment 1: Handbeispiel (ueberlebenszeit-planung/handbeispiele.py)


def test_handbeispiel_tabelle_zwei_klassen(hand_world):
    """Raten 1 und 4 je Tag zu je 50 %, Maßnahme verdoppelt; Werte von Hand gerechnet (h = Σ w r e^(−rt) / Σ w e^(−rt))."""
    want = {0.0: (2.5, 5.0, 2.0), 0.5: (1.5473, 2.2846, 1.4765), 1.0: (1.1423, 2.0148, 1.7639), 2.0: (1.0074, 2.0000, 1.9853), 3.0: (1.0004, 2.0000, 1.9993)}
    rows = {r["t"]: r for r in M.class_rows(hand_world, C.CLASS_TIMES)}
    assert set(rows) == set(want)
    for t, (hc, ht, hr) in want.items():
        assert rows[t]["h_control"] == pytest.approx(hc, abs=5e-5)
        assert rows[t]["h_treated"] == pytest.approx(ht, abs=5e-5)
        assert rows[t]["hr"] == pytest.approx(hr, abs=5e-5)


def test_handbeispiel_bei_null_in_bruechen(hand_world):
    ex = M.class_zero_rates_exact(hand_world)
    assert ex["control"] == F(5, 2) and ex["treated"] == F(5) and ex["hr"] == F(2)
    # von Hand: ½ · 1 + ½ · 4 = 5/2; mit Maßnahme ½ · 2 + ½ · 8 = 5
    assert F(1, 2) * 1 + F(1, 2) * 4 == F(5, 2) and F(1, 2) * 2 + F(1, 2) * 8 == 5


def test_klassenanteil_der_uebrigen_von_hand(hand_world):
    """Anteil der Klasse B unter den Übrigen: w_B e^(−r_B t) / Σ w e^(−r t). Bei t = 0,5 in der Kontrolle: e^(−2) / (e^(−0,5) + e^(−2)) (die Hälften kürzen sich)."""
    row = M.class_rows(hand_world, [0.5])[0]
    assert row["share_b_control"] == pytest.approx(math.exp(-2) / (math.exp(-0.5) + math.exp(-2)), abs=1e-12)
    assert row["share_b_treated"] == pytest.approx(math.exp(-4) / (math.exp(-1) + math.exp(-4)), abs=1e-12)


def test_tiefpunkt_und_rueckkehr_des_handbeispiels(hand_world):
    t_min, hr_min = M.class_hr_minimum(hand_world)
    assert t_min == pytest.approx(0.445, abs=0.002) and hr_min == pytest.approx(1.4699, abs=5e-4)
    assert float(M.class_hr_curve(hand_world, 6.0)) == pytest.approx(2.0, abs=1e-6)   # am Ende fast nur noch Klasse A
    assert M.class_survivals_never_cross(hand_world)


def test_ohne_unterschied_der_klassen_oder_ohne_wirkung_gibt_es_keinen_tiefpunkt():
    same = ClassWorld(rate_a=2.0, rate_b=2.0, share_b=0.5, factor=2.0)
    assert M.class_hr_minimum(same)[1] == pytest.approx(2.0, abs=1e-9)
    one = ClassWorld(factor=1.0)
    assert M.class_hr_minimum(one)[1] == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------------------------------------ Gesetze


def test_geschlossene_formel_an_handwerten():
    """HR(t) = f (1 + vλt) / (1 + f vλt); v = 1, λ = 0,1, f = 2 bei t = 5,5: 2 · 1,55 / 2,1 = 1,4762 (ln = 0,389)."""
    hr = float(M.gamma_hr(1.0, 0.1, 2.0, 5.5))
    assert hr == pytest.approx(2 * 1.55 / 2.1, abs=1e-12) and hr == pytest.approx(1.4762, abs=5e-5)
    assert math.log(hr) == pytest.approx(0.389, abs=5e-4)
    assert float(M.gamma_hr(1.0, 0.1, 2.0, 0.0)) == 2.0 and float(M.gamma_hr(0.0, 0.1, 2.0, 100.0)) == 2.0


def test_gamma_gesetz_ueberlebensfunktion_rate_und_verhaeltnis():
    law0, law1 = M.frailty_laws(FrailtyWorld(v=1.0))
    t = 5.5
    assert float(law0.surv(t)) == pytest.approx(1 / (1 + 0.1 * t), abs=1e-14)          # v = 1: S = 1 / (1 + λt)
    assert float(law1.surv(t)) == pytest.approx(1 / (1 + 0.2 * t), abs=1e-14)
    assert float(law0.hazard(t)) == pytest.approx(0.1 / (1 + 0.1 * t), abs=1e-14)
    assert float(law1.hazard(t) / law0.hazard(t)) == pytest.approx(float(M.gamma_hr(1.0, 0.1, 2.0, t)), abs=1e-12)
    assert float(law0.dens(t)) == pytest.approx(float(law0.hazard(t) * law0.surv(t)), abs=1e-15)


def test_gamma_gesetz_randfaelle_v_null_und_extreme_v():
    exp_law = M.GammaLaw(0.1, 0.0)
    assert float(exp_law.surv(10.0)) == pytest.approx(math.exp(-1.0), abs=1e-15) and float(exp_law.hazard(10.0)) == 0.1
    tiny = M.GammaLaw(0.1, 1e-7)                       # v → 0: stetig gegen die Exponentialverteilung
    assert float(tiny.surv(10.0)) == pytest.approx(math.exp(-1.0), abs=1e-6)
    huge = M.GammaLaw(0.1, 1e6)                        # extrem große Streuung: fast alle Container haben eine winzige Rate
    assert 0.99 < float(huge.surv(100.0)) <= 1.0 and float(huge.hazard(100.0)) < 1e-4
    assert M.GammaLaw(0.1, 1.0).rmst(0.0) == 0.0


def test_mittlere_lebenszeit_geschlossen_gegen_zerlegte_summe():
    """∫₀^τ S dt von Hand für v = 1 (ln(1 + λτ)/λ) und v = 0 ((1 − e^(−λτ))/λ), und gegen eine Trapezsumme für v = 2."""
    assert M.GammaLaw(0.1, 1.0).rmst(10.0) == pytest.approx(math.log(2.0) / 0.1, abs=1e-12)
    assert M.GammaLaw(0.1, 0.0).rmst(10.0) == pytest.approx((1 - math.exp(-1.0)) / 0.1, abs=1e-12)
    law = M.GammaLaw(0.2, 2.0)
    t = np.linspace(0, 10, 200001)
    s = law.surv(t)
    assert law.rmst(10.0) == pytest.approx(float(np.sum((s[1:] + s[:-1]) / 2 * np.diff(t))), abs=1e-8)


def test_mischung_bei_gleichen_raten_ist_die_exponentialverteilung():
    mix = M.Mixture((0.3, 0.3), (0.4, 0.6))
    assert float(mix.surv(2.0)) == pytest.approx(math.exp(-0.6), abs=1e-14) and float(mix.hazard(2.0)) == pytest.approx(0.3, abs=1e-14)


# ------------------------------------------------------------------------------------------------ Grenzwert des Cox-Schätzers


def test_grenzwert_ohne_streuung_ist_der_faktor_in_jeder_phase():
    for factor in (1.5, 2.0, 3.0):
        world = FrailtyWorld(v=0.0, factor=factor)
        for b in M.population_windows(world) + M.population_follow_ups(world):
            assert b == pytest.approx(math.log(factor), abs=1e-9)


def test_grenzwert_ohne_wirkung_ist_null():
    for v in (0.0, 1.0, 3.0):
        for b in M.population_windows(FrailtyWorld(v=v, factor=1.0)):
            assert b == pytest.approx(0.0, abs=1e-9)


def test_grenzwert_eines_fensters_ohne_ereignisse_ist_nan():
    """Ab Tag 1e5 ist bei v = 0 (Exponentialverteilung, Rate 0,1) kein Container mehr da (S = e^(−1e4) = 0 in Gleitkomma): kein Schätzer, nan statt einer Zahl."""
    law0, law1 = M.frailty_laws(FrailtyWorld(v=0.0))
    assert math.isnan(M.population_beta(law0, law1, 1e5, math.inf))


def test_quadratur_integriert_bekannte_funktionen():
    t, w = M._quadrature(0.0, 3.0)
    assert float(np.sum(w * np.exp(-t))) == pytest.approx(1 - math.exp(-3.0), abs=1e-10)
    t, w = M._quadrature(8.0, math.inf)
    assert float(np.sum(w * 0.1 * np.exp(-0.1 * t))) == pytest.approx(math.exp(-0.8), abs=1e-9)


# ------------------------------------------------------------------------------------------------ schnelle Cox-Anpassung


def test_cox_schnell_handbeispiel_vier_container():
    r = M.cox_fast([1, 2, 3, 4], [1, 1, 1, 1], [1, 0, 1, 0])
    assert r["ok"] and r["beta"][0] == pytest.approx(0.9406, abs=5e-4)
    assert r["beta"][0] == pytest.approx(K.cox_fit([1, 2, 3, 4], [1, 1, 1, 1], [1, 0, 1, 0])["beta"][0], abs=1e-10)


def test_cox_schnell_meldet_trennung_singularitaet_und_ohne_ereignis():
    sep = M.cox_fast([1, 2, 3, 4, 5, 6], [1] * 6, [1, 1, 1, 0, 0, 0])
    assert sep["ok"] is False and M.status_of(sep) == "keine_konvergenz"
    one = M.cox_fast([1, 2, 3], [1, 0, 0], [1, 0, 0])                         # ein einziges Ereignis: Trennung
    assert one["ok"] is False
    const = M.cox_fast([1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1])
    assert const["ok"] is False and M.status_of(const) == "singulaer"
    none = M.cox_fast([1, 2], [0, 0], [0, 1])
    assert none["ok"] is False and M.status_of(none) == "ohne_ereignis"
    assert M.status_of({"ok": True}) == "ok"


def test_cox_schnell_prueft_die_eingabe():
    with pytest.raises(K.SurvivalError):
        M.cox_fast([0, 1], [1, 1], [0, 1])
    with pytest.raises(K.SurvivalError):
        M.cox_fast([1, 2], [1, 1], [0.0, float("nan")])


# ------------------------------------------------------------------------------------------------ Phasen und Nachbeobachtung


def _sample(times, x):
    return Sample(time=np.array(times, dtype=float), x=np.array(x, dtype=float), z=np.ones(len(times)))


def test_phasen_zerlegen_die_zeiten_von_hand():
    """Zeiten 1, 2, 4, 5, 9, 10, 12: früh = Zeiten bis 3 (zensiert bei 3), mittel = nur Zeiten > 3, ab 3 gezählt, bei 8 zensiert, spät = Zeiten > 8, ab 8."""
    t = [1.0, 2.0, 4.0, 5.0, 9.0, 10.0, 12.0, 3.5, 6.0, 8.5]
    x = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    seen = {}

    def spy(obs, ev, xx):
        seen[len(seen)] = (np.sort(obs).tolist(), int(np.sum(ev)), len(obs))
        return {"ok": False, "beta": np.array([np.nan]), "message": "Ohne Ereignis gibt es keinen Schätzer."}

    M.phase_betas(_sample(t, x), fit=spy)
    assert seen[0] == ([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], 2, 10)     # früh: zwei Ereignisse (1, 2), acht bei 3 zensiert
    assert seen[1] == ([0.5, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0], 4, 8)               # mittel: 3,5 → 0,5; 4 → 1; 5 → 2; 6 → 3; 8,5, 9, 10, 12 bei 8 zensiert (Länge 5)
    assert seen[2][2] == 4 and seen[2][1] == 4 and seen[2][0] == [0.5, 1.0, 2.0, 4.0]    # spät: 9, 10, 12, 8,5 → 1, 2, 4, 0,5 (alle Ereignisse)


def test_nachbeobachtung_zensiert_bei_f_und_laesst_sonst_alles_zu():
    seen = []

    def spy(obs, ev, xx):
        seen.append((float(obs.max()), int(ev.sum())))
        return {"ok": False, "beta": np.array([np.nan]), "message": "Ohne Ereignis gibt es keinen Schätzer."}

    t = [1.0, 3.0, 6.0, 15.0, 30.0, 90.0]
    M.follow_up_betas(_sample(t, [1, 0, 1, 0, 1, 0]), fit=spy)
    assert seen == [(2.0, 1), (5.0, 2), (10.0, 3), (20.0, 4), (50.0, 5), (90.0, 6)]    # Länge 2, 5, 10, 20, 50, ∞ (größte Beobachtungszeit, Zahl der Ereignisse)


def test_phasen_ohne_container_in_einer_phase_geben_nan_statt_fehler():
    betas, states = M.phase_betas(_sample([1.0, 2.0, 2.5, 1.5], [1, 0, 1, 0]))
    assert all(math.isnan(b) for b in betas[1:]) and states[1] == "ohne_ereignis"


def test_spalte_zaehlt_laeufe_ohne_schaetzer_nicht_mit():
    c = M.column([1.0, 2.0, float("nan"), 3.0], 4)
    assert (c.reps, c.n_ok) == (4, 3) and c.mean == pytest.approx(2.0) and c.sd == pytest.approx(1.0) and c.se_of_mean == pytest.approx(1 / math.sqrt(3))
    empty = M.column([float("nan")] * 3)
    assert empty.n_ok == 0 and math.isnan(empty.mean)
    single = M.column([5.0])
    assert single.mean == 5.0 and math.isnan(single.se_of_mean)


# ------------------------------------------------------------------------------------------------ Experiment 4


def test_greenwood_standardfehler_handbeispiel():
    km = K.kaplan_meier([2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1])
    assert M.survival_se(km, 5.0) == pytest.approx(math.sqrt(85 / 1728), abs=1e-12)    # Handbeispiel Stück 1: Var = 85/1728 bei t = 5
    assert M.survival_se(km, 1.0) == 0.0                                               # vor der ersten Ereigniszeit


def test_analyse_stay_wahrheit_ist_die_formel_und_intervall_ordnung():
    world = FrailtyWorld(n=600, v=1.0)
    st = M.analyse_stay(world, 10.0)
    assert st.truth_s0 == pytest.approx(0.5, abs=1e-12) and st.truth_s1 == pytest.approx(1 / 3, abs=1e-12)       # S = 1/(1 + λ f^x τ) bei v = 1
    assert st.truth_rmst0 == pytest.approx(math.log(2.0) / 0.1, abs=1e-12) and st.truth_rmst1 == pytest.approx(math.log(3.0) / 0.2, abs=1e-12)
    assert st.s_diff_band[0] < st.s_diff < st.s_diff_band[1] and st.rmst_band[0] < st.rmst_diff < st.rmst_band[1]
    assert st.s0_band[0] < st.s0 < st.s0_band[1] and st.rmst_status == "ok" and st.n == 600
    assert len(st.phase_hr) == 3


def test_analyse_stay_ist_deterministisch_und_laeufe_unterscheiden_sich():
    w = FrailtyWorld(n=400)
    assert M.analyse_stay(w, 5.0, 0).s_diff == M.analyse_stay(w, 5.0, 0).s_diff
    assert M.analyse_stay(w, 5.0, 0).s_diff != M.analyse_stay(w, 5.0, 1).s_diff
