"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung, Aussagen der Hilfetexte gegen die Rechnung."""

import hzv_constants as C
import hzv_model as M
import hzv_presets as P
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld


def _frailty(preset):
    return FrailtyWorld(n=preset.get("n", C.DEFAULT_N), v=preset.get("v", C.DEFAULT_V), factor=preset.get("factor", C.DEFAULT_FACTOR), seed=preset.get("seed", C.DEFAULT_SEED))


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert C.PRESETS["Standard (v = 1, Faktor 2)"] == {}
    assert _frailty({}) == FrailtyWorld() and RandomWorld().n == C.DEFAULT_N and ClassWorld().factor == C.DEFAULT_FACTOR


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_defaults_lie_inside_their_bounds():
    for key, spec in P.SETTING_SPECS.items():
        assert P._clean(spec, spec.default) == spec.default, key


def test_permalink_values_are_clamped():
    assert P._clean(P.SETTING_SPECS["n_slider"], 10**6) == C.N_MAX
    assert P._clean(P.SETTING_SPECS["n_slider"], -5) == C.N_MIN
    assert P._clean(P.SETTING_SPECS["v_slider"], 99.0) == C.V_MAX
    assert P._clean(P.SETTING_SPECS["factor_slider"], 0.2) == C.FACTOR_MIN


def test_float_comparison_tolerates_slider_rounding():
    assert P._gleich(0.7000000000000001, 0.7) and not P._gleich(0.75, 0.7) and P._gleich(300, 300) and not P._gleich("a", "b")


def test_preset_help_statements_hold_against_the_calculation():
    """Aussagen der Preset-Hilfen, soweit sie etwas über Zahlen sagen: gerechnet, nicht nur behauptet."""
    # „Keine Streuung“: das Hazard-Verhältnis bleibt in jeder Phase beim Faktor (Grenzwert und Formel)
    w0 = _frailty(C.PRESETS["Keine Streuung (v = 0)"])
    assert all(abs(b - 0.6931471805599453) < 1e-9 for b in M.population_windows(w0))
    # „Starke Streuung (v = 2)“: der späte Grenzwert liegt weit unter dem Faktor (ln 2 = 0,693)
    w2 = _frailty(C.PRESETS["Starke Streuung (v = 2)"])
    assert M.population_windows(w2)[2] < 0.1 < 0.693
    # „Keine Wirkung (Faktor 1)“: Grenzwert in jeder Phase 0
    w1 = _frailty(C.PRESETS["Keine Wirkung (Faktor 1)"])
    assert all(abs(b) < 1e-9 for b in M.population_windows(w1))
    # „Zwei fast gleiche Klassen“: der Tiefpunkt liegt näher am Faktor als im Standardbeispiel
    p = C.PRESETS["Zwei fast gleiche Klassen"]
    close = M.class_hr_minimum(ClassWorld(rate_a=p["ratea"], rate_b=p["rateb"]))[1]
    assert close > M.class_hr_minimum(ClassWorld())[1] and close < 2.0
    # „Zweites Merkmal stark (γ = 2)“: größerer Abstand zwischen dem Grenzwert nur mit x und ln f als beim Standard (γ = 1)
    gap = lambda g: abs(M.population_random(RandomWorld(gamma=g)) - 0.6931471805599453)  # noqa: E731
    assert gap(C.PRESETS["Zweites Merkmal stark (γ = 2)"]["gamma"]) > gap(1.0)
