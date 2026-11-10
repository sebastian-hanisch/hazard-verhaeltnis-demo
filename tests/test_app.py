"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Ergänzungen dieser Demo (der Anfang der Datei ist die Vorlage tools/templates/test_app_smoke.py) ------------------------------------------------
# Regler dieser App: Slider (Faktor, n, Raten und Anteil der Klassen, Zeit zum Ausprobieren, Varianz v, γ, τ), Zahlenfeld (Zufalls-Seed).
# Randfälle (keine Streuung, keine Wirkung, kleinste Stichprobe) müssen als Meldung erscheinen, nie als Fehler.
import pytest  # noqa: E402

import hzv_constants as C  # noqa: E402


def _texts(at) -> str:
    return " ".join(m.value for kind in (at.success, at.info, at.warning, at.error, at.markdown, at.caption) for m in kind)


@pytest.mark.parametrize("v", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_every_variance_of_the_grid_runs_and_reports_its_branch(v):
    at = _fresh_app()
    at.slider(key="v_slider").set_value(v).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    if v == 0.0:
        assert "Ohne Streuung (v = 0) bleibt das Verhältnis in jeder Phase beim Faktor" in _texts(at)
    else:
        assert "sinkt von Phase zu Phase" in _texts(at)


def test_without_effect_the_app_says_there_is_no_drift():
    at = _fresh_app()
    at.slider(key="factor_slider").set_value(1.0).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    text = _texts(at)
    assert "Ohne Wirkung gibt es keinen Drift" in text and "Faktor 1: Die Maßnahme ändert keine Rate" in text
    assert "Ohne Wirkung der Maßnahme" in text   # Experiment 3


def test_extreme_combinations_of_all_controls_do_not_raise():
    """Alle Regler zugleich am Minimum bzw. am Maximum (Eckfall-Matrix)."""
    mins = {"factor_slider": C.FACTOR_MIN, "n_slider": C.N_MIN, "ratea_slider": C.RATE_A_MIN, "rateb_slider": C.RATE_B_MIN, "shareb_slider": C.SHARE_B_MIN, "probe_slider": C.PROBE_MIN,
            "v_slider": C.V_MIN, "gamma_slider": C.GAMMA_MIN, "tau_slider": C.TAU_MIN}
    maxs = {"factor_slider": C.FACTOR_MAX, "n_slider": C.N_MAX, "ratea_slider": C.RATE_A_MAX, "rateb_slider": C.RATE_B_MAX, "shareb_slider": C.SHARE_B_MAX, "probe_slider": C.PROBE_MAX,
            "v_slider": C.V_MAX, "gamma_slider": C.GAMMA_MAX, "tau_slider": C.TAU_MAX}
    for values in (mins, maxs):
        at = _fresh_app()
        for key, value in values.items():
            at.slider(key=key).set_value(int(value) if key in ("n_slider", "tau_slider") else float(value))
        at.run(timeout=TIMEOUT)
        assert not at.exception, (values, [str(e) for e in at.exception])


def test_smallest_sample_with_strong_spread_reports_or_computes_without_error():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(C.N_MIN)
    at.slider(key="v_slider").set_value(C.V_MAX).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Über 200 Läufe mit n = 400" in _texts(at)


def test_seed_extremes_do_not_raise():
    for value in (0, C.SEED_MAX):
        at = _fresh_app()
        at.number_input(key="seed_input").set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (value, [str(e) for e in at.exception])


def test_the_independence_assumption_is_visible_in_the_intro_and_in_the_assumptions_expander():
    at = _fresh_app()
    assert any("Die Zensierung ist unabhängig vom Ereignis" in m.value for m in at.markdown)
    expander = next(e for e in at.expander if e.label == "⚠️ Annahmen, Grenzen und bekannte Probleme")
    assert any("Die Zensierung ist unabhängig vom Ereignis" in m.value for m in expander.markdown)
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)


def test_the_dispute_is_shown_as_a_dispute_with_reading_levels_and_without_a_verdict():
    at = _fresh_app()
    text = _texts(at)
    for needle in ("Hernán (2010)", "Abrahamowicz u. a. (2025)", "Ying und Xu", "Royston und Parmar (2013)", "Volltext über Abruf", "Volltext selbst gelesen", "Keine der Positionen ist hier als Urteil gemeint"):
        assert needle in text, needle
    assert "fachlich umstritten" in text and "entscheidet ihn nicht" in text
    for banned in ("ist falsch", "sind falsch", "widerlegt", "Saatwert", "Schwanz"):
        assert banned not in text, banned


def test_the_page_draws_its_five_figures():
    at = _fresh_app()
    assert len(at.get("plotly_chart")) == 5   # Klassen, Verhältnis über die Zeit, Nachbeobachtung, zweites Merkmal, Kaplan–Meier


def test_permalink_sets_the_controls_clamped_and_the_address_follows_the_controls():
    at = AppTest.from_file(APP_PATH)
    at.query_params["v"] = "2"
    at.query_params["factor"] = "99"            # außerhalb der Grenzen: wird auf das Maximum begrenzt
    at.query_params["tau"] = "5"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="v_slider").value == 2.0 and at.slider(key="factor_slider").value == C.FACTOR_MAX and at.slider(key="tau_slider").value == 5
    at.slider(key="gamma_slider").set_value(1.5).run(timeout=TIMEOUT)
    assert not at.exception
    assert str(at.query_params["gamma"]).strip("[]'") == "1.5" and str(at.query_params["v"]).strip("[]'") == "2.0"
