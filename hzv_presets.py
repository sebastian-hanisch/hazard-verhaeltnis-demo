"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import hzv_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None   # erlaubte Werte (Auswahllisten); ein Wert außerhalb fällt auf den Standard zurück


SETTING_SPECS = {
    "factor_slider": SettingSpec("factor", float, C.DEFAULT_FACTOR, C.FACTOR_MIN, C.FACTOR_MAX),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "ratea_slider": SettingSpec("ratea", float, C.DEFAULT_RATE_A, C.RATE_A_MIN, C.RATE_A_MAX),
    "rateb_slider": SettingSpec("rateb", float, C.DEFAULT_RATE_B, C.RATE_B_MIN, C.RATE_B_MAX),
    "shareb_slider": SettingSpec("shareb", float, C.DEFAULT_SHARE_B, C.SHARE_B_MIN, C.SHARE_B_MAX),
    "probe_slider": SettingSpec("probe", float, C.DEFAULT_PROBE, C.PROBE_MIN, C.PROBE_MAX),
    "v_slider": SettingSpec("v", float, C.DEFAULT_V, C.V_MIN, C.V_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "tau_slider": SettingSpec("tau", int, C.DEFAULT_TAU, C.TAU_MIN, C.TAU_MAX),
}
PRESET_KEYS = {"factor": "factor_slider", "n": "n_slider", "seed": "seed_input", "ratea": "ratea_slider", "rateb": "rateb_slider", "shareb": "shareb_slider",
               "probe": "probe_slider", "v": "v_slider", "gamma": "gamma_slider", "tau": "tau_slider"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clean(spec: SettingSpec, value):
    """Wert auf die Grenzen begrenzen und (bei Auswahllisten) auf einen erlaubten Wert zurückführen."""
    if spec.choices is not None and value not in spec.choices:
        return spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            st.session_state[state_key] = _clean(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler auf ihren Standard, dann die Werte des Presets (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert, wie in `aktives_preset`)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = spec.default
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _gleich(a, b) -> bool:
    """Gleichheit; Gleitkommawerte der Regler (0,7 gegen 0,7000000000000001) mit Toleranz."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
