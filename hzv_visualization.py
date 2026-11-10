"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hzv_constants as C
import hzv_model as M
from hzv_scenario import ClassWorld, FrailtyWorld


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


# ------------------------------------------------------------------------------------------------ Experiment 1


def build_class_curves(world: ClassWorld, probe: float, t_min: float, hr_min: float):
    """Links: marginale Raten der Kontrolle und der Maßnahme über die Zeit. Rechts: ihr Verhältnis, mit dem Faktor als gestrichelter Linie, dem Tiefpunkt und dem Zeitpunkt zum Ausprobieren."""
    grid = np.linspace(0.0, C.CLASS_HORIZON, 241)
    c, t = M.class_laws(world)
    hc, ht = c.hazard(grid), t.hazard(grid)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Marginale Rate der Übrigen (je Tag)", "Verhältnis der marginalen Raten"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=grid, y=hc, mode="lines", line=dict(color=C.COLOR_CONTROL, width=3), name="Kontrolle", hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=grid, y=ht, mode="lines", line=dict(color=C.COLOR_TREATED, width=3), name="Maßnahme", hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=grid, y=ht / hc, mode="lines", line=dict(color=C.COLOR_HR, width=3), name="Verhältnis", hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=world.factor, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text=f"Faktor je Container = {world.factor:g}", annotation_position="top right", row=1, col=2)
    if world.rate_a != world.rate_b and world.factor != 1.0:
        fig.add_trace(go.Scatter(x=[t_min], y=[hr_min], mode="markers", marker=dict(size=11, symbol="triangle-down", color=C.COLOR_TREATED), name="Tiefpunkt",
                                 hovertemplate="Tiefpunkt bei t = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=2)
    row = M.class_rows(world, [probe])[0]
    fig.add_trace(go.Scatter(x=[probe], y=[row["hr"]], mode="markers", marker=dict(size=13, symbol="diamond", color=C.COLOR_SIM), name="Zeit zum Ausprobieren",
                             hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Zeit (Tage)")
    fig.update_yaxes(title_text="Rate", row=1, col=1)
    fig.update_yaxes(title_text="Verhältnis", range=[min(0.9, world.factor * 0.9), max(1.5, world.factor * 1.1)], row=1, col=2)
    return _base(fig, height=380)


# ------------------------------------------------------------------------------------------------ Experiment 2


def build_hr_over_time(world: FrailtyWorld, study: M.FrailtyStudy, pops: list[float]):
    """Hazard-Verhältnis über der Zeit: geschlossene Kurve für die gewählte Streuung (dick) und die Streuungen des Rasters der Messungen (dünn); je Phase der Cox-Schätzer
    der Simulation (Mittel über die Läufe, Linie über die Phase) und sein Grenzwert für unendlich viele Container (Raute)."""
    grid = np.linspace(0.0, C.HR_PLOT_HORIZON, 401)
    fig = go.Figure()
    for v in C.V_GRID:
        if v == world.v:
            continue
        fig.add_trace(go.Scatter(x=grid, y=M.gamma_hr(v, C.BASE_RATE, world.factor, grid), mode="lines", line=dict(color=C.COLOR_TRUTH, width=1, dash="dot"),
                                 name=f"Formel, v = {v:g}", hovertemplate="t = %{x:.1f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=M.gamma_hr(world.v, C.BASE_RATE, world.factor, grid), mode="lines", line=dict(color=C.COLOR_HR, width=3),
                             name=f"Formel, v = {world.v:g} (gewählt)", hovertemplate="t = %{x:.1f}: %{y:.3f}<extra></extra>"))
    first = True
    for j, (a, b) in enumerate(C.PHASES):
        col = M.column(study.windows[:, j], study.reps)
        if col.n_ok == 0:
            continue
        end = min(b, C.HR_PLOT_HORIZON)
        fig.add_trace(go.Scatter(x=[a, end], y=[math.exp(col.mean)] * 2, mode="lines", line=dict(color=C.COLOR_SIM, width=5), name="Cox, Mittel der Läufe", legendgroup="sim",
                                 showlegend=first, hovertemplate=f"{C.PHASE_NAMES[j]}: %{{y:.3f}}<extra></extra>"))
        first = False
        if math.isfinite(pops[j]):
            fig.add_trace(go.Scatter(x=[(a + end) / 2], y=[math.exp(pops[j])], mode="markers", marker=dict(size=10, symbol="diamond", color="#000000"), name="Grenzwert (unendlich viele Container)",
                                     legendgroup="pop", showlegend=(j == 0), hovertemplate=f"{C.PHASE_NAMES[j]}: %{{y:.3f}}<extra></extra>"))
    for edge in (3.0, 8.0):
        fig.add_vline(x=edge, line=dict(color=C.COLOR_TRUTH, width=1, dash="dot"))
    fig.update_xaxes(title_text="Zeit seit Beginn (Tage)", range=[0, C.HR_PLOT_HORIZON])
    fig.update_yaxes(title_text="Hazard-Verhältnis", range=[min(0.9, world.factor * 0.9), max(1.2, world.factor * 1.08)])
    return _base(fig, height=420)


def build_follow_up(world: FrailtyWorld, study: M.FrailtyStudy, pops: list[float]):
    """Einzelnes Hazard-Verhältnis bei wachsender Nachbeobachtung: Mittel der Läufe (Balken, Fehlerbalken ± 1 Streuung über die Läufe) und Grenzwert (Raute)."""
    labels = [f"bis Tag {f:g}" if math.isfinite(f) else "bis zum Ende" for f in C.FOLLOW_UPS]
    cols = [M.column(study.follow_ups[:, j], study.reps) for j in range(len(C.FOLLOW_UPS))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[math.exp(c.mean) if c.n_ok else None for c in cols], marker_color=C.COLOR_SIM, name="Cox, Mittel der Läufe",
                         error_y=dict(type="data", array=[(math.exp(c.mean) * c.sd) if c.n_ok > 1 else 0 for c in cols], visible=True),
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[math.exp(p) if math.isfinite(p) else None for p in pops], mode="markers", marker=dict(size=10, symbol="diamond", color="#000000"),
                             name="Grenzwert (unendlich viele Container)", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=world.factor, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text=f"Faktor je Container = {world.factor:g}", annotation_position="top right")
    fig.update_xaxes(title_text="Länge der Nachbeobachtung")
    fig.update_yaxes(title_text="Hazard-Verhältnis", range=[min(0.9, world.factor * 0.9), max(1.2, world.factor * 1.08)])
    return _base(fig, height=360)


# ------------------------------------------------------------------------------------------------ Experiment 3


def build_random(world, study: M.RandomStudy, pop_x: float):
    """Cox-β̂ für die Maßnahme: nur mit x, mit x und z, und die Wahrheit je Container (ln f). Fehlerbalken: ± 1 Standardfehler des Mittels über die Läufe; Raute: Grenzwert nur mit x."""
    a, b = M.column(study.only_x, study.reps), M.column(study.with_z, study.reps)
    truth = math.log(world.factor)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Cox nur mit x", "Cox mit x und z", "Wahrheit je Container"], y=[a.mean if a.n_ok else None, b.mean if b.n_ok else None, truth],
                         marker_color=[C.COLOR_SIM, C.COLOR_HR, C.COLOR_TRUTH], error_y=dict(type="data", array=[a.se_of_mean if a.n_ok > 1 else 0, b.se_of_mean if b.n_ok > 1 else 0, 0], visible=True),
                         showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    if math.isfinite(pop_x):
        fig.add_trace(go.Scatter(x=["Cox nur mit x"], y=[pop_x], mode="markers", marker=dict(size=11, symbol="diamond", color="#000000"), name="Grenzwert (unendlich viele Container)",
                                 hovertemplate="Grenzwert: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="β̂ für die Maßnahme", range=[min(0.0, truth * 1.1 - 0.05), max(1.0, truth * 1.3)])
    return _base(fig, height=360)


# ------------------------------------------------------------------------------------------------ Experiment 4


def _band(km: dict, step: int, level: float = 0.95):
    """Kaplan–Meier-Treppe mit punktweisem log-log-Band (Greenwood), auf jeden `step`-ten Ereigniszeitpunkt ausgedünnt (nur die Anzeige)."""
    idx = np.arange(0, km["t"].size, max(step, 1))
    t, s, gw = km["t"][idx], km["s"][idx], km["gw"][idx]
    z = 1.959963984540054
    with np.errstate(divide="ignore", invalid="ignore"):
        se_ll = np.sqrt(gw) / np.abs(np.log(s))
        lo, hi = s ** np.exp(z * se_ll), s ** np.exp(-z * se_ll)
    ok = np.isfinite(lo) & np.isfinite(hi) & (s > 0) & (s < 1)
    return t[ok], s[ok], lo[ok], hi[ok], t, s


def build_stay_curves(st: M.Stay, world: FrailtyWorld):
    """Überlebenskurven (Kaplan–Meier) der beiden Gruppen mit 95-%-Band, die wahren Kurven gestrichelt, und die Marke τ."""
    grid = np.linspace(0.0, C.HR_PLOT_HORIZON, 301)
    law0, law1 = M.frailty_laws(world)
    fig = go.Figure()
    for km, color, name, law in ((st.km0, C.COLOR_CONTROL, "Kontrolle", law0), (st.km1, C.COLOR_TREATED, "Maßnahme", law1)):
        step = max(km["t"].size // 150, 1)
        tb, _, lo, hi, tt, ss = _band(km, step)
        fig.add_trace(go.Scatter(x=np.concatenate([tb, tb[::-1]]), y=np.concatenate([hi, lo[::-1]]), fill="toself", fillcolor=color, opacity=0.2, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=np.concatenate([[0.0], tt]), y=np.concatenate([[1.0], ss]), mode="lines", line=dict(color=color, width=2, shape="hv"), name=f"{name}: Kaplan–Meier",
                                 hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=grid, y=law.surv(grid), mode="lines", line=dict(color=color, width=1.5, dash="dash"), name=f"{name}: Wahrheit", hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"))
    fig.add_vline(x=st.tau, line=dict(color=C.COLOR_TRUTH, dash="dot"), annotation_text=f"τ = {st.tau:g}", annotation_position="top right")
    fig.update_xaxes(title_text="Zeit (Tage)", range=[0, C.HR_PLOT_HORIZON])
    fig.update_yaxes(title_text="Anteil noch nicht abgeholt", range=[0, 1.02])
    return _base(fig, height=420)
