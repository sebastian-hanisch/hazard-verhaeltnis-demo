"""Kennzahlen, Texte und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text an handgesetzten Werten)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import hzv_constants as C
import hzv_model as M
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld, simulate_frailty


def num(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    """Anteil als Prozentzahl, zum Beispiel 0,456 → „46 %“."""
    return f"{num(x * 100, digits)} %"


def frac(value) -> str:
    """Bruch als Text: 5/2, −1/3, 0 (ganze Zahlen ohne Nenner)."""
    f = Fraction(value)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}".replace("-", "−")


def hr_text(beta: float, digits: int = 2) -> str:
    """Hazard-Verhältnis exp(β̂) als Text; „–“ ohne Schätzer."""
    return num(math.exp(beta), digits) if math.isfinite(beta) else "–"


# ------------------------------------------------------------------------------------------------ Experiment 1


@dataclass(frozen=True)
class ClassAnalysis:
    world: ClassWorld
    exact: dict           # Raten bei t = 0 in Brüchen
    rows: list            # Handtabelle (feste Zeiten)
    probe_row: dict       # eine Zeile beim Zeitpunkt des Reglers
    t_min: float          # Tiefpunkt des Hazard-Verhältnisses auf [0, Horizont]
    hr_min: float
    hr_end: float         # Hazard-Verhältnis am Ende des Horizonts
    share_b_end: float    # Anteil der Klasse B unter den Übrigen der Kontrolle am Ende des Horizonts
    no_cross: bool        # die Überlebenskurven kreuzen sich nicht


def analyse_classes(world: ClassWorld, probe: float) -> ClassAnalysis:
    t_min, hr_min = M.class_hr_minimum(world)
    return ClassAnalysis(world, M.class_zero_rates_exact(world), M.class_rows(world, C.CLASS_TIMES), M.class_rows(world, [probe])[0], t_min, hr_min,
                         float(M.class_hr_curve(world, C.CLASS_HORIZON)), M.class_rows(world, [C.CLASS_HORIZON])[0]["share_b_control"], M.class_survivals_never_cross(world))


def verdict_classes(a: ClassAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 1. Ohne Wirkung (Faktor 1) oder ohne Unterschied der Klassen gibt es nichts, was das Verhältnis verändern könnte; sonst Tiefpunkt und Rückkehr."""
    w = a.world
    f = num(w.factor, 2)
    if w.factor == 1.0:
        return "info", "Faktor 1: Die Maßnahme ändert keine Rate. Das Verhältnis der marginalen Raten ist zu jedem Zeitpunkt 1, die Zusammensetzung der übrigen Container spielt keine Rolle."
    if w.rate_a == w.rate_b:
        return "info", (f"Beide Klassen haben dieselbe Rate ({num(w.rate_a, 1)} je Tag): Es gibt keine Streuung, das Verhältnis der marginalen Raten bleibt zu jedem Zeitpunkt bei {f}.")
    head = (f"Das Verhältnis der marginalen Raten startet bei {f} (für jede Aufteilung), fällt bis Tag {num(a.t_min, 2)} auf {num(a.hr_min, 2)} und liegt am Ende des gezeigten Bereichs "
            f"(Tag {C.CLASS_HORIZON:g}) bei {num(a.hr_end, 2)}; dort gehören in der Kontrolle noch {pct(a.share_b_end, 1)} der Übrigen zur schnelleren Klasse B (zu Beginn {pct(w.share_b)}). "
            f"Jeder einzelne Container hat zu jedem Zeitpunkt das Verhältnis {f}. ")
    tail = "Die Überlebenskurven der beiden Gruppen kreuzen sich nicht." if a.no_cross else "Die Überlebenskurven der beiden Gruppen kreuzen sich."
    return "success", head + tail


# ------------------------------------------------------------------------------------------------ Experiment 2


def survivor_frailty_rows(world: FrailtyWorld, times=(0.0, 2.0, 5.0, 10.0, 20.0), run: int = 0) -> list[dict]:
    """Mittlere Anfälligkeit der noch vorhandenen Container je Gruppe bei Zeit t (aus dem Datensatz `run`; nur die Simulation kennt Z), dazu das Verhältnis
    behandelt/Kontrolle, die Formel HR(t)/f und die Formel für die mittlere Anfälligkeit der Übrigen 1/(1 + v·λ·f^x·t)."""
    sm = simulate_frailty(world, run)
    rows = []
    for t in times:
        alive = sm.time > t
        z0, z1 = sm.z[alive & (sm.x == 0)], sm.z[alive & (sm.x == 1)]
        m0 = float(z0.mean()) if z0.size else float("nan")
        m1 = float(z1.mean()) if z1.size else float("nan")
        rows.append({"t": t, "alive0": int(z0.size), "alive1": int(z1.size), "mean0": m0, "mean1": m1, "ratio": m1 / m0 if z0.size and z1.size else float("nan"),
                     "formula0": 1.0 / (1.0 + world.v * C.BASE_RATE * t), "formula1": 1.0 / (1.0 + world.v * C.BASE_RATE * world.factor * t),
                     "hr_over_f": float(M.gamma_hr(world.v, C.BASE_RATE, world.factor, t)) / world.factor})
    return rows


def verdict_frailty(world: FrailtyWorld, study: M.FrailtyStudy, pops: list[float]) -> tuple[str, str]:
    """Verdikt zu Experiment 2: Cox-Schätzer je Phase (Mittel über die Läufe) gegen die Wahrheit je Container (ln f) und den Grenzwert."""
    cols = [M.column(study.windows[:, j], study.reps) for j in range(3)]
    if any(c.n_ok == 0 for c in cols):
        return "warning", f"In mindestens einer Phase gibt es in keinem der {study.reps} Läufe einen Cox-Schätzer (siehe Hinweis darunter); ohne Schätzer gibt es nichts zu vergleichen."
    truth = math.log(world.factor)
    head = (f"Über {study.reps} Läufe mit n = {world.n} Containern: Cox-β̂ je Phase im Mittel "
            f"**{num(cols[0].mean, 3)}** (früh), **{num(cols[1].mean, 3)}** (mittel), **{num(cols[2].mean, 3)}** (spät), also Hazard-Verhältnisse "
            f"{hr_text(cols[0].mean)}, {hr_text(cols[1].mean)} und {hr_text(cols[2].mean)}. Wahrheit je Container: ln {num(world.factor, 2)} = {num(truth, 3)} in jeder Phase. ")
    if world.factor == 1.0:
        within = all(abs(c.mean) <= 3 * c.se_of_mean for c in cols)
        if within:
            return "info", head + "**Ohne Wirkung gibt es keinen Drift:** Alle drei Phasen unterscheiden sich nicht von 0 (innerhalb von 3 Standardfehlern des Mittels); es gibt nichts, was die Auswahl der Übrigen verzerren könnte."
        return "warning", head + "Mindestens eine Phase liegt mehr als 3 Standardfehler des Mittels von 0 entfernt; bei Faktor 1 ist das ein Hinweis auf einen Fehler oder auf eine ungewöhnliche Zufallsziehung."
    if world.v == 0.0:
        spread = max(c.mean for c in cols) - min(c.mean for c in cols)
        return "info", head + (f"**Ohne Streuung (v = 0) bleibt das Verhältnis in jeder Phase beim Faktor:** der größte Abstand zwischen den Phasen im Mittel ist {num(spread, 3)}; "
                               "es gibt keine unterschiedlich anfälligen Container, die sich aussortieren könnten.")
    drop = cols[0].mean - cols[2].mean
    pop = f"Grenzwert für unendlich viele Container: {num(pops[0], 3)}, {num(pops[1], 3)}, {num(pops[2], 3)}. "
    return "success", head + pop + (f"**Das Verhältnis der noch vorhandenen Container sinkt von Phase zu Phase** (früh minus spät: {num(drop, 3)}), obwohl jeder einzelne Container zu jedem Zeitpunkt "
                                    f"den Faktor {num(world.factor, 2)} erfährt. Die Kurven kreuzen sich nicht; das Verhältnis fällt gegen 1, nicht darunter.")


def follow_up_text(world: FrailtyWorld, study: M.FrailtyStudy, pops: list[float]) -> str:
    """Satz zur Nachbeobachtung: erste und letzte Länge, Hazard-Verhältnis im Mittel und Grenzwert."""
    cols = [M.column(study.follow_ups[:, j], study.reps) for j in range(len(C.FOLLOW_UPS))]
    if cols[0].n_ok == 0 or cols[-1].n_ok == 0:
        return "Für die kürzeste oder die längste Nachbeobachtung gibt es keinen Schätzer."
    return (f"Dasselbe Datenmaterial, nur kürzer oder länger beobachtet: bei Nachbeobachtung bis Tag {C.FOLLOW_UPS[0]:g} ist das Hazard-Verhältnis im Mittel {hr_text(cols[0].mean)}, "
            f"bei Beobachtung bis zur Abholung aller Container {hr_text(cols[-1].mean)} (Wahrheit je Container: {num(world.factor, 2)}; Grenzwerte {num(math.exp(pops[0]), 2)} und {num(math.exp(pops[-1]), 2)}).")


def failure_text(counts: tuple, reps: int, label: str = "Anpassungen") -> str:
    """Hinweis auf Anpassungen ohne Cox-Schätzer (leer, wenn es keine gibt): Zahl und Gründe."""
    d = dict(counts)
    bad = sum(v for k, v in d.items() if k != "ok")
    if bad == 0:
        return ""
    names = {"ohne_ereignis": "ohne Ereignis", "singulaer": "Information 0", "keine_konvergenz": "keine Konvergenz (häufig perfekte Trennung)"}
    parts = ", ".join(f"{names.get(k, k)}: {v}" for k, v in sorted(d.items()) if k != "ok")
    return (f"**{bad} von {sum(d.values())} {label}** (über {reps} Läufe) haben keinen endlichen Cox-Schätzer ({parts}). Sie stehen weder im Mittel noch im Diagramm; "
            "das Verfahren meldet den Grund, statt eine Zahl zu liefern.")


# ------------------------------------------------------------------------------------------------ Experiment 3


def verdict_random(world: RandomWorld, study: M.RandomStudy, pop_x: float) -> tuple[str, str]:
    """Verdikt zu Experiment 3: Cox nur mit x gegen Cox mit x und z, Wahrheit je Container und Grenzwert der Anpassung nur mit x."""
    a, b = M.column(study.only_x, study.reps), M.column(study.with_z, study.reps)
    if a.n_ok == 0 or b.n_ok == 0:
        return "warning", "Es gibt in keinem Lauf einen Cox-Schätzer (siehe Hinweis darunter)."
    truth = math.log(world.factor)
    head = (f"Über {study.reps} Läufe mit n = {world.n} Containern und zufälliger Zuteilung: Cox nur mit der Maßnahme β̂ im Mittel **{num(a.mean, 3)}** (± {num(a.se_of_mean, 3)}), "
            f"mit der Maßnahme und dem zweiten Merkmal **{num(b.mean, 3)}** (± {num(b.se_of_mean, 3)}); Wahrheit je Container ln {num(world.factor, 2)} = {num(truth, 3)}. ")
    if world.gamma == 0.0 or world.factor == 1.0:
        why = "Ohne Wirkung des zweiten Merkmals" if world.gamma == 0.0 else "Ohne Wirkung der Maßnahme"
        return "info", head + f"{why} (γ = {num(world.gamma, 2)}, Faktor {num(world.factor, 2)}) gibt es nichts, was sich beim Weglassen ändern könnte: beide Schätzer stimmen im Mittel überein."
    return "success", head + (f"Der Grenzwert für unendlich viele Container ist beim Weglassen des zweiten Merkmals {num(pop_x, 3)}. **Das Hazard-Verhältnis ändert sich, wenn man ein Merkmal weglässt,** "
                              "das mit der Zuteilung nichts zu tun hat; ein Störfaktor ist hier ausgeschlossen, weil zufällig zugeteilt wird.")


# ------------------------------------------------------------------------------------------------ Experiment 4


def verdict_stay(st: M.Stay) -> tuple[str, str]:
    """Verdikt zu Experiment 4: Überlebenswahrscheinlichkeit und mittlere Lebenszeit bis τ je Gruppe, Differenzen mit Intervall, Wahrheit."""
    tau = f"{st.tau:g}"
    s = (f"Nach {tau} Tagen sind in diesem Datensatz (n = {st.n}) noch **{pct(st.s0, 1)}** der Kontrolle und **{pct(st.s1, 1)}** der Maßnahme-Gruppe da "
         f"(Wahrheit {pct(st.truth_s0, 1)} und {pct(st.truth_s1, 1)}): Differenz **{num(100 * st.s_diff, 1)} Prozentpunkte** "
         f"(95-%-Intervall {num(100 * st.s_diff_band[0], 1)} bis {num(100 * st.s_diff_band[1], 1)}; Wahrheit {num(100 * (st.truth_s1 - st.truth_s0), 1)}). ")
    if st.rmst_status != "ok":
        return "warning", s + "Für die mittlere Lebenszeit gibt es keinen Standardfehler (Varianz 0): keine Differenz mit Intervall."
    r = (f"Die mittlere Verweilzeit bis Tag {tau} ist {num(st.rmst0, 2)} Tage in der Kontrolle und {num(st.rmst1, 2)} in der Maßnahme-Gruppe (Wahrheit {num(st.truth_rmst0, 2)} und {num(st.truth_rmst1, 2)}): "
         f"Differenz **{num(st.rmst_diff, 2)} Tage** (95-%-Intervall {num(st.rmst_band[0], 2)} bis {num(st.rmst_band[1], 2)}; Wahrheit {num(st.truth_rmst1 - st.truth_rmst0, 2)}).")
    return "info", s + r


def phase_table_text(st: M.Stay) -> str:
    """Satz zu den Hazard-Verhältnissen derselben Daten (ein einzelnes Verhältnis über alles und eines je Phase)."""
    if not st.cox_ok:
        return "Für diesen Datensatz gibt es keinen Cox-Schätzer."
    ph = ", ".join(num(x, 2) if math.isfinite(x) else "–" for x in st.phase_hr)
    return f"Dieselben Daten als Hazard-Verhältnis gelesen: ein einziges über alle Zeiten {num(st.cox_hr_all, 2)}, je Phase (früh, mittel, spät) {ph}."
