"""Rechenkern der Demo: Gesetze der Verweilzeit (Mischung, Gamma), Grenzwert des Cox-Schätzers, schnelle Cox-Anpassung, Studien und Auswertung.

Alles, was die Experimente rechnen, steht hier als kleine Einheiten (DEMO-PLAYBOOK §1b); Zufall nur über hzv_scenario (SplitMix64).
Der kopierte Kern (hzv_core.py) bleibt unverändert; was er nicht hat (schnelle Newton-Anpassung mit vektorisierter Hesse-Matrix,
Grenzwert der Anpassung) steht hier und wird in den Tests gegen ihn und gegen unabhängige Rechenwege geprüft.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import hzv_constants as C
import hzv_core as K
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld, Sample, simulate_frailty, simulate_random

T_INF = 1e30   # „unendlich“ der Quadratur (Tage); die Überlebensfunktionen sind dort in jeder Welt der Demo 0 oder vernachlässigbar


# ------------------------------------------------------------------------------------------------ Gesetze der Verweilzeit


@dataclass(frozen=True)
class Mixture:
    """Mischung exponentieller Klassen mit Anteilen w_k und Raten r_k: S(t) = Σ w_k e^(−r_k t), Dichte f(t) = Σ w_k r_k e^(−r_k t), Rate h = f / S."""

    rates: tuple
    weights: tuple

    def surv(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-np.multiply.outer(t, np.asarray(self.rates, dtype=float))) @ np.asarray(self.weights, dtype=float)

    def dens(self, t):
        t = np.asarray(t, dtype=float)
        r = np.asarray(self.rates, dtype=float)
        return (np.exp(-np.multiply.outer(t, r)) * r) @ np.asarray(self.weights, dtype=float)

    def hazard(self, t):
        s = self.surv(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, self.dens(t) / np.where(s > 0, s, 1.0), 0.0)


@dataclass(frozen=True)
class GammaLaw:
    """Verweilzeit mit gamma-verteilter Anfälligkeit (Mittel 1, Varianz v) und Grundrate `rate`: S(t) = (1 + v·rate·t)^(−1/v), Rate h(t) = rate / (1 + v·rate·t).
    Für v = 0 die Exponentialverteilung (kein Grenzübergang nötig: die Formeln gehen stetig über)."""

    rate: float
    v: float

    def surv(self, t):
        t = np.asarray(t, dtype=float)
        if self.v < 1e-9:
            return np.exp(-self.rate * t)
        return np.exp(-np.log1p(self.v * self.rate * t) / self.v)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        return self.rate / (1.0 + self.v * self.rate * t)

    def dens(self, t):
        return self.hazard(t) * self.surv(t)

    def rmst(self, tau: float) -> float:
        """Mittlere Lebenszeit bis τ in geschlossener Form: ∫₀^τ S(t) dt."""
        a, v = self.rate, self.v
        if v < 1e-9:
            return float(-math.expm1(-a * tau) / a)
        big_l = math.log1p(v * a * tau)
        p = 1.0 - 1.0 / v
        core = big_l if abs(p) < 1e-12 else math.expm1(p * big_l) / p
        return float(core / (v * a))


def gamma_hr(v: float, lam: float, factor: float, t):
    """Geschlossene Formel des Hazard-Verhältnisses bei gamma-verteilter Anfälligkeit (Mittel 1, Varianz v):
    HR(t) = f (1 + v λ t) / (1 + f v λ t). Bei v = 0 gleich f, für t → ∞ (v > 0) gegen 1, bei t = 0 gleich f."""
    t = np.asarray(t, dtype=float)
    return factor * (1.0 + v * lam * t) / (1.0 + factor * v * lam * t)


def frailty_laws(world: FrailtyWorld) -> tuple[GammaLaw, GammaLaw]:
    return GammaLaw(C.BASE_RATE, world.v), GammaLaw(C.BASE_RATE * world.factor, world.v)


def class_laws(world: ClassWorld) -> tuple[Mixture, Mixture]:
    """Kontrolle und Maßnahme von Experiment 1: dieselben Anteile, Raten der Maßnahme mit dem Faktor multipliziert."""
    w = (1.0 - world.share_b, world.share_b)
    return Mixture((world.rate_a, world.rate_b), w), Mixture((world.rate_a * world.factor, world.rate_b * world.factor), w)


# ------------------------------------------------------------------------------------------------ Experiment 1: zwei Klassen von Hand


def class_zero_rates_exact(world: ClassWorld) -> dict:
    """Marginale Raten bei t = 0 in Brüchen: Kontrolle Σ w r, Maßnahme f · Σ w r, und ihr Verhältnis (gleich f für jede Aufteilung).
    Die Reglerwerte werden auf Brüche mit Nenner höchstens 1000 gebracht (Schrittweite der Regler: Zehntel bzw. Halbe)."""
    a, b = (Fraction(x).limit_denominator(1000) for x in (world.rate_a, world.rate_b))
    sb = Fraction(world.share_b).limit_denominator(1000)
    f = Fraction(world.factor).limit_denominator(1000)
    control = (1 - sb) * a + sb * b
    return {"control": control, "treated": f * control, "hr": (f * control) / control}


def class_rows(world: ClassWorld, times) -> list[dict]:
    """Zeilen der Handtabelle: Zeit, marginale Rate der Kontrolle und der Maßnahme, ihr Verhältnis, dazu die Anteile der Klasse B unter den Übrigen."""
    c, t = class_laws(world)
    rows = []
    for x in times:
        hc, ht = float(c.hazard(x)), float(t.hazard(x))
        w = np.asarray(c.weights) * np.exp(-np.asarray(c.rates) * x)
        rows.append({"t": float(x), "h_control": hc, "h_treated": ht, "hr": ht / hc, "share_b_control": float(w[1] / w.sum()),
                     "share_b_treated": float((np.asarray(t.weights) * np.exp(-np.asarray(t.rates) * x))[1] / (np.asarray(t.weights) * np.exp(-np.asarray(t.rates) * x)).sum())})
    return rows


def class_hr_curve(world: ClassWorld, t):
    c, tr = class_laws(world)
    return tr.hazard(t) / c.hazard(t)


def class_hr_minimum(world: ClassWorld, horizon: float = C.CLASS_HORIZON, points: int = 6001) -> tuple[float, float]:
    """Tiefpunkt des Hazard-Verhältnisses auf [0, horizon] (feines Gitter): (Zeit, Wert)."""
    grid = np.linspace(0.0, horizon, points)
    hr = class_hr_curve(world, grid)
    i = int(np.argmin(hr))
    return float(grid[i]), float(hr[i])


def class_survivals_never_cross(world: ClassWorld, horizon: float = C.CLASS_HORIZON, points: int = 2001) -> bool:
    """Die marginalen Überlebenskurven berühren sich nur bei t = 0 (Faktor ≥ 1: die Maßnahme-Kurve liegt nie über der Kontrolle)."""
    grid = np.linspace(0.0, horizon, points)[1:]
    c, t = class_laws(world)
    return bool(np.all(t.surv(grid) <= c.surv(grid) + 1e-15))


# ------------------------------------------------------------------------------------------------ Grenzwert des Cox-Schätzers (unendlich viele Container)


def _quadrature(lo: float, hi: float, panels: int = 400, order: int = 12):
    """Knoten und Gewichte einer zusammengesetzten Gauß–Legendre-Quadratur auf (lo, hi] mit geometrisch wachsenden Feldern (bei lo = 0 beginnt das erste Feld bei 1e-9 · hi, bei hi = ∞ bei 1e-9)."""
    hi = T_INF if not math.isfinite(hi) else hi
    start = lo if lo > 0 else (1e-9 if hi >= T_INF else 1e-9 * hi)
    edges = np.geomspace(start, hi, panels + 1)
    if lo <= 0:
        edges = np.concatenate([[0.0], edges])
    xg, wg = np.polynomial.legendre.leggauss(order)
    a, b = edges[:-1], edges[1:]
    nodes = (0.5 * (b - a)[:, None] * xg[None, :] + 0.5 * (b + a)[:, None]).ravel()
    weights = (0.5 * (b - a)[:, None] * wg[None, :]).ravel()
    return nodes, weights


def population_score(beta: float, s0, s1, f0, f1, weights) -> float:
    """Erwarteter Score der partiellen Likelihood bei β (je Container, ohne den Faktor n/2): ∫ [ f₁ − (f₀ + f₁) · S₁ e^β / (S₀ + S₁ e^β) ] dt.
    Die Risikomenge der Gruppe x zur Zeit t hat den Umfang n/2 · S_x(t) (gleich große Gruppen, alle bis t noch nicht abgeholt)."""
    denom = s0 + s1 * math.exp(beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(denom > 0, s1 * math.exp(beta) / np.where(denom > 0, denom, 1.0), 0.0)
    return float(np.sum(weights * (f1 - (f0 + f1) * m)))


def population_beta(law0, law1, lo: float = 0.0, hi: float = math.inf, panels: int = 400) -> float:
    """Grenzwert des Cox-Schätzers β̂ für ein binäres Merkmal, wenn die Zuteilung 50:50 ist und man die Ereignisse im Fenster (lo, hi] mit Beginn bei lo auswertet
    (Container, die bei lo noch da sind, werden bei hi zensiert): die Nullstelle des erwarteten Scores, per Bisektion. Die Gesetze liefern `surv` und `dens`."""
    t, w = _quadrature(lo, hi, panels)
    s0, s1, f0, f1 = law0.surv(t), law1.surv(t), law0.dens(t), law1.dens(t)
    lo_b, hi_b = -12.0, 12.0
    if population_score(lo_b, s0, s1, f0, f1, w) <= 0 or population_score(hi_b, s0, s1, f0, f1, w) >= 0:
        return float("nan")
    for _ in range(100):
        mid = 0.5 * (lo_b + hi_b)
        if population_score(mid, s0, s1, f0, f1, w) > 0:
            lo_b = mid
        else:
            hi_b = mid
    return 0.5 * (lo_b + hi_b)


def population_windows(world: FrailtyWorld) -> list[float]:
    """Grenzwerte des Cox-Schätzers in den drei Phasen (früh, mittel, spät) für die Welt von Experiment 2."""
    law0, law1 = frailty_laws(world)
    return [population_beta(law0, law1, a, b) for a, b in C.PHASES]


def population_follow_ups(world: FrailtyWorld) -> list[float]:
    """Grenzwerte des Cox-Schätzers für Nachbeobachtung bis F (Beginn bei Tag 0), F aus C.FOLLOW_UPS."""
    law0, law1 = frailty_laws(world)
    return [population_beta(law0, law1, 0.0, f) for f in C.FOLLOW_UPS]


def random_mixtures(world: RandomWorld, nodes: int = 801, half_width: float = 7.0) -> tuple[Mixture, Mixture]:
    """Marginale Verweilzeit je Gruppe in Experiment 3 als Mischung über das zweite Merkmal z ~ N(0, 1) (Trapezregel auf [−7, 7], Raten λ f^x e^(γ z))."""
    z = np.linspace(-half_width, half_width, nodes)
    w = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * (z[1] - z[0])
    w = w / w.sum()
    base = C.BASE_RATE * np.exp(world.gamma * z)
    return Mixture(tuple(base), tuple(w)), Mixture(tuple(base * world.factor), tuple(w))


def population_random(world: RandomWorld) -> float:
    """Grenzwert des Cox-Schätzers nur mit dem Merkmal x (zweites Merkmal weggelassen) bei unendlich vielen Containern; mit beiden Merkmalen ist er exakt ln f."""
    law0, law1 = random_mixtures(world)
    return population_beta(law0, law1, 0.0, math.inf)


# ------------------------------------------------------------------------------------------------ Schnelle Cox-Anpassung


def status_of(fit: dict) -> str:
    """Kurzname des Zustands einer Anpassung: ok, ohne_ereignis, singulaer oder keine_konvergenz (häufig perfekte Trennung)."""
    if fit["ok"]:
        return "ok"
    msg = fit.get("message", "")
    if msg.startswith("Ohne Ereignis"):
        return "ohne_ereignis"
    if "singulär" in msg or "leer" in msg:
        return "singulaer"
    return "keine_konvergenz"


def cox_fast(time, event, x, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Cox-Partial-Likelihood nach Breslow wie `hzv_core.cox_fit`, aber mit vektorisierter Hesse-Matrix statt Python-Schleife über die Ereigniszeiten
    (die Schleife des Kerns kostet bei n = 2000 etwa 0,2 s je Anpassung, hier genügen Millisekunden). Dieselbe Newton-Iteration mit Schrittweitenhalbierung
    und dieselben Abbruch- und Statusregeln; tests/test_model.py und tests/test_oracle_hazard_verhaeltnis.py halten sie gegen den Kern."""
    t, e, en = K.check_data(time, event)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != t.size or not np.all(np.isfinite(x)):
        raise K.SurvivalError("Die Merkmale müssen zu jeder Beobachtung endlich angegeben werden.")
    p = x.shape[1]
    nan = np.full(p, np.nan)
    ts, inv = np.unique(t[e], return_inverse=True)
    if ts.size == 0:
        return {"beta": nan, "se": nan, "loglik": float("nan"), "iterations": 0, "ok": False, "message": "Ohne Ereignis gibt es keinen Schätzer."}
    dcount = np.bincount(inv, minlength=ts.size).astype(float)
    sumx = np.stack([np.bincount(inv, weights=x[e, j], minlength=ts.size) for j in range(p)], axis=1)

    def evaluate(beta):
        s0, s1, s2, shift = K._cox_sums(en, t, x, beta, ts)
        if np.any(s0 <= 0):
            return None
        m = s1 / s0[:, None]
        ll = float(np.sum(sumx @ beta) - np.sum(dcount * (np.log(s0) + shift)))
        grad = (sumx - dcount[:, None] * m).sum(axis=0)
        hess = np.einsum("k,kij->ij", dcount, s2 / s0[:, None, None] - m[:, :, None] * m[:, None, :])
        return ll, grad, hess

    beta = np.zeros(p)
    cur = evaluate(beta)
    if cur is None:
        return {"beta": nan, "se": nan, "loglik": float("nan"), "iterations": 0, "ok": False, "message": "Die Risikomengen sind leer."}
    ll, grad, hess = cur
    for it in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return {"beta": nan, "se": nan, "loglik": ll, "iterations": it, "ok": False,
                    "message": "Die Information ist singulär (konstantes oder kollineares Merkmal, oder zu wenige Ereignisse)."}
        scale = 1.0
        for _ in range(40):
            new = evaluate(beta + scale * step)
            if new is not None and new[0] >= ll - 1e-12:
                break
            scale *= 0.5
        else:
            break
        beta = beta + scale * step
        converged = abs(new[0] - ll) < tol * (1.0 + abs(ll)) and np.max(np.abs(scale * step)) < 1e-7
        ll, grad, hess = new
        if converged:
            try:
                cov = np.linalg.inv(hess)
            except np.linalg.LinAlgError:
                break
            se = np.sqrt(np.abs(np.diag(cov)))
            if not np.all(np.isfinite(se)) or np.any(se > 50.0) or np.max(np.abs(beta)) > 20.0:
                break
            return {"beta": beta, "se": se, "loglik": ll, "iterations": it, "ok": True, "message": ""}
        if np.max(np.abs(beta)) > 25:
            break
    return {"beta": beta, "se": np.full(p, np.nan), "loglik": ll, "iterations": it, "ok": False,
            "message": "Keine Konvergenz; bei perfekter Trennung der Gruppen (alle Ereignisse in einer Gruppe zuerst) gibt es keinen endlichen Schätzer."}


def _beta_or_nan(fit: dict, j: int = 0) -> float:
    return float(fit["beta"][j]) if fit["ok"] else float("nan")


# ------------------------------------------------------------------------------------------------ Studien (Experiment 2 und 3)


def phase_betas(sample: Sample, fit=cox_fast) -> tuple[list[float], list[str]]:
    """Cox-β̂ für das Maßnahme-Merkmal in den drei Phasen (früh: bis Tag 3, mittel: Tag 3 bis 8 mit Beginn bei 3, spät: nach Tag 8 mit Beginn bei 8) und der Zustand je Anpassung.
    Die Zensierung am Ende einer Phase ist administrativ (feste Zeit, unabhängig vom Container)."""
    t, x = sample.time, sample.x[:, None]
    out, states = [], []
    for a, b in C.PHASES:
        m = t > a
        obs = np.minimum(t[m], b) - a
        ev = t[m] <= b
        r = fit(obs, ev, x[m]) if m.sum() else {"ok": False, "message": "Ohne Ereignis gibt es keinen Schätzer.", "beta": np.array([np.nan])}
        out.append(_beta_or_nan(r))
        states.append(status_of(r))
    return out, states


def follow_up_betas(sample: Sample, fit=cox_fast) -> tuple[list[float], list[str]]:
    """Cox-β̂ bei Nachbeobachtung bis F Tage nach Beginn (Container, die bei F noch da sind, sind dort zensiert); F = ∞: alle bis zur Abholung beobachtet."""
    t, x = sample.time, sample.x[:, None]
    out, states = [], []
    for f in C.FOLLOW_UPS:
        obs = np.minimum(t, f)
        ev = t <= f
        r = fit(obs, ev, x)
        out.append(_beta_or_nan(r))
        states.append(status_of(r))
    return out, states


@dataclass(frozen=True)
class Column:
    """Kennzahlen einer Spalte von Läufen: Läufe, Läufe mit Schätzer, Mittel, Standardfehler des Mittels, Streuung."""

    reps: int
    n_ok: int
    mean: float
    se_of_mean: float
    sd: float


def column(values, reps: int | None = None) -> Column:
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    n_ok = int(ok.size)
    if n_ok == 0:
        return Column(int(v.size if reps is None else reps), 0, float("nan"), float("nan"), float("nan"))
    sd = float(ok.std(ddof=1)) if n_ok > 1 else float("nan")
    return Column(int(v.size if reps is None else reps), n_ok, float(ok.mean()), sd / math.sqrt(n_ok) if n_ok > 1 else float("nan"), sd)


@dataclass(frozen=True)
class FrailtyStudy:
    windows: np.ndarray      # (reps, 3): β̂ je Phase, nan ohne Schätzer
    follow_ups: np.ndarray   # (reps, len(FOLLOW_UPS))
    status_counts: tuple     # ((Zustand, Zahl), …) über alle Anpassungen

    @property
    def reps(self) -> int:
        return int(self.windows.shape[0])


def run_frailty_study(world: FrailtyWorld, reps: int = C.REPS) -> FrailtyStudy:
    """`reps` Datensätze der Welt von Experiment 2 (Lauf r hat den festen Strom run_seed(seed, r)); je Datensatz drei Phasen- und sechs Nachbeobachtungs-Anpassungen."""
    wins = np.full((reps, len(C.PHASES)), np.nan)
    fus = np.full((reps, len(C.FOLLOW_UPS)), np.nan)
    counts: dict[str, int] = {}
    for r in range(reps):
        sm = simulate_frailty(world, r)
        b, s1 = phase_betas(sm)
        f, s2 = follow_up_betas(sm)
        wins[r], fus[r] = b, f
        for s in s1 + s2:
            counts[s] = counts.get(s, 0) + 1
    return FrailtyStudy(wins, fus, tuple(sorted(counts.items())))


@dataclass(frozen=True)
class RandomStudy:
    only_x: np.ndarray       # β̂ für x, Cox nur mit x
    with_z: np.ndarray       # β̂ für x, Cox mit x und z
    z_coef: np.ndarray       # β̂ für z in der Anpassung mit beiden Merkmalen
    status_counts: tuple

    @property
    def reps(self) -> int:
        return int(self.only_x.size)


def run_random_study(world: RandomWorld, reps: int = C.REPS) -> RandomStudy:
    a, b, c = np.full(reps, np.nan), np.full(reps, np.nan), np.full(reps, np.nan)
    counts: dict[str, int] = {}
    for r in range(reps):
        sm = simulate_random(world, r)
        ev = np.ones(sm.time.size, dtype=bool)
        f1 = cox_fast(sm.time, ev, sm.x[:, None])
        f2 = cox_fast(sm.time, ev, np.column_stack([sm.x, sm.z]))
        a[r], b[r], c[r] = _beta_or_nan(f1), _beta_or_nan(f2, 0), _beta_or_nan(f2, 1)
        for s in (status_of(f1), status_of(f2)):
            counts[s] = counts.get(s, 0) + 1
    return RandomStudy(a, b, c, tuple(sorted(counts.items())))


# ------------------------------------------------------------------------------------------------ Experiment 4: Was stattdessen


def survival_se(km: dict, tau: float) -> float:
    """Greenwood-Standardfehler von Ŝ(τ): Ŝ · sqrt(Σ d/(n(n−d))); 0 vor der ersten Ereigniszeit."""
    i = int(np.searchsorted(km["t"], tau, side="right")) - 1
    return 0.0 if i < 0 else float(km["s"][i] * math.sqrt(km["gw"][i]))


@dataclass(frozen=True)
class Stay:
    """Dieselben Daten, anders gelesen: Überlebenswahrscheinlichkeit und mittlere Lebenszeit bis τ je Gruppe, mit Band, dazu die Differenzen und die Wahrheit."""

    tau: float
    n: int
    s0: float
    s0_band: tuple
    s1: float
    s1_band: tuple
    s_diff: float
    s_diff_band: tuple
    rmst0: float
    rmst1: float
    rmst_diff: float
    rmst_se: float
    rmst_band: tuple
    rmst_status: str
    truth_s0: float
    truth_s1: float
    truth_rmst0: float
    truth_rmst1: float
    cox_hr_all: float
    cox_ok: bool
    phase_hr: tuple          # exp(β̂) je Phase, nan ohne Schätzer
    km0: dict
    km1: dict


def analyse_stay(world: FrailtyWorld, tau: float, run: int = 0) -> Stay:
    """Ein Datensatz (Lauf `run` der Welt von Experiment 2), gelesen als Ŝ(τ) und mittlere Lebenszeit bis τ je Gruppe."""
    sm = simulate_frailty(world, run)
    ev = np.ones(sm.time.size, dtype=bool)
    m1 = sm.x == 1
    km0, km1 = K.kaplan_meier(sm.time[~m1], ev[~m1]), K.kaplan_meier(sm.time[m1], ev[m1])
    s0, lo0, hi0 = K.greenwood_band(km0, tau)
    s1, lo1, hi1 = K.greenwood_band(km1, tau)
    se = math.sqrt(survival_se(km0, tau) ** 2 + survival_se(km1, tau) ** 2)
    d = K.survival_at(km1, tau) - K.survival_at(km0, tau)
    rm = K.rmst_difference_test(sm.time, ev, sm.x.astype(int), tau)
    law0, law1 = frailty_laws(world)
    fit = cox_fast(sm.time, ev, sm.x)
    betas, _ = phase_betas(sm)
    return Stay(
        tau=float(tau), n=int(sm.time.size), s0=s0, s0_band=(lo0, hi0), s1=s1, s1_band=(lo1, hi1),
        s_diff=d, s_diff_band=(d - C.Z95 * se, d + C.Z95 * se),
        rmst0=rm["area0"], rmst1=rm["area1"], rmst_diff=rm["diff"], rmst_se=rm["se"],
        rmst_band=(rm["diff"] - C.Z95 * rm["se"], rm["diff"] + C.Z95 * rm["se"]), rmst_status=rm["status"],
        truth_s0=float(law0.surv(tau)), truth_s1=float(law1.surv(tau)), truth_rmst0=law0.rmst(tau), truth_rmst1=law1.rmst(tau),
        cox_hr_all=math.exp(_beta_or_nan(fit)) if fit["ok"] else float("nan"), cox_ok=bool(fit["ok"]),
        phase_hr=tuple(math.exp(b) if math.isfinite(b) else float("nan") for b in betas), km0=km0, km1=km1,
    )


def coverage_of_stay(world: FrailtyWorld, tau: float, reps: int) -> dict:
    """Über `reps` Läufe: in wie vielen enthält das 95-%-Intervall die Wahrheit (Differenz der Überlebenswahrscheinlichkeit, Differenz der mittleren Lebenszeit)?"""
    hits_s = hits_r = 0
    for r in range(reps):
        st = analyse_stay(world, tau, r)
        td_s = st.truth_s1 - st.truth_s0
        td_r = st.truth_rmst1 - st.truth_rmst0
        hits_s += st.s_diff_band[0] <= td_s <= st.s_diff_band[1]
        hits_r += st.rmst_band[0] <= td_r <= st.rmst_band[1]
    return {"reps": reps, "hits_s_diff": int(hits_s), "hits_rmst_diff": int(hits_r)}
