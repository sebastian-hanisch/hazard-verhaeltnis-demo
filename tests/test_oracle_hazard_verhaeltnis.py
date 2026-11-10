"""Orakel-Tests: andere Rechenwege als die der Demo (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

Geprüft werden die vier Kerne, auf denen die Zahlen der Demo stehen:
  1. Die marginale Rate einer Zwei-Klassen-Mischung (`hzv_model.class_rows`): gegen die numerische Ableitung der Mischungs-Überlebensfunktion und gegen eine
     numerische Optimierung des Tiefpunkts (scipy).
  2. Die geschlossene Formel der Gamma-Anfälligkeit (`GammaLaw`, `gamma_hr`, `rmst`): gegen numerische Integration über die Gamma-Dichte (scipy.integrate.quad
     mit scipy.stats.gamma). Dasselbe für die Zufallszahlen (Kolmogorow–Smirnow-Test gegen die Verteilungsfunktion und die Laplace-Transformierte).
  3. Der Grenzwert des Cox-Schätzers (`population_beta`): gegen eine adaptive Quadratur mit Wurzelsuche (brentq), gegen eine große Simulation und gegen eine
     Gauß–Hermite-Mischung für das zweite Merkmal.
  4. Die schnelle Cox-Anpassung (`cox_fast`): gegen den Kern (`hzv_core.cox_fit`) und gegen eine Nullstellensuche auf dem Score in reinen Schleifen (ohne Newton);
     Kaplan–Meier, Greenwood und mittlere Lebenszeit gegen Aufzählung mit Brüchen.
Zufall nur zum Erzeugen kleiner Instanzen (feste Seeds über SplitMix64).
"""

import math
from fractions import Fraction as F

import numpy as np
import pytest

scipy_integrate = pytest.importorskip("scipy.integrate")
scipy_stats = pytest.importorskip("scipy.stats")
scipy_optimize = pytest.importorskip("scipy.optimize")

import hzv_constants as C  # noqa: E402
import hzv_core as K  # noqa: E402
import hzv_model as M  # noqa: E402
import hzv_scenario as S  # noqa: E402
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld  # noqa: E402

# ------------------------------------------------------------------------------------------------ 1. Zwei-Klassen-Rate


def _mix_survival(rates, weights, t):
    return sum(w * math.exp(-r * t) for r, w in zip(rates, weights))


def _random_class_worlds(count=12):
    rng = K.SplitMix64(7)
    out = []
    for _ in range(count):
        u = rng.uniform(4)
        out.append(ClassWorld(rate_a=0.2 + 0.8 * u[0], rate_b=1.5 + 8.5 * u[1], share_b=0.1 + 0.8 * u[2], factor=1.0 + 3.0 * u[3]))
    return out


def test_marginale_rate_gleicht_der_numerischen_ableitung_der_mischung():
    h = 1e-6
    for world in _random_class_worlds():
        for group, scale in (("h_control", 1.0), ("h_treated", world.factor)):
            rates = (world.rate_a * scale, world.rate_b * scale)
            weights = (1 - world.share_b, world.share_b)
            for t in (0.1, 0.5, 1.0, 2.0, 4.0):
                s = _mix_survival(rates, weights, t)
                ref = -(_mix_survival(rates, weights, t + h) - _mix_survival(rates, weights, t - h)) / (2 * h * s)
                got = [r for r in M.class_rows(world, [t])][0][group]
                assert got == pytest.approx(ref, rel=1e-6), (world, group, t)


def test_tiefpunkt_gleicht_der_numerischen_optimierung():
    for world in _random_class_worlds(8):
        if world.factor < 1.2:
            continue
        t_grid, hr_grid = M.class_hr_minimum(world)
        res = scipy_optimize.minimize_scalar(lambda t: float(M.class_hr_curve(world, t)), bounds=(0.0, C.CLASS_HORIZON), method="bounded", options={"xatol": 1e-9})
        assert hr_grid == pytest.approx(res.fun, abs=2e-4), world
        assert float(M.class_hr_curve(world, res.x)) <= hr_grid + 1e-9




# ------------------------------------------------------------------------------------------------ 2. Gamma-Anfälligkeit


def _gamma_pdf(v):
    return lambda z: scipy_stats.gamma.pdf(z, a=1.0 / v, scale=v)


def _mixture_integral(v, fun):
    val, _ = scipy_integrate.quad(lambda z: _gamma_pdf(v)(z) * fun(z), 0.0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-12)
    return val


def test_gamma_ueberlebensfunktion_und_rate_gleichen_der_integration_ueber_die_gamma_dichte():
    for v in (0.25, 1.0, 2.0):
        for rate in (0.1, 0.2):
            law = M.GammaLaw(rate, v)
            for t in (0.0, 1.0, 5.0, 20.0):
                s = _mixture_integral(v, lambda z: math.exp(-rate * z * t))
                num = rate * _mixture_integral(v, lambda z: z * math.exp(-rate * z * t))
                assert float(law.surv(t)) == pytest.approx(s, rel=1e-8), (v, rate, t)
                assert float(law.hazard(t)) == pytest.approx(num / s, rel=1e-7), (v, rate, t)


def test_geschlossene_formel_des_verhaeltnisses_gegen_die_zahlen_der_integration():
    for v in (0.5, 1.0, 2.0):
        for f in (1.5, 2.0, 3.0):
            for t in (0.0, 2.0, 8.0, 30.0):
                haz = []
                for rate in (0.1, 0.1 * f):
                    s = _mixture_integral(v, lambda z: math.exp(-rate * z * t))
                    haz.append(rate * _mixture_integral(v, lambda z: z * math.exp(-rate * z * t)) / s)
                assert float(M.gamma_hr(v, 0.1, f, t)) == pytest.approx(haz[1] / haz[0], rel=1e-7), (v, f, t)


def test_mittlere_lebenszeit_geschlossen_gleicht_der_doppelten_integration():
    """∫₀^τ S(t) dt = E_Z[(1 − e^(−rate Z τ)) / (rate Z)] (Vertauschung der Integrale): ein anderer Weg als die geschlossene Stammfunktion."""
    for v in (0.5, 1.0, 2.0):
        for rate in (0.1, 0.2):
            for tau in (2.0, 10.0, 40.0):
                ref = _mixture_integral(v, lambda z: -math.expm1(-rate * z * tau) / (rate * z) if z > 0 else tau)
                assert M.GammaLaw(rate, v).rmst(tau) == pytest.approx(ref, rel=1e-8), (v, rate, tau)


def test_gamma_zufallszahlen_gleichen_der_verteilungsfunktion():
    for v in (0.25, 1.0, 3.0):
        z = S.frailty_values(K.SplitMix64(11), 20000, v)
        res = scipy_stats.kstest(z, scipy_stats.gamma(a=1.0 / v, scale=v).cdf)
        assert res.pvalue > 1e-3, (v, res.pvalue)
        for s in (0.5, 1.0, 3.0):   # Laplace-Transformierte (1 + vs)^(−1/v) = E exp(−sZ)
            est = float(np.mean(np.exp(-s * z)))
            assert est == pytest.approx((1 + v * s) ** (-1 / v), abs=5 * float(np.std(np.exp(-s * z))) / math.sqrt(z.size)), (v, s)


# ------------------------------------------------------------------------------------------------ 3. Grenzwert des Cox-Schätzers


def _ref_population_beta(law0, law1, a, b):
    """Nullstelle des erwarteten Scores mit adaptiver Quadratur (über ln t) und brentq: kein gemeinsamer Code mit `population_beta` (Gauß–Legendre, Bisektion)."""
    lo = math.log(a) if a > 0 else math.log(1e-12)
    hi = math.log(b) if math.isfinite(b) else math.log(1e30)

    def score(beta):
        e = math.exp(beta)

        def g(u):
            t = math.exp(u)
            s0, s1, f0, f1 = (float(x(t)) for x in (law0.surv, law1.surv, law0.dens, law1.dens))
            den = s0 + s1 * e
            m = s1 * e / den if den > 0 else 0.0
            return (f1 - (f0 + f1) * m) * t

        val, _ = scipy_integrate.quad(g, lo, hi, limit=3000, epsabs=1e-14, epsrel=1e-12)
        return val

    return scipy_optimize.brentq(score, -6.0, 6.0, xtol=1e-12)


@pytest.mark.parametrize("v", [0.5, 1.0, 2.0])
def test_grenzwert_der_phasen_gleicht_quadratur_mit_wurzelsuche(v):
    world = FrailtyWorld(v=v)
    law0, law1 = M.frailty_laws(world)
    for (a, b), got in zip(C.PHASES, M.population_windows(world)):
        assert got == pytest.approx(_ref_population_beta(law0, law1, a, b), abs=1e-7), (v, a, b)
    for f, got in zip(C.FOLLOW_UPS, M.population_follow_ups(world)):
        assert got == pytest.approx(_ref_population_beta(law0, law1, 0.0, f), abs=1e-7), (v, f)


def test_grenzwert_gleicht_einer_grossen_simulation():
    """Ein einziger Datensatz mit 300 000 Containern (v = 1, Faktor 2): β̂ je Phase liegt in 0,03 um den Grenzwert (Standardfehler etwa 0,006 je Phase)."""
    world = FrailtyWorld(n=300_000, v=1.0, seed=5)
    sample = S.simulate_frailty(world, 0)
    betas, states = M.phase_betas(sample)
    assert states == ["ok", "ok", "ok"]
    for got, ref in zip(betas, M.population_windows(world)):
        assert abs(got - ref) < 0.03
    fus, _ = M.follow_up_betas(sample)
    for got, ref in zip(fus, M.population_follow_ups(world)):
        assert abs(got - ref) < 0.03


def test_grenzwert_des_zweiten_merkmals_gleicht_gauss_hermite_und_einer_grossen_simulation():
    world = RandomWorld(gamma=1.0)
    got = M.population_random(world)
    nodes, weights = np.polynomial.hermite.hermgauss(200)                         # ∫ φ(z) g(z) dz = Σ w_i g(√2 ξ_i) / √π
    base = C.BASE_RATE * np.exp(world.gamma * math.sqrt(2.0) * nodes)
    w = weights / math.sqrt(math.pi)
    ref = M.population_beta(M.Mixture(tuple(base), tuple(w)), M.Mixture(tuple(base * world.factor), tuple(w)))
    assert got == pytest.approx(ref, abs=2e-3)
    big = S.simulate_random(RandomWorld(n=300_000, gamma=1.0, seed=3), 0)
    fit = M.cox_fast(big.time, np.ones(big.time.size, bool), big.x)
    assert fit["ok"] and abs(fit["beta"][0] - got) < 0.02
    both = M.cox_fast(big.time, np.ones(big.time.size, bool), np.column_stack([big.x, big.z]))
    assert both["ok"] and abs(both["beta"][0] - math.log(2.0)) < 0.02 and abs(both["beta"][1] - 1.0) < 0.02


def test_mittlere_anfaelligkeit_der_uebrigen_gleicht_der_formel():
    """E[Z | T > t, x] = 1/(1 + vλ f^x t): aus einem großen Datensatz gezählt (nur die Simulation kennt Z)."""
    world = FrailtyWorld(n=300_000, v=2.0, seed=9)
    sm = S.simulate_frailty(world, 0)
    for t in (0.0, 2.0, 5.0, 10.0):
        for x in (0, 1):
            alive = (sm.time > t) & (sm.x == x)
            assert float(sm.z[alive].mean()) == pytest.approx(1.0 / (1.0 + 2.0 * 0.1 * 2.0**x * t), abs=0.02), (t, x)


# ------------------------------------------------------------------------------------------------ 4. Cox-Anpassung und Kaplan–Meier


def _instance(seed, n=None, p=1):
    rng = K.SplitMix64(seed)
    n = n or 8 + seed % 40
    x = rng.normal(n * p).reshape(n, p) if seed % 2 else np.floor(rng.uniform(n * p) * 3).reshape(n, p)
    t = np.maximum(np.round(rng.exponential(n) * 3 * np.exp(-0.5 * x[:, 0]), 0), 1.0)       # ganzzahlige Zeiten: viele Gleichstände
    e = rng.uniform(n) > 0.3
    return t, e, x


def _ref_score(beta, t, e, x):
    """Score in reinen Schleifen: je Container mit Ereignis x_i minus gewichtetes Mittel der Risikomenge."""
    total = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        risk = [j for j in range(len(t)) if t[j] >= t[i]]
        w = [math.exp(beta * x[j]) for j in risk]
        total += x[i] - sum(wj * x[j] for wj, j in zip(w, risk)) / sum(w)
    return total


def test_schnelle_cox_anpassung_gleicht_dem_kern():
    compared = 0
    for seed in range(40):
        t, e, x = _instance(seed, p=1 + seed % 2)
        fast, slow = M.cox_fast(t, e, x), K.cox_fit(t, e, x)
        assert fast["ok"] == slow["ok"], seed
        if slow["ok"]:
            assert np.allclose(fast["beta"], slow["beta"], atol=1e-9) and np.allclose(fast["se"], slow["se"], atol=1e-9), seed
            assert fast["loglik"] == pytest.approx(slow["loglik"], abs=1e-9)
            compared += 1
    assert compared >= 25


def test_schnelle_cox_anpassung_gleicht_der_nullstellensuche_ohne_newton():
    compared = 0
    for seed in range(40):
        t, e, x = _instance(seed)
        fit = M.cox_fast(t, e, x)
        if not fit["ok"]:
            continue
        root = scipy_optimize.brentq(lambda b: _ref_score(b, t, e, x[:, 0]), -15.0, 15.0, xtol=1e-12)
        assert fit["beta"][0] == pytest.approx(root, abs=1e-7), seed
        compared += 1
    assert compared >= 20


def _ref_km(times, events):
    """Produkt-Grenzwert-Schätzer mit Brüchen: [(t, n, d, S, Greenwood-Summe)]."""
    s, gw, out = F(1), F(0), []
    for t in sorted({x for x, e in zip(times, events) if e}):
        n = sum(1 for x in times if x >= t)
        d = sum(1 for x, e in zip(times, events) if x == t and e)
        s *= F(n - d, n)
        if n > d:
            gw += F(d, n * (n - d))
        out.append((t, n, d, s, gw))
    return out


def test_kaplan_meier_greenwood_und_flaeche_gleichen_der_aufzaehlung_mit_bruechen():
    for seed in range(25):
        rng = K.SplitMix64(100 + seed)
        n = 6 + seed % 20
        times = [int(v) for v in np.maximum(np.round(rng.exponential(n) * 4), 1)]
        events = [bool(v) for v in (rng.uniform(n) > 0.25)]
        km = K.kaplan_meier(times, events)
        ref = _ref_km(times, events)
        assert km["t"].tolist() == [r[0] for r in ref]
        assert np.allclose(km["s"], [float(r[3]) for r in ref], atol=1e-14) and np.allclose(km["gw"], [float(r[4]) for r in ref], atol=1e-14)
        for tau in (2.0, 5.0, 11.0):
            area = F(0)
            grid = [F(0)] + [F(r[0]) for r in ref if r[0] < tau] + [F(tau).limit_denominator(1000)]
            levels = [F(1)] + [r[3] for r in ref if r[0] < tau]
            for lv, a, b in zip(levels, grid[:-1], grid[1:]):
                area += lv * (b - a)
            assert K.restricted_mean(km, tau) == pytest.approx(float(area), abs=1e-12), (seed, tau)
