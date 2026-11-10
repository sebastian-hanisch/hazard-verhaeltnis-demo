"""Messskript der Demo: jede Zahl der README steht in `tools/measure_hazard_result.json` (eingefroren, feste Seeds).

Aufruf (Repo-Wurzel):  _venvs/test/Scripts/python.exe -X utf8 tools/measure_hazard.py            schreibt tools/measure_hazard_result.json
                       _venvs/test/Scripts/python.exe -X utf8 tools/measure_hazard.py --print    zeigt die Zahlen, schreibt nichts

Alle Messungen laufen mit dem Code der Demo (SplitMix64, hzv_model); Läufe je Zelle: C.REPS = 200, Lauf r hat den Seed `run_seed(Zufalls-Seed, r)`.
Die Funktionen sind rein, die Tests (tests/test_claims.py) rechnen Teile davon neu. Rechenzeit der ganzen Messung: einige Minuten.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hzv_constants as C  # noqa: E402
import hzv_evaluation as ev  # noqa: E402
import hzv_model as M  # noqa: E402
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld  # noqa: E402

RESULT = Path(__file__).with_name("measure_hazard_result.json")
STANDARD = FrailtyWorld()                 # n = 2000, v = 1, Faktor 2, Zufalls-Seed 12
SEEDS_STABILITY = (1, 2, 3, 4, 5)
TAUS = (2, 5, 10, 20)
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def col(c: M.Column) -> dict:
    return {"n_ok": c.n_ok, "mean": c.mean, "se_of_mean": c.se_of_mean, "sd": c.sd}


def frailty_cell(world: FrailtyWorld, reps: int = C.REPS) -> dict:
    """Experiment 2 bei einer Einstellung: Cox-β̂ je Phase und je Länge der Nachbeobachtung (Mittel, Standardfehler des Mittels, Streuung) und die Grenzwerte."""
    study = M.run_frailty_study(world, reps)
    return {
        "n": world.n, "v": world.v, "factor": world.factor, "seed": world.seed, "reps": study.reps,
        "windows": [col(M.column(study.windows[:, j], study.reps)) for j in range(len(C.PHASES))],
        "follow_ups": [col(M.column(study.follow_ups[:, j], study.reps)) for j in range(len(C.FOLLOW_UPS))],
        "status_counts": dict(study.status_counts),
        "pop_windows": M.population_windows(world), "pop_follow_ups": M.population_follow_ups(world),
        "truth": float(np.log(world.factor)),
    }


def random_cell(world: RandomWorld, reps: int = C.REPS) -> dict:
    """Experiment 3: Cox-β̂ nur mit x, mit x und z, und der Koeffizient von z."""
    study = M.run_random_study(world, reps)
    return {
        "n": world.n, "gamma": world.gamma, "factor": world.factor, "seed": world.seed, "reps": study.reps,
        "only_x": col(M.column(study.only_x, study.reps)), "with_z": col(M.column(study.with_z, study.reps)), "z_coef": col(M.column(study.z_coef, study.reps)),
        "status_counts": dict(study.status_counts), "pop_only_x": M.population_random(world), "truth": float(np.log(world.factor)),
    }


def stay_cell(world: FrailtyWorld, tau: float, reps: int = C.REPS) -> dict:
    """Experiment 4 bei einem τ: Mittel der Differenzen über die Läufe, Wahrheit, Überdeckung des 95-%-Intervalls (Zählung) und das einzelne Hazard-Verhältnis über alles."""
    sd, rd, hits_s, hits_r, hr_all, nofit = [], [], 0, 0, [], 0
    truth_s = truth_r = None
    for r in range(reps):
        st = M.analyse_stay(world, tau, r)
        truth_s, truth_r = st.truth_s1 - st.truth_s0, st.truth_rmst1 - st.truth_rmst0
        sd.append(st.s_diff)
        rd.append(st.rmst_diff)
        hits_s += st.s_diff_band[0] <= truth_s <= st.s_diff_band[1]
        hits_r += st.rmst_band[0] <= truth_r <= st.rmst_band[1]
        if st.cox_ok:
            hr_all.append(st.cox_hr_all)
        else:
            nofit += 1
    sd, rd, hr_all = np.array(sd), np.array(rd), np.array(hr_all)
    return {
        "tau": tau, "n": world.n, "v": world.v, "factor": world.factor, "seed": world.seed, "reps": reps,
        "s_diff_mean": float(sd.mean()), "s_diff_se": float(sd.std(ddof=1) / np.sqrt(reps)), "s_diff_truth": truth_s, "s_diff_hits": int(hits_s),
        "rmst_diff_mean": float(rd.mean()), "rmst_diff_se": float(rd.std(ddof=1) / np.sqrt(reps)), "rmst_diff_truth": truth_r, "rmst_diff_hits": int(hits_r),
        "cox_hr_all_mean": float(hr_all.mean()) if hr_all.size else float("nan"), "cox_no_fit": nofit,
    }


def mechanism_cell(world: FrailtyWorld, times=(2.0, 5.0, 10.0, 20.0), reps: int = C.REPS) -> dict:
    """Mittlere Anfälligkeit der Übrigen je Gruppe (Mittel über die Läufe) gegen die Formel 1/(1 + vλf^x t); dazu das Verhältnis behandelt/Kontrolle gegen HR(t)/f."""
    acc = {t: {"m0": [], "m1": [], "ratio": []} for t in times}
    formula = {}
    for r in range(reps):
        for row in ev.survivor_frailty_rows(world, times, run=r):
            acc[row["t"]]["m0"].append(row["mean0"])
            acc[row["t"]]["m1"].append(row["mean1"])
            acc[row["t"]]["ratio"].append(row["ratio"])
            formula[row["t"]] = (row["formula0"], row["formula1"], row["hr_over_f"])
    out = []
    for t in times:
        a = acc[t]
        out.append({"t": t, "mean0": float(np.mean(a["m0"])), "mean1": float(np.mean(a["m1"])), "ratio": float(np.mean(a["ratio"])),
                    "formula0": formula[t][0], "formula1": formula[t][1], "hr_over_f": formula[t][2]})
    return {"n": world.n, "v": world.v, "factor": world.factor, "seed": world.seed, "reps": reps, "rows": out}


def class_cell(world: ClassWorld) -> dict:
    t_min, hr_min = M.class_hr_minimum(world)
    return {"rate_a": world.rate_a, "rate_b": world.rate_b, "share_b": world.share_b, "factor": world.factor, "t_min": t_min, "hr_min": hr_min,
            "hr_end": float(M.class_hr_curve(world, C.CLASS_HORIZON))}


def measure() -> dict:
    out: dict = {"reps": C.REPS}
    out["by_v"] = [frailty_cell(FrailtyWorld(v=v)) for v in C.V_GRID + (3.0,)]                       # Messung V12 und v = 3 (Reglermaximum)
    out["no_effect"] = [frailty_cell(FrailtyWorld(v=v, factor=1.0)) for v in (1.0, 2.0)]              # ohne Wirkung
    out["stability"] = [frailty_cell(FrailtyWorld(seed=s)) for s in SEEDS_STABILITY]                  # v = 1, andere Zufalls-Seeds
    out["stability_v2"] = [frailty_cell(FrailtyWorld(v=2.0, seed=s)) for s in SEEDS_STABILITY]         # v = 2, andere Zufalls-Seeds
    out["by_n"] = [frailty_cell(FrailtyWorld(n=n)) for n in (400, 1000, 3000)]                        # kleinere und größere Stichproben (Standard n = 2000 steht in by_v)
    out["by_factor"] = [frailty_cell(FrailtyWorld(factor=f)) for f in (1.5, 3.0)]                     # andere Faktoren
    out["random"] = [random_cell(RandomWorld(gamma=g)) for g in GAMMAS]                               # Experiment 3
    out["random_factor1"] = random_cell(RandomWorld(factor=1.0))
    out["stay"] = [stay_cell(STANDARD, tau) for tau in TAUS]                                          # Experiment 4
    out["stay_pooled"] = [stay_cell(FrailtyWorld(seed=s), 10) for s in SEEDS_STABILITY]                # Überdeckung bei τ = 10 über fünf Zufalls-Seeds
    out["mechanism"] = mechanism_cell(STANDARD)                                                       # mittlere Anfälligkeit der Übrigen
    out["classes"] = [class_cell(ClassWorld()), class_cell(ClassWorld(rate_b=1.5))]                   # Experiment 1: Standard und „fast gleiche Klassen“
    return out


def tables(d: dict) -> dict[str, str]:
    """Die Markdown-Tabellen der README aus den eingefrorenen Zahlen (Dezimalkomma, echtes Minuszeichen). tests/test_claims.py prüft, dass jede Tabelle wörtlich in der README steht."""
    from hzv_evaluation import num, pct

    def w(c, j, digits=3):
        return f"{num(c['windows'][j]['mean'], digits)} ± {num(c['windows'][j]['se_of_mean'], digits)}"

    rows = ["| Varianz v | früh (Tag 0 bis 3) | mittel (Tag 3 bis 8) | spät (nach Tag 8) | Grenzwerte früh / mittel / spät |", "|---|---|---|---|---|"]
    for c in d["by_v"]:
        pw = c["pop_windows"]
        rows.append(f"| {num(c['v'], 1)} | {w(c, 0)} | {w(c, 1)} | {w(c, 2)} | {num(pw[0], 3)} / {num(pw[1], 3)} / {num(pw[2], 3)} |")
    t_by_v = "\n".join(rows)

    rows = ["| Zufalls-Seed | früh | mittel | spät |", "|---|---|---|---|"]
    for c in d["stability"]:
        rows.append(f"| {c['seed']} | {num(c['windows'][0]['mean'], 3)} | {num(c['windows'][1]['mean'], 3)} | {num(c['windows'][2]['mean'], 3)} |")
    t_stability = "\n".join(rows)

    rows = ["| n | früh | mittel | spät | Läufe ohne Schätzer in der späten Phase |", "|---|---|---|---|---|"]
    for c in sorted(d["by_n"] + [d["by_v"][2]], key=lambda c: c["n"]):
        rows.append(f"| {c['n']} | {w(c, 0)} | {w(c, 1)} | {w(c, 2)} | {c['reps'] - c['windows'][2]['n_ok']} von {c['reps']} |")
    t_by_n = "\n".join(rows)

    rows = ["| Faktor | Wahrheit ln f | früh | mittel | spät | Grenzwerte früh / mittel / spät |", "|---|---|---|---|---|---|"]
    for c in sorted(d["by_factor"] + [d["by_v"][2]], key=lambda c: c["factor"]):
        pw = c["pop_windows"]
        rows.append(f"| {num(c['factor'], 1)} | {num(c['truth'], 3)} | {w(c, 0)} | {w(c, 1)} | {w(c, 2)} | {num(pw[0], 3)} / {num(pw[1], 3)} / {num(pw[2], 3)} |")
    t_by_factor = "\n".join(rows)

    rows = ["| Faktor | Varianz v | früh | mittel | spät |", "|---|---|---|---|---|"]
    for c in d["no_effect"]:
        rows.append(f"| {num(c['factor'], 1)} | {num(c['v'], 1)} | {w(c, 0)} | {w(c, 1)} | {w(c, 2)} |")
    t_no_effect = "\n".join(rows)

    rows = ["| Zufalls-Seed | früh | mittel | spät |", "|---|---|---|---|"]
    for c in d["stability_v2"]:
        rows.append(f"| {c['seed']} | {num(c['windows'][0]['mean'], 3)} | {num(c['windows'][1]['mean'], 3)} | {num(c['windows'][2]['mean'], 3)} |")
    t_stability_v2 = "\n".join(rows)

    rows = ["| Zufalls-Seed | Anteil noch da: Intervall enthält die Wahrheit | mittlere Verweilzeit: Intervall enthält die Wahrheit |", "|---|---|---|"]
    for c in d["stay_pooled"]:
        rows.append(f"| {c['seed']} | {c['s_diff_hits']} von {c['reps']} | {c['rmst_diff_hits']} von {c['reps']} |")
    t_stay_pooled = "\n".join(rows)

    base = d["by_v"][2]
    labels = [f"bis Tag {f:g}" if f < 1e9 else "bis zum Ende" for f in C.FOLLOW_UPS]
    rows = ["| Nachbeobachtung | Cox β̂ (Mittel ± Standardfehler) | Hazard-Verhältnis exp(β̂) | Grenzwert des Hazard-Verhältnisses |", "|---|---|---|---|"]
    import math
    for lab, c, p in zip(labels, base["follow_ups"], base["pop_follow_ups"]):
        rows.append(f"| {lab} | {num(c['mean'], 3)} ± {num(c['se_of_mean'], 3)} | {num(math.exp(c['mean']), 2)} | {num(math.exp(p), 2)} |")
    t_follow = "\n".join(rows)

    rows = ["| γ | Cox nur mit x (Mittel ± Standardfehler) | Grenzwert nur mit x | Cox mit x und z | Koeffizient von z (Wahrheit γ) |", "|---|---|---|---|---|"]
    for c in d["random"]:
        rows.append(f"| {num(c['gamma'], 1)} | {num(c['only_x']['mean'], 3)} ± {num(c['only_x']['se_of_mean'], 3)} | {num(c['pop_only_x'], 3)} | "
                    f"{num(c['with_z']['mean'], 3)} ± {num(c['with_z']['se_of_mean'], 3)} | {num(c['z_coef']['mean'], 3)} |")
    t_random = "\n".join(rows)

    rows = ["| τ (Tage) | Differenz Anteil noch da (Mittel ± Standardfehler) | Wahrheit | Intervall enthält die Wahrheit | Differenz mittlere Verweilzeit (Tage) | Wahrheit | Intervall enthält die Wahrheit |", "|---|---|---|---|---|---|---|"]
    for c in d["stay"]:
        rows.append(f"| {c['tau']} | {num(100 * c['s_diff_mean'], 2)} ± {num(100 * c['s_diff_se'], 2)} Prozentpunkte | {num(100 * c['s_diff_truth'], 2)} Prozentpunkte | {c['s_diff_hits']} von {c['reps']} | "
                    f"{num(c['rmst_diff_mean'], 3)} ± {num(c['rmst_diff_se'], 3)} | {num(c['rmst_diff_truth'], 3)} | {c['rmst_diff_hits']} von {c['reps']} |")
    t_stay = "\n".join(rows)

    rows = ["| Zeit t (Tage) | mittleres Z der Übrigen, Kontrolle (Formel) | mittleres Z der Übrigen, Maßnahme (Formel) | Verhältnis (Formel HR(t) / f) |", "|---|---|---|---|"]
    for r in d["mechanism"]["rows"]:
        rows.append(f"| {r['t']:g} | {num(r['mean0'], 3)} ({num(r['formula0'], 3)}) | {num(r['mean1'], 3)} ({num(r['formula1'], 3)}) | {num(r['ratio'], 3)} ({num(r['hr_over_f'], 3)}) |")
    t_mech = "\n".join(rows)

    return {"by_v": t_by_v, "stability": t_stability, "stability_v2": t_stability_v2, "stay_pooled": t_stay_pooled, "by_n": t_by_n, "by_factor": t_by_factor, "no_effect": t_no_effect, "follow": t_follow, "random": t_random, "stay": t_stay, "mechanism": t_mech}


if __name__ == "__main__":
    t0 = time.time()
    result = measure()
    text = json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True)
    if "--print" in sys.argv:
        print(text)
    else:
        RESULT.write_text(text + "\n", encoding="utf-8")
        print(f"geschrieben: {RESULT} ({time.time() - t0:.0f} s)")
