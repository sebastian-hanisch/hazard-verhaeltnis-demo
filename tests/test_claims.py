"""Jede Zahl der README steht in einer Messung (`tools/measure_hazard_result.json`, eingefroren, feste Zufalls-Seeds) oder in einem Test.

Dieser Test prüft dreierlei:
  1. Die Tabellen der README stehen wörtlich so in der README, wie `tools/measure_hazard.tables()` sie aus der eingefrorenen JSON-Datei bildet.
  2. Die Prosa-Zahlen der README (Mittel, Zählungen, Spannen) folgen aus denselben Zahlen.
  3. Ausgewählte Zellen werden mit dem Code der Demo neu gerechnet und stimmen mit der eingefrorenen Datei überein (Band 1e-6; Zählungen exakt).
Die übrigen Zellen sind nur eingefroren (die README sagt das in der Tabelle „Zahl → Test“)."""

import json
import math
from pathlib import Path

import pytest

import hzv_constants as C
import hzv_evaluation as ev
import hzv_model as M
import measure_hazard as MH
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FLAT = " ".join(README.split())   # Zeilenumbrüche der Absätze eingeebnet (für Sätze, die über mehrere Zeilen laufen)
DATA = json.loads((ROOT / "tools" / "measure_hazard_result.json").read_text(encoding="utf-8"))
TOL = 1e-6


def _by_v(v):
    return next(c for c in DATA["by_v"] if c["v"] == v)


def _dev(cell, j):
    w = cell["windows"][j]
    return abs(w["mean"] - cell["pop_windows"][j]) / w["se_of_mean"]


def test_alle_tabellen_der_readme_stehen_woertlich_in_der_readme():
    for name, table in MH.tables(DATA).items():
        assert table in README, name


def test_annahmen_text_der_app_steht_in_der_readme():
    assert C.ASSUMPTIONS_MARKDOWN.strip() in README


def test_messaufbau_der_standardwelt():
    c = _by_v(1.0)
    assert (c["n"], c["v"], c["factor"], c["seed"], c["reps"]) == (2000, 1.0, 2.0, 12, 200) and DATA["reps"] == 200 == C.REPS
    assert all(c["status_counts"] == {"ok": 1800} for c in DATA["by_v"])        # 9 Anpassungen mal 200 Läufe, keine ohne Schätzer
    assert "Lauf r hat den Seed 12 · 1 000 003 + r" in FLAT and "200 Läufe" in FLAT


def test_readme_zahlen_der_standardwelt():
    c = _by_v(1.0)
    got = [ev.num(w["mean"], 3) for w in c["windows"]]
    assert got == ["0,592", "0,400", "0,169"]
    assert [ev.num(math.exp(w["mean"]), 2) for w in c["windows"]] == ["1,81", "1,49", "1,18"]
    assert "β̂ = 0,592 (Hazard-Verhältnis 1,81)" in FLAT and "β̂ = 0,400 (1,49)" in FLAT and "β̂ = 0,169 (1,18)" in FLAT
    assert ev.num(math.log(2.0), 3) == "0,693" and "ln 2 = 0,693" in FLAT
    assert [ev.num(w["mean"], 3) for w in _by_v(0.0)["windows"]] == ["0,703", "0,684", "0,690"]
    assert "(0,703, 0,684 und 0,690)" in FLAT
    assert [ev.num(w["mean"], 3) for w in _by_v(2.0)["windows"]] == ["0,541", "0,297", "0,075"] and "v = 2 auf 0,541, 0,297 und 0,075" in FLAT


def test_grenzwert_gegen_laeufe_gezaehlt_15_von_15_innerhalb_von_drei_standardfehlern():
    cells = [_dev(c, j) for c in DATA["by_v"] for j in range(3)]
    assert len(cells) == 15 and sum(d <= 3 for d in cells) == 15
    assert ev.num(max(cells), 1) == "2,7"
    assert "in 15 von 15 Zellen (5 Varianzen mal 3 Phasen) innerhalb von 3 Standardfehlern des Grenzwerts" in FLAT
    assert "die größte Abweichung beträgt 2,7 Standardfehler (v = 2, Phase früh)" in FLAT
    # die Zellen am Rand der Spanne stehen in der Stabilitätsmessung unten
    assert max(cells) == _dev(_by_v(2.0), 0)


def test_grenzwerte_der_tabelle_sind_die_neu_gerechneten_werte():
    for c in DATA["by_v"]:
        world = FrailtyWorld(v=c["v"])
        assert M.population_windows(world) == pytest.approx(c["pop_windows"], abs=TOL)
        assert M.population_follow_ups(world) == pytest.approx(c["pop_follow_ups"], abs=TOL)


def test_stabilitaet_ueber_zufalls_seeds():
    late = [c["windows"][2]["mean"] for c in DATA["stability"]]
    assert [c["seed"] for c in DATA["stability"]] == [1, 2, 3, 4, 5] and all(c["reps"] == 200 and c["n"] == 2000 and c["v"] == 1.0 for c in DATA["stability"])
    assert (ev.num(min(late), 3), ev.num(max(late), 3), ev.num(sum(late) / 5, 3)) == ("0,158", "0,171", "0,164")
    early2 = [c["windows"][0]["mean"] for c in DATA["stability_v2"]]
    assert [c["seed"] for c in DATA["stability_v2"]] == [1, 2, 3, 4, 5] and all(c["v"] == 2.0 for c in DATA["stability_v2"])
    assert (ev.num(min(early2), 3), ev.num(max(early2), 3), ev.num(sum(early2) / 5, 3)) == ("0,519", "0,538", "0,528")
    assert "zwischen 0,158 und 0,171 (Mittel 0,164; Grenzwert 0,164; Zufalls-Seed 12 aus Befund 1: 0,169)" in FLAT
    assert "zwischen 0,519 und 0,538 (Mittel 0,528; Grenzwert 0,525; Zufalls-Seed 12 aus Befund 1: 0,541)" in FLAT


def test_andere_stichprobengroessen_und_faktoren():
    by_n = {c["n"]: c for c in DATA["by_n"]}
    assert set(by_n) == {400, 1000, 3000}
    assert all(c["windows"][2]["n_ok"] == 200 for c in DATA["by_n"])        # keine Läufe ohne Schätzer in der späten Phase
    sd = lambda n, j: by_n[n]["windows"][j]["sd"]  # noqa: E731
    assert sd(400, 2) > sd(1000, 2) > sd(3000, 2)                            # kleineres n: breitere Streuung der Schätzwerte (Hilfetext des Schnellstarts)
    by_f = {c["factor"]: c for c in DATA["by_factor"]}
    assert set(by_f) == {1.5, 3.0}
    for c in DATA["by_factor"] + [_by_v(1.0)]:
        w = [x["mean"] for x in c["windows"]]
        assert w[0] > w[1] > w[2] > 0 and w[0] < c["truth"]                  # fällt von Phase zu Phase und bleibt unter der Wahrheit je Container


def test_ohne_wirkung_kein_drift_gezaehlt():
    assert [c["factor"] for c in DATA["no_effect"]] == [1.0, 1.0]
    cells = [abs(c["windows"][j]["mean"]) / c["windows"][j]["se_of_mean"] for c in DATA["no_effect"] for j in range(3)]
    assert len(cells) == 6 and sum(d <= 3 for d in cells) == 6 and ev.num(max(cells), 1) == "2,8"
    assert "in 6 von 6 Zellen (zwei Varianzen mal 3 Phasen) innerhalb von 3 Standardfehlern von 0" in FLAT and "die größte Abweichung beträgt 2,8 Standardfehler (v = 2, Phase früh)" in FLAT
    for c in DATA["no_effect"]:
        assert M.population_windows(FrailtyWorld(v=c["v"], factor=1.0)) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_mechanismus_mittlere_anfaelligkeit_der_uebrigen():
    rows = DATA["mechanism"]["rows"]
    assert [r["t"] for r in rows] == [2.0, 5.0, 10.0, 20.0]
    assert max(abs(r["mean0"] - r["formula0"]) for r in rows) < 0.0035 and max(abs(r["mean1"] - r["formula1"]) for r in rows) < 0.0025
    assert max(abs(r["ratio"] - r["hr_over_f"]) for r in rows) < 0.0045 and all(r["ratio"] < 1.0 for r in rows)
    assert "höchstens 0,003 (Kontrolle), 0,002 (Maßnahme) und 0,004 (Verhältnis)" in FLAT
    for r in rows:   # die Formelspalte folgt aus HR(t)/f = (1 + vλt)/(1 + f vλt)
        assert r["hr_over_f"] == pytest.approx((1 + 0.1 * r["t"]) / (1 + 2 * 0.1 * r["t"]), abs=1e-12)


def test_nachbeobachtung_haengt_das_einzelne_verhaeltnis_an_der_laenge():
    c = _by_v(1.0)
    hr = [math.exp(x["mean"]) for x in c["follow_ups"]]
    assert all(a > b for a, b in zip(hr, hr[1:])) and [ev.num(x, 2) for x in hr] == ["1,86", "1,73", "1,63", "1,55", "1,48", "1,42"]
    devs = [abs(x["mean"] - p) / x["se_of_mean"] for x, p in zip(c["follow_ups"], c["pop_follow_ups"])]
    assert ev.num(max(devs), 1) == "0,6" and "größter Abstand 0,6 Standardfehler" in FLAT
    assert "fällt von 1,86 (Nachbeobachtung bis Tag 2) auf 1,42 (bis zur Abholung aller Container)" in FLAT


def test_zweites_merkmal_zaehlungen_und_prosa():
    r = DATA["random"]
    assert [c["gamma"] for c in r] == list(MH.GAMMAS)
    xs = [c["only_x"]["mean"] for c in r]
    assert xs == sorted(xs, reverse=True) and [ev.num(x, 3) for x in xs] == ["0,695", "0,583", "0,438", "0,338", "0,272"]
    zs = [c["with_z"]["mean"] for c in r]
    assert (ev.num(min(zs), 3), ev.num(max(zs), 3)) == ("0,694", "0,695")
    assert all(abs(c["z_coef"]["mean"] - c["gamma"]) < 4 * c["z_coef"]["se_of_mean"] + 0.01 for c in r)
    gaps = [abs(c["only_x"]["mean"] - c["pop_only_x"]) / c["only_x"]["se_of_mean"] for c in r]
    assert ev.num(max(gaps), 1) == "0,5" and "größter Abstand 0,5 Standardfehler" in FLAT
    f1 = DATA["random_factor1"]
    assert "Cox nur mit x −0,001 ± 0,003, mit x und z −0,002 ± 0,003" in FLAT
    assert abs(f1["only_x"]["mean"]) < 3 * f1["only_x"]["se_of_mean"] and abs(f1["with_z"]["mean"]) < 3 * f1["with_z"]["se_of_mean"]


def test_experiment_4_differenzen_und_ueberdeckung_gezaehlt():
    st = DATA["stay"]
    assert [c["tau"] for c in st] == [2, 5, 10, 20]
    assert all(abs(c["s_diff_mean"] - c["s_diff_truth"]) < 1.0 * c["s_diff_se"] and abs(c["rmst_diff_mean"] - c["rmst_diff_truth"]) < 1.0 * c["rmst_diff_se"] for c in st)
    assert [c["s_diff_hits"] for c in st] == [182, 184, 187, 189] and [c["rmst_diff_hits"] for c in st] == [187, 185, 182, 185]
    pooled = DATA["stay_pooled"]
    assert [c["seed"] for c in pooled] == [1, 2, 3, 4, 5] and sum(c["reps"] for c in pooled) == 1000
    assert (sum(c["s_diff_hits"] for c in pooled), sum(c["rmst_diff_hits"] for c in pooled)) == (957, 953)
    assert "in 957 von 1000 bzw. 953 von 1000 Läufen (95,7 % und 95,3 %)" in FLAT
    assert "182 bis 189 von 200 Läufen (Anteil) bzw. 182 bis 187 von 200 (mittlere Verweilzeit)" in FLAT
    assert ev.num(st[0]["cox_hr_all_mean"], 2) == "1,42" and "ist im Mittel der Läufe 1,42 (je Container: 2)" in FLAT
    assert all(c["cox_no_fit"] == 0 for c in st)


def test_klassen_tiefpunkt_der_presets():
    c0, c1 = DATA["classes"]
    assert (ev.num(c0["hr_min"], 2), ev.num(c0["t_min"], 3), ev.num(c0["hr_end"], 2)) == ("1,47", "0,445", "2,00")
    assert (ev.num(c1["hr_min"], 2), ev.num(c1["t_min"], 3), ev.num(c1["hr_end"], 2)) == ("1,87", "2,248", "1,96")
    assert "Tiefpunkt 1,47 bei Tag 0,445" in FLAT and "Tiefpunkt 1,87 bei Tag 2,248, bei Tag 6 1,96" in FLAT
    assert "liegt bei 1,87 (im Standardbeispiel mit den Raten 1 und 4 bei 1,47)" in C.PRESET_HELP["Zwei fast gleiche Klassen"]    # Hilfetext des Schnellstarts


def test_grenzen_spanne_der_spaeten_phase():
    lo, hi = _by_v(0.5), _by_v(3.0)
    assert (ev.num(math.exp(lo["windows"][2]["mean"]), 2), ev.num(math.exp(hi["windows"][2]["mean"]), 2)) == ("1,35", "1,04")
    assert "zwischen 1,35 und 1,04 (Faktor je Container 2)" in FLAT


# ------------------------------------------------------------------------------------------------ Neuberechnung ausgewählter Zellen (Band 1e-6, Zählungen exakt)


def test_neuberechnung_standardzelle_experiment_2():
    got = MH.frailty_cell(FrailtyWorld())
    ref = _by_v(1.0)
    for j in range(3):
        for key in ("mean", "se_of_mean", "sd"):
            assert got["windows"][j][key] == pytest.approx(ref["windows"][j][key], abs=TOL)
    for j in range(len(C.FOLLOW_UPS)):
        assert got["follow_ups"][j]["mean"] == pytest.approx(ref["follow_ups"][j]["mean"], abs=TOL)
    assert got["status_counts"] == ref["status_counts"] and got["pop_windows"] == pytest.approx(ref["pop_windows"], abs=TOL)


def test_neuberechnung_zweites_merkmal_gamma_eins():
    got = MH.random_cell(RandomWorld(gamma=1.0))
    ref = next(c for c in DATA["random"] if c["gamma"] == 1.0)
    for key in ("only_x", "with_z", "z_coef"):
        assert got[key]["mean"] == pytest.approx(ref[key]["mean"], abs=TOL) and got[key]["se_of_mean"] == pytest.approx(ref[key]["se_of_mean"], abs=TOL)
    assert got["pop_only_x"] == pytest.approx(ref["pop_only_x"], abs=TOL)


def test_neuberechnung_experiment_4_bei_tau_zehn():
    got = MH.stay_cell(FrailtyWorld(), 10)
    ref = next(c for c in DATA["stay"] if c["tau"] == 10)
    assert (got["s_diff_hits"], got["rmst_diff_hits"], got["cox_no_fit"]) == (ref["s_diff_hits"], ref["rmst_diff_hits"], ref["cox_no_fit"])
    for key in ("s_diff_mean", "s_diff_se", "s_diff_truth", "rmst_diff_mean", "rmst_diff_se", "rmst_diff_truth", "cox_hr_all_mean"):
        assert got[key] == pytest.approx(ref[key], abs=TOL), key


def test_neuberechnung_klassen_und_mechanismus():
    for got, ref in zip((MH.class_cell(ClassWorld()), MH.class_cell(ClassWorld(rate_b=1.5))), DATA["classes"]):
        for key in ("t_min", "hr_min", "hr_end"):
            assert got[key] == pytest.approx(ref[key], abs=TOL)
    mech = MH.mechanism_cell(FrailtyWorld(), times=(5.0,), reps=40)       # weniger Läufe: nur die Formelspalten und die Größenordnung
    row = mech["rows"][0]
    assert row["formula0"] == pytest.approx(next(r for r in DATA["mechanism"]["rows"] if r["t"] == 5.0)["formula0"], abs=TOL)
    assert abs(row["mean0"] - row["formula0"]) < 0.02 and abs(row["ratio"] - row["hr_over_f"]) < 0.03


# ------------------------------------------------------------------------------------------------ Formulierungen


def test_kein_satz_das_hazard_verhaeltnis_ist_falsch():
    """Plan-Entscheidung Z3: nur „in dieser Annahme verändert es sich so“, kein Urteil über das Hazard-Verhältnis."""
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (FLAT, app):
        for banned in ("ist falsch", "sind falsch", "widerlegt", "Saatwert", "Schwanz", "läuft lokal"):
            assert banned not in text, banned
        assert "Hazard-Verhältnis" in text and "Streit" in text


def test_lesestufen_stehen_in_readme_und_app():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (FLAT, app):
        for needle in ("Hernán (2010)", "Abrahamowicz u. a. (2025)", "Ying und Xu", "Royston und Parmar (2013)", "Volltext über Abruf", "Volltext selbst gelesen"):
            assert needle in text, needle
    assert FLAT.count("Volltext über Abruf") >= 4 and "nicht wörtlich bei Hernán belegt" in FLAT and "nicht wörtlich bei Hernán belegt" in app
    assert "eine „GMDS-Auswertung von 28 Studien“ ist nicht belegt" in FLAT


def test_die_nachbarn_sind_nur_verlinkt_ohne_ihre_zahlen():
    for repo in ("cox-regression-demo", "proportionalitaet-demo", "log-rank-demo", "verweilzeit-demo"):
        assert f"https://github.com/sebastian-hanisch/{repo})" in FLAT
