"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und der Annahmen-Text (App und README)."""

# --- Welt ---------------------------------------------------------------------------------------------------------------------------------------
# Container-Verweilzeit: jeder Container hat eine eigene Abholrate. Kontrolle: BASE_RATE · Z (Z = unbeobachtete Anfälligkeit, Mittel 1);
# behandelt: mit dem Faktor FACTOR multipliziert, für jeden Container derselbe.
BASE_RATE = 0.1                            # λ, Abholrate je Tag bei Anfälligkeit 1 (Mittel der Verweilzeit ohne Streuung: 10 Tage)

FACTOR_MIN, FACTOR_MAX, DEFAULT_FACTOR, FACTOR_STEP = 1.0, 4.0, 2.0, 0.25
N_MIN, N_MAX, DEFAULT_N, N_STEP = 400, 3000, 2000, 100   # Komplexität am Maximum: je Cox-Anpassung O(n log n) je Newton-Schritt, mal 9 Anpassungen und REPS Läufe
SEED_MAX, DEFAULT_SEED = 999999, 12

# Experiment 1: zwei Klassen
RATE_A_MIN, RATE_A_MAX, DEFAULT_RATE_A, RATE_A_STEP = 0.2, 1.0, 1.0, 0.1   # Klasse A ist die langsamere ...
RATE_B_MIN, RATE_B_MAX, DEFAULT_RATE_B, RATE_B_STEP = 1.5, 10.0, 4.0, 0.5   # ... und B die schnellere (die Bereiche überlappen nicht)
SHARE_B_MIN, SHARE_B_MAX, DEFAULT_SHARE_B, SHARE_B_STEP = 0.1, 0.9, 0.5, 0.1
PROBE_MIN, PROBE_MAX, DEFAULT_PROBE, PROBE_STEP = 0.0, 6.0, 0.5, 0.25     # Zeit t zum Ausprobieren (Tage)
CLASS_TIMES = (0.0, 0.5, 1.0, 2.0, 3.0)    # Zeilen der Handtabelle
CLASS_HORIZON = 6.0                        # Zeitachse der Klassen-Diagramme (Tage)

# Experiment 2: Streuung der Anfälligkeit
V_MIN, V_MAX, DEFAULT_V, V_STEP = 0.0, 3.0, 1.0, 0.25                     # Varianz der Anfälligkeit (Gamma, Mittel 1)
V_GRID = (0.0, 0.5, 1.0, 2.0)              # die Zeilen des Messrasters (dazu v = 3, das Reglermaximum)
PHASES = ((0.0, 3.0), (3.0, 8.0), (8.0, float("inf")))   # früh, mittel, spät (Tage seit Beginn)
PHASE_NAMES = ("früh (Tag 0 bis 3)", "mittel (Tag 3 bis 8)", "spät (nach Tag 8)")
FOLLOW_UPS = (2.0, 5.0, 10.0, 20.0, 50.0, float("inf"))   # Länge der Nachbeobachtung für das einzelne Hazard-Verhältnis
HR_PLOT_HORIZON = 40.0                     # Zeitachse des Hazard-Verhältnis-Diagramms (Tage)

# Experiment 3: zufällige Zuteilung, zweites Merkmal
GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA, GAMMA_STEP = 0.0, 2.0, 1.0, 0.25    # Wirkung des zweiten Merkmals auf den Logarithmus der Rate je Standardabweichung

# Experiment 4: Was stattdessen
TAU_MIN, TAU_MAX, DEFAULT_TAU, TAU_STEP = 1, 40, 10, 1                    # Zeitpunkt τ (Tage)

REPS = 200                                 # Läufe je Zelle (feste Seeds), in der App und in tools/measure_hazard.py
Z95 = 1.959963984540054                    # Normalquantil für das zweiseitige 95-%-Intervall

COLOR_CONTROL = "#4c78a8"
COLOR_TREATED = "#e15759"
COLOR_HR = "#2F6B65"
COLOR_TRUTH = "#7f7f7f"
COLOR_SIM = "#f28e2b"

# --- Presets ------------------------------------------------------------------------------------------------------------------------------------
# Schlüssel: factor, n, seed (allgemein), ratea, rateb, shareb, probe (Experiment 1), v (2), gamma (3), tau (4); fehlt ein Schlüssel, gilt der Standardwert.
PRESETS = {
    "Standard (v = 1, Faktor 2)": {},
    "Keine Streuung (v = 0)": {"v": 0.0},
    "Starke Streuung (v = 2)": {"v": 2.0},
    "Keine Wirkung (Faktor 1)": {"factor": 1.0},
    "Kleine Stichprobe (n = 400)": {"n": 400},
    "Zweites Merkmal stark (γ = 2)": {"gamma": 2.0},
    "Zwei fast gleiche Klassen": {"ratea": 1.0, "rateb": 1.5},
}
PRESET_HELP = {
    "Standard (v = 1, Faktor 2)": "Faktor 2 für jeden Container, Anfälligkeit gamma-verteilt mit Varianz 1, 2000 Container: die Welt, auf die sich die Messungen der README beziehen.",
    "Keine Streuung (v = 0)": "Alle Container haben dieselbe Anfälligkeit: das Hazard-Verhältnis bleibt in jeder Phase beim Faktor.",
    "Starke Streuung (v = 2)": "Anfälligkeit mit Varianz 2: das Hazard-Verhältnis der späten Phase fällt weit unter den Faktor.",
    "Keine Wirkung (Faktor 1)": "Die Maßnahme ändert keine Rate: es gibt nichts, was wegsortiert werden könnte, und das Hazard-Verhältnis bleibt in jeder Phase bei 1.",
    "Kleine Stichprobe (n = 400)": "400 Container: breitere Streuung der Schätzwerte, in der späten Phase gibt es weniger Ereignisse.",
    "Zweites Merkmal stark (γ = 2)": "Experiment 3 mit stark wirkendem zweitem Merkmal: der Unterschied zwischen dem Cox-Schätzer mit und ohne dieses Merkmal wird größer.",
    "Zwei fast gleiche Klassen": "Experiment 1 mit den Raten 1 und 1,5 je Tag: der Tiefpunkt des Verhältnisses liegt bei 1,87 (im Standardbeispiel mit den Raten 1 und 4 bei 1,47).",
}

# --- Annahmen, Grenzen und bekannte Probleme (derselbe Text steht in der README; ein Test vergleicht beide) -------------------------------------
ASSUMPTIONS_MARKDOWN = """\
**Annahmen**
- Die Zensierung ist unabhängig vom Ereignis. Diese Demo erzeugt keine zufällige Zensierung; zensiert wird nur an festen, vorab gewählten Zeiten (Ende einer Phase oder der Nachbeobachtung), unabhängig von jedem Container. Die Annahme ist damit erfüllt, geprüft wird sie hier nicht.
- Jeder Container hat in der Simulation dieselbe Wirkung der Maßnahme: die Rate wird mit demselben Faktor multipliziert. Unterschiede zwischen Containern gibt es nur in der Grundrate (Anfälligkeit).
- Die Anfälligkeit ist gamma-verteilt (Experiment 2) bzw. lognormal-verteilt (das zweite Merkmal in Experiment 3). Das ist eine Annahme der Simulation. Ob echte Hafendaten so streuen, zeigt die Demo nicht.
- Die Maßnahme wird zufällig zugeteilt: kein Störfaktor, keine Auswahl in die Gruppen.

**Grenzen**
- Alle Zahlen stammen aus simulierten Welten mit exponentiell verteilter Verweilzeit je Container; andere Welten sind nicht gemessen; Größen und Richtung können dort anders sein.
- Die Demo zeigt den Mechanismus (das Hazard-Verhältnis der noch Vorhandenen driftet, ein weggelassenes Merkmal verschiebt den Wert) und nicht, wie stark er in echten Studien ist. Zu dieser Frage gibt es einen offenen fachlichen Streit; die Demo stellt ihn dar und entscheidet ihn nicht.
- Die Maßnahme beschleunigt in jedem Container die Abholung (Faktor größer als 1, oder 1 ohne Wirkung); ein Hazard-Verhältnis unter 1 kommt in dieser Welt nicht vor.
- Der Kausalbegriff (wann ein Hazard-Verhältnis als Wirkung der Maßnahme gelten darf) wird nicht geprüft, nur dargestellt.

**Bekannte Probleme und Unsicherheiten**
- Die Läufe sind Stichproben mit festen Zufalls-Seeds: Zahlen wie „0,17 im Mittel über 200 Läufe“ schwanken um einige Einheiten der letzten Stelle, wenn man den Zufalls-Seed ändert.
- Der Grenzwert für unendlich viele Container (Spalte „Grenzwert“) wird numerisch integriert; die Genauigkeit ist auf mindestens 4 Stellen geprüft (tests/test_oracle_hazard_verhaeltnis.py).
- Die Seite ist im Browser nur am Rechner geprüft (Seitenaufruf, Regler über AppTest), nicht auf Mobilgeräten.
"""
