"""Was das Hazard-Verhältnis misst - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Eine Maßnahme verdoppelt (Faktor f) die Abholrate jedes einzelnen Containers. Was misst dann das Hazard-Verhältnis der Gruppen, das die Cox-Regression ausgibt?
Vier Experimente: zwei Klassen von Containern von Hand, gamma-verteilte Anfälligkeit (das Verhältnis der noch vorhandenen Container driftet), zufällige Zuteilung mit einem
weggelassenen zweiten Merkmal, und dieselben Daten als Überlebenswahrscheinlichkeit und mittlere Verweilzeit bis τ gelesen. Dazu der fachliche Streit um das Hazard-Verhältnis.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import hzv_constants as C
import hzv_evaluation as ev
import hzv_model as M
import hzv_visualization as V
from hzv_layout import dunkles_design, kopf_css
from hzv_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from hzv_scenario import ClassWorld, FrailtyWorld, RandomWorld

st.set_page_config(page_title="Hazard-Verhältnis – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _frailty(world: FrailtyWorld):
    return M.run_frailty_study(world, C.REPS), M.population_windows(world), M.population_follow_ups(world)


@st.cache_data(show_spinner=False)
def _random(world: RandomWorld):
    return M.run_random_study(world, C.REPS), M.population_random(world)


@st.cache_data(show_spinner=False)
def _stay(world: FrailtyWorld, tau: float):
    return M.analyse_stay(world, tau)


def md_table(headers, rows) -> str:
    """Markdown-Tabelle aus Texten (kein | in den Zellen)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in hzv_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Was das Hazard-Verhältnis misst")
st.markdown(
    """
Eine **Maßnahme** (etwa eine andere Abfertigungsregel) vervielfacht die **Abholrate** jedes einzelnen Containers mit demselben **Faktor**. Das **Hazard-Verhältnis** (HR) vergleicht die Raten
der beiden Gruppen **unter den Containern, die zu diesem Zeitpunkt noch da sind**; die Cox-Regression gibt dafür eine einzige Zahl aus. Die Demo zeigt in Simulationen, wann diese Zahl
etwas anderes misst als den Faktor, den jeder Container erfährt: wenn die Container **unterschiedlich anfällig** sind (Experimente 1 und 2) und wenn man ein **Merkmal weglässt**, obwohl zufällig
zugeteilt wurde (Experiment 3). Experiment 4 liest dieselben Daten als **Überlebenswahrscheinlichkeit** und **mittlere Verweilzeit** bis zu einem festen Zeitpunkt τ.
Wie stark das in echten Studien ins Gewicht fällt, ist **fachlich umstritten**; die Demo stellt den Streit dar (Abschnitt „Der Streit um das Hazard-Verhältnis“) und entscheidet ihn nicht.

**Annahme in dieser Demo: Die Zensierung ist unabhängig vom Ereignis.** Es gibt hier keine zufällige Zensierung; zensiert wird nur an festen, vorab gewählten Zeiten (Ende einer Phase
oder der Nachbeobachtung), unabhängig von jedem Container. Die Annahme ist damit erfüllt, geprüft wird sie hier nicht.

Weiter unten: **🔬 Experiment 1** (zwei Klassen von Hand), **❓ Kernfrage mit Experiment 2** (Streuung der Anfälligkeit), **🔬 Experiment 3** (zufällige Zuteilung, zweites Merkmal),
**🔬 Experiment 4** (was stattdessen berichtet werden kann), **⚖️ Der Streit um das Hazard-Verhältnis**, **🚧 Wo die Annahmen enden** sowie die Expander
**⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man ein Hazard-Verhältnis", expanded=True):
    st.markdown(
        """
1. **Rate.** Die Rate (der Hazard) $h(t)$ einer Gruppe ist der Anteil der Container, die zur Zeit $t$ abgeholt werden, **gemessen an den Containern, die zu $t$ noch da sind**.
2. **Verhältnis.** $\\mathrm{HR}(t) = h_{\\text{Maßnahme}}(t) / h_{\\text{Kontrolle}}(t)$. Die Cox-Regression schätzt eine **einzige** Zahl, die für alle Zeiten gelten soll (proportionale Raten).
3. **Je Container gilt ein Faktor $f$**, in dieser Welt für jeden Container derselbe. Das Verhältnis der **Übrigen** ist aber ein Verhältnis zweier **Mischungen** von Containern mit
   verschiedenen Grundraten, und die Mischungen verändern sich mit der Zeit.
4. **Aussieben.** Container mit hoher Grundrate verlassen eine Gruppe früher; bei einem Faktor über 1 geschieht das in der Maßnahme-Gruppe schneller. Dort bleiben dann im Mittel Container mit niedrigerer Grundrate
   übrig als in der Kontrolle; Experiment 2 misst diese mittlere Anfälligkeit der Übrigen.
5. **Alternativen ohne Proportionalität:** der Anteil, der nach τ Tagen noch da ist, und die mittlere Verweilzeit bis τ (Experiment 4). Beide verlangen ein vorab festgelegtes τ.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt der Simulation** (Experimente 2 bis 4; Faktor auch in Experiment 1)")
    factor = st.slider("Faktor der Maßnahme je Container", *bounds("factor_slider"), step=C.FACTOR_STEP, key="factor_slider",
                       help="Die Rate jedes behandelten Containers ist das f-fache seiner Rate ohne Maßnahme. 1 heißt: keine Wirkung.")
    n = st.slider("Container n je Lauf", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Zahl der Container in jedem der Läufe von Experiment 2 bis 4; die Hälfte erhält zufällig die Maßnahme.")
    seed = st.number_input("Zufalls-Seed der Daten", *bounds("seed_input"), key="seed_input", step=1,
                           help="Jeder der Läufe hat einen eigenen, festen Zufallsstrom, der aus diesem Zufalls-Seed abgeleitet ist.")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

# --- Experiment 1: zwei Klassen von Hand ---------------------------------------------------------------------------------------------------------

st.markdown("## 🧮 Das Verfahren von Hand: zwei Klassen von Containern")
st.subheader("🔬 Experiment 1: Das Verhältnis der Übrigen von Hand")
st.markdown(
    """
Zwei Klassen von Containern: **Klasse A** (langsam) und **Klasse B** (schnell), die Rate je Tag steht in den Reglern. Die Maßnahme vervielfacht die Rate **jedes** Containers mit dem Faktor $f$
aus der Seitenleiste. Die marginale Rate einer Gruppe zur Zeit $t$ ist der Mittelwert der Klassenraten, gewichtet mit dem Anteil der Klasse **unter den noch vorhandenen** Containern:
$h(t) = \\sum_k w_k r_k e^{-r_k t} \\big/ \\sum_k w_k e^{-r_k t}$. Ein Zufallselement gibt es hier nicht.
"""
)
c1, c2, c3, c4 = st.columns(4)
rate_a = c1.slider("Rate der Klasse A (je Tag)", *bounds("ratea_slider"), step=C.RATE_A_STEP, key="ratea_slider",
                   help="Die langsamere Klasse: ihre Container werden mit dieser Rate je Tag abgeholt.")
rate_b = c2.slider("Rate der Klasse B (je Tag)", *bounds("rateb_slider"), step=C.RATE_B_STEP, key="rateb_slider",
                   help="Die schnellere Klasse: ihre Rate liegt immer über der der Klasse A (die Bereiche der beiden Regler überlappen nicht).")
share_b = c3.slider("Anteil der Klasse B", *bounds("shareb_slider"), step=C.SHARE_B_STEP, key="shareb_slider",
                    help="Anteil der Klasse B an allen Containern zu Beginn (in beiden Gruppen gleich).")
probe = c4.slider("Zeit t zum Ausprobieren (Tage)", *bounds("probe_slider"), step=C.PROBE_STEP, key="probe_slider",
                  help="Die Raute im rechten Diagramm und die drei Kennzahlen darunter gelten für diese Zeit.")
class_world = ClassWorld(float(rate_a), float(rate_b), float(share_b), float(factor))
ca = ev.analyse_classes(class_world, float(probe))

ex = ca.exact
st.markdown(
    f"**Bei t = 0, in Brüchen:** Rate der Kontrolle $= (1 - s)\\,r_A + s\\,r_B$ = {ev.frac(ex['control'])} je Tag; Rate der Maßnahme $= f \\cdot$ Rate der Kontrolle = {ev.frac(ex['treated'])} je Tag; "
    f"Verhältnis = **{ev.frac(ex['hr'])}**, gleich dem Faktor, für jede Aufteilung der Klassen."
)
rows = [[f"{r['t']:g}", ev.num(r["h_control"], 4), ev.num(r["h_treated"], 4), ev.num(r["hr"], 4), ev.pct(r["share_b_control"], 1), ev.pct(r["share_b_treated"], 1)] for r in ca.rows]
st.markdown(md_table(["Zeit t (Tage)", "Rate Kontrolle", "Rate Maßnahme", "Verhältnis", "Anteil Klasse B unter den Übrigen, Kontrolle", "Anteil Klasse B unter den Übrigen, Maßnahme"], rows))
k1, k2, k3 = st.columns(3)
k1.metric(f"Rate der Kontrolle bei t = {ev.num(ca.probe_row['t'], 2)}", ev.num(ca.probe_row["h_control"], 4))
k2.metric(f"Rate der Maßnahme bei t = {ev.num(ca.probe_row['t'], 2)}", ev.num(ca.probe_row["h_treated"], 4))
k3.metric("Verhältnis", ev.num(ca.probe_row["hr"], 4), delta=f"Faktor je Container {ev.num(class_world.factor, 2)}", delta_color="off")
st.plotly_chart(V.build_class_curves(class_world, float(probe), ca.t_min, ca.hr_min), width="stretch", key="class_chart")
kind, text = ev.verdict_classes(ca)
getattr(st, kind)(text)

st.markdown("---")

# --- Kernfrage und Experiment 2: Streuung der Anfälligkeit ---------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Misst das Hazard-Verhältnis den Faktor, den jeder Container erfährt?")
st.subheader("🔬 Experiment 2: Streuung der Anfälligkeit")
st.markdown(
    f"""
Welt: Jeder Container hat eine eigene **Anfälligkeit** $Z$ (unbeobachtet, gamma-verteilt mit Mittel 1 und der Varianz $v$ aus dem Regler). Seine Rate ist {C.BASE_RATE:g} je Tag mal $Z$, mit Maßnahme
zusätzlich mal $f$; die Verweilzeit eines Containers ist bei gegebenem $Z$ exponentiell verteilt. Die Hälfte der Container erhält die Maßnahme (zufällig). Es laufen **{C.REPS} Läufe** mit festen
Zufalls-Seeds, alle Container werden bis zur Abholung beobachtet. Das Cox-$\\hat\\beta$ wird in **drei Phasen** getrennt geschätzt: früh (Tage 0 bis 3), mittel (Tage 3 bis 8, nur Container, die bei
Tag 3 noch da sind, Zeit ab 3) und spät (nur Container, die bei Tag 8 noch da sind, Zeit ab 8). Wahrheit je Container: $\\ln f$ in jeder Phase.
"""
)
v = st.slider("Varianz v der Anfälligkeit", *bounds("v_slider"), step=C.V_STEP, key="v_slider",
              help="0: alle Container gleich anfällig. Je größer v, desto stärker streut die Grundrate zwischen den Containern. Die Messungen der README nutzen 0, 0,5, 1, 2 und 3.")
fw = FrailtyWorld(int(n), float(v), float(factor), int(seed))
with st.spinner(f"Rechne {C.REPS} feste Läufe mit je neun Cox-Anpassungen (die Rechenzeit wächst mit n) …"):
    study, pops, fu_pops = _frailty(fw)

cols = [M.column(study.windows[:, j], study.reps) for j in range(3)]
m1, m2, m3 = st.columns(3)
for mcol, j, label in ((m1, 0, "Hazard-Verhältnis früh (Tag 0 bis 3)"), (m2, 1, "mittel (Tag 3 bis 8)"), (m3, 2, "spät (nach Tag 8)")):
    mcol.metric(label, ev.hr_text(cols[j].mean) if cols[j].n_ok else "–", delta=f"Grenzwert {ev.hr_text(pops[j])}; Faktor je Container {ev.num(fw.factor, 2)}", delta_color="off")
st.plotly_chart(V.build_hr_over_time(fw, study, pops), width="stretch", key="hr_time_chart")
st.caption(
    "Die Linie ist die geschlossene Formel HR(t) = f (1 + vλt) / (1 + f vλt) für das Hazard-Verhältnis der noch vorhandenen Container; orange: Mittel der Cox-Schätzer der Läufe je Phase; "
    "Raute: Grenzwert des Cox-Schätzers für unendlich viele Container (numerisch aus derselben Formel gerechnet, nicht aus den Läufen)."
)
kind, text = ev.verdict_frailty(fw, study, pops)
getattr(st, kind)(text)
note = ev.failure_text(study.status_counts, study.reps)
if note:
    st.warning(note)

st.markdown("**Warum? Die mittlere Anfälligkeit der Übrigen** (Datensatz des ersten Laufs; nur die Simulation kennt $Z$)")
srows = ev.survivor_frailty_rows(fw)
st.markdown(md_table(
    ["Zeit t (Tage)", "noch da, Kontrolle", "noch da, Maßnahme", "mittleres Z der Übrigen, Kontrolle", "mittleres Z der Übrigen, Maßnahme", "Verhältnis", "Formel HR(t) / f"],
    [[f"{r['t']:g}", r["alive0"], r["alive1"], ev.num(r["mean0"], 3), ev.num(r["mean1"], 3), ev.num(r["ratio"], 3), ev.num(r["hr_over_f"], 3)] for r in srows],
))
st.caption(
    "Bei gamma-verteilter Anfälligkeit ist das mittlere $Z$ der Übrigen in Gruppe $x$ zur Zeit $t$ gleich $1/(1 + v\\lambda f^x t)$; das Verhältnis der beiden ist HR(t)/f. "
    "In dieser Welt ist das Hazard-Verhältnis der Übrigen daher der Faktor mal dem Verhältnis der mittleren Anfälligkeit der Übrigen. Zufallsschwankung steht in den Zählungen "
    "(wenige Container bei großem t)."
)

st.markdown("**Ein einziges Hazard-Verhältnis hängt an der Länge der Nachbeobachtung**")
st.plotly_chart(V.build_follow_up(fw, study, fu_pops), width="stretch", key="follow_up_chart")
st.info(ev.follow_up_text(fw, study, fu_pops))

st.markdown("---")

# --- Experiment 3: zufällige Zuteilung, zweites Merkmal ------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Zufällige Zuteilung, aber ein Merkmal weggelassen")
st.markdown(
    f"""
Jetzt gibt es keine Anfälligkeit mit Gamma-Verteilung, sondern ein **zweites Merkmal** $z$ (standardnormal, etwa die Größe der Fracht), das die Rate jedes Containers mit $e^{{\\gamma z}}$ vervielfacht;
$\\gamma$ steht im Regler. Die Maßnahme wird **zufällig** zugeteilt, unabhängig von $z$: ein Störfaktor ist ausgeschlossen. Rate eines Containers: {C.BASE_RATE:g} · $f^x$ · $e^{{\\gamma z}}$.
Verglichen werden zwei Cox-Anpassungen derselben Daten ({C.REPS} Läufe, alle Container bis zur Abholung beobachtet): nur mit der Maßnahme $x$, und mit $x$ und $z$.
"""
)
gamma = st.slider("Wirkung γ des zweiten Merkmals", *bounds("gamma_slider"), step=C.GAMMA_STEP, key="gamma_slider",
                  help="Logarithmus des Faktors je Standardabweichung des zweiten Merkmals. 0: das Merkmal beeinflusst die Rate nicht.")
rw = RandomWorld(int(n), float(gamma), float(factor), int(seed))
with st.spinner(f"Rechne {C.REPS} feste Läufe mit je zwei Cox-Anpassungen (die Rechenzeit wächst mit n) …"):
    rstudy, pop_x = _random(rw)
ra, rb = M.column(rstudy.only_x, rstudy.reps), M.column(rstudy.with_z, rstudy.reps)
q1, q2, q3 = st.columns(3)
q1.metric("Cox nur mit x: β̂ (Mittel)", ev.num(ra.mean, 3) if ra.n_ok else "–", delta=f"Grenzwert {ev.num(pop_x, 3)}", delta_color="off")
q2.metric("Cox mit x und z: β̂ (Mittel)", ev.num(rb.mean, 3) if rb.n_ok else "–", delta="Wahrheit je Container ln f = " + ev.num(math.log(rw.factor), 3), delta_color="off")
q3.metric("Hazard-Verhältnis nur mit x", ev.hr_text(ra.mean) if ra.n_ok else "–", delta=f"Faktor je Container {ev.num(rw.factor, 2)}", delta_color="off")
st.plotly_chart(V.build_random(rw, rstudy, pop_x), width="stretch", key="random_chart")
st.caption("Fehlerbalken: ± 1 Standardfehler des Mittels über die Läufe. Raute: Grenzwert des Cox-Schätzers nur mit x für unendlich viele Container (numerisch integriert).")
kind, text = ev.verdict_random(rw, rstudy, pop_x)
getattr(st, kind)(text)
note = ev.failure_text(rstudy.status_counts, rstudy.reps)
if note:
    st.warning(note)

st.markdown("---")

# --- Experiment 4: Was stattdessen ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Was stattdessen berichtet werden kann")
st.markdown(
    """
Dieselben Daten wie in Experiment 2 (Lauf 1 der Welt dort, gleiche Streuung $v$), anders gelesen: **Wie viele Container sind nach $\\tau$ Tagen noch da** (Kaplan–Meier-Schätzer mit Greenwood-Band),
und **wie lang ist die mittlere Verweilzeit bis $\\tau$** (Fläche unter der Kurve). Beides ist an keine Proportionalität gebunden. Der Zeitpunkt $\\tau$ gehört vorab festgelegt; ändert man ihn nach
Ansicht der Daten, ändern sich die Zahlen (und ihr Niveau als Test, siehe Hinweis unten).
"""
)
tau = st.slider("Zeitpunkt τ (Tage)", *bounds("tau_slider"), step=C.TAU_STEP, key="tau_slider",
                help="Beide Maße werden zu diesem Zeitpunkt gelesen. Die Intervalle sind punktweise 95-%-Intervalle bei diesem τ.")
sync_query_params({
    "factor_slider": float(factor), "n_slider": int(n), "seed_input": int(seed), "ratea_slider": float(rate_a), "rateb_slider": float(rate_b), "shareb_slider": float(share_b),
    "probe_slider": float(probe), "v_slider": float(v), "gamma_slider": float(gamma), "tau_slider": int(tau),
})
sy = _stay(fw, float(tau))
st.plotly_chart(V.build_stay_curves(sy, fw), width="stretch", key="stay_chart")
st.caption("Durchgezogen: Kaplan–Meier-Kurve mit punktweisem 95-%-Band (log-log-Greenwood); gestrichelt: die wahre Kurve der Welt (Formel). Die Treppe ist für die Anzeige ausgedünnt.")
srows = [
    ["Anteil noch da nach τ", ev.pct(sy.s0, 1), ev.pct(sy.s1, 1), f"{ev.num(100 * sy.s_diff, 1)} Prozentpunkte [{ev.num(100 * sy.s_diff_band[0], 1)}; {ev.num(100 * sy.s_diff_band[1], 1)}]",
     f"{ev.num(100 * (sy.truth_s1 - sy.truth_s0), 1)} Prozentpunkte"],
    ["mittlere Verweilzeit bis τ (Tage)", ev.num(sy.rmst0, 2), ev.num(sy.rmst1, 2),
     f"{ev.num(sy.rmst_diff, 2)} [{ev.num(sy.rmst_band[0], 2)}; {ev.num(sy.rmst_band[1], 2)}]" if sy.rmst_status == "ok" else "kein Standardfehler",
     ev.num(sy.truth_rmst1 - sy.truth_rmst0, 2)],
]
st.markdown(md_table(["Maß", "Kontrolle", "Maßnahme", "Differenz [95-%-Intervall]", "Wahrheit der Differenz"], srows))
kind, text = ev.verdict_stay(sy)
getattr(st, kind)(text)
st.info(ev.phase_table_text(sy))
st.caption(
    "Ein einzelnes τ ist selbst eine Wahl: Hernán (2010) weist darauf hin, dass auch eine einzelne Überlebenswahrscheinlichkeit irreführen kann (Volltext über Abruf gelesen). "
    "Royston und Parmar (2013) legen für die Hauptanalyse mit der mittleren Lebenszeit ein vorab festgelegtes τ zugrunde (Volltext über Abruf gelesen)."
)

st.markdown("---")

# --- Der Streit ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("⚖️ Der Streit um das Hazard-Verhältnis")
st.markdown(
    """
Wie weit das Hazard-Verhältnis als Wirkungsmaß taugt, ist **offen**. Die Tabelle gibt die Positionen in eigenen Worten wieder, mit der **Lesestufe** der jeweiligen Quelle:
*Volltext selbst gelesen* (Text aus der PDF gelesen); *Volltext über Abruf* (Volltextseite vom Abrufwerkzeug zusammengefasst, Rohtext nicht selbst gesehen, Zahlen und Satzbau noch gegenzulesen);
*Abstract* (nur Kurzfassung). Die Spalte rechts sagt, was diese Demo dazu misst und was nicht. **Keine der Positionen ist hier als Urteil gemeint.**
"""
)
st.markdown(md_table(
    ["Position", "Lesestufe", "Aussage in eigenen Worten", "Was diese Demo dazu zeigt und was nicht"],
    [
        ["**Hernán (2010)**, Epidemiology 21(1):13–15 (Kommentar)", "Volltext über Abruf",
         "Ein über die Nachbeobachtung gemitteltes HR hängt von deren Dauer ab. Periodenspezifische HR enthalten eine eingebaute Auswahlverzerrung: Anfälligere der behandelten Gruppe scheiden früher aus, "
         "späte HR sinken Richtung oder unter 1. Empfohlen: angepasste Überlebenskurven und eine "
         "Reihe gemittelter HR. Der Text enthält weder eine Simulation noch eine Herleitung. *Lesart des Lektüre-Berichts dieser Linie (nicht wörtlich bei Hernán belegt):* Die Verzerrung setzt eine Wirkung der Behandlung voraus (etwa frühe Schädigung); bei strenger Nullwirkung gibt es sie nicht; bei Ying und Xu steht dazu sinngemäß, die Gruppen nach dem Überleben seien nur unter der scharfen Nullhypothese identisch (nicht wörtlich geprüft).",
         "Der Mechanismus tritt in Experiment 2 auf (HR der Übrigen sinkt gegen 1), die Abhängigkeit von der Nachbeobachtung in der Tabelle darunter, und ohne Wirkung (Faktor 1) gibt es keinen Drift. "
         "Nicht gezeigt: ein Fall unter 1 (hier beschleunigt die Maßnahme in jedem Container die Abholung), und wie stark der Effekt in echten Studien ist."],
        ["**Abrahamowicz u. a. (2025)**, Eur. J. Epidemiol. 40(6):611–629 (Simulationen, Fallstudien)", "Volltext über Abruf (rund 100 000 von 122 000 Zeichen; Referenzen und Zusatzmaterial fehlen)",
         "Bestätigen den Mechanismus (Erschöpfung Anfälliger, Verzerrung zur Null). Halten ihn **für die nachgebaute WHI-Studie unter den geprüften Annahmen** für eine sehr unwahrscheinliche Erklärung "
         "der beobachteten Kreuzung; bei sehr starker Anfälligkeit (HR der Anfälligkeit 25) bleibt ein Bias bis etwa 58 %, bei HR der Anfälligkeit höchstens 5 unter 20 % (Gesamt-HR). Nach Ansicht der Autoren habe Hernán seine Behauptung weder "
         "algebraisch noch per Simulation gestützt. Ihr Gesamtschluss, nur für die nachgebaute WHI-Konstellation unter den geprüften Annahmen: Als Alternativen zur Anfälligkeit nennen sie "
         "Nichteinnahme und echte zeitabhängige Wirkung; Bedenken gegen das HR könnten übertrieben sein.",
         "Die Größe des Effekts hängt an Stärke und Streuung der Anfälligkeit (Experiment 2: Regler v). Die Welt der Demo (gamma-verteilte Anfälligkeit, kein WHI-Nachbau) ist nicht ihre; "
         "die Zahlen sind nicht vergleichbar, und die Demo entscheidet nicht, ob echte Daten so streuen."],
        ["**Ying und Xu**, arXiv 2307.11971 (v2, Okt. 2023), „On Defense of the Hazard Ratio“", "Volltext selbst gelesen",
         "Streit um den **Kausalbegriff**, nicht um den Mechanismus. Ihr Hauptergebnis: Die Hazard-Funktion habe in den drei Rahmen von Rubin, Robins und Pearl eine kausale Deutung. "
         "Nach der strengen Definition von Rubin sei das HR über Risikomengen nicht kausal (außer bei der scharfen Nullhypothese), "
         "nach Robins und Pearl schon. Die Erschöpfung der Anfälligen sei die bekannte unbeobachtete Heterogenität. Ob ein späteres niedrigeres Hazard Wirkung oder Auswahl ist, lasse sich an den "
         "Überlebenskurven prüfen (im reinen Auswahlfall kreuzen sie sich nicht; kreuzen sie sich, deute das auf eine echte Wirkung; zusätzlich können Sensitivitätsanalysen nötig sein); man solle den ganzen Verlauf des HR ansehen, nicht einen späten Zeitpunkt allein.",
         "Die Überlebenskurven kreuzen sich in dieser Welt nicht (Experiment 1 und Formel in Experiment 2), und die Wirkung je Container ist konstant: Das Absinken ist hier reine Auswahl. "
         "Zur Kausalfrage selbst sagt die Demo nichts."],
        ["**Royston und Parmar (2013)**, BMC Med. Res. Methodol. 13:152", "Volltext über Abruf (rund 200 000 von 218 000 Zeichen)",
         "Die mittlere Lebenszeit bis τ (RMST) als Alternative: Verlust an Lebenszeit bis τ, gültig ohne Proportionalität, relatives und absolutes Maß. Für die Hauptanalyse legen sie ein **vorab festgelegtes** τ zugrunde; "
         "in ihrer Simulation hat der RMST-Test bei einem nicht proportionalen Szenario eine höhere Trennschärfe als der Log-Rank-Test. Nachteil laut Tabelle 5: τ muss gewählt werden.",
         "Experiment 4 liest dieselben Daten als Überlebenswahrscheinlichkeit und mittlere Verweilzeit bis τ. Die Trennschärfe vergleicht diese Demo nicht."],
    ],
))
st.caption(
    "Die in der Neuanalyse genannte Übersicht über 27 große Studien (Strobel u. a. 2023) wurde nicht gefunden und ist hier nicht verwendet. Hernán (2010), die Neuanalyse 2025 und Royston und Parmar (2013) sind nur über "
    "Abruf-Zusammenfassungen gelesen: Zahlen und Satzbau stehen unter dem Vorbehalt, im PDF gegengelesen zu werden."
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was gilt, wenn sie verletzt ist |
|---|---|
| **Zensierung unabhängig vom Ereignis** | Nicht gemessen: Die Welten dieser Demo haben nur feste, vom Container unabhängige Zensierzeiten. Keine Zahl hier sagt etwas über abhängige Zensierung. |
| **Dieselbe Wirkung für jeden Container** (Faktor $f$) | Nicht gemessen: Hat die Maßnahme in einem Teil der Container einen anderen Faktor (oder wirkt nur früh), verändert das die Verläufe; ein Wechsel zu einem Verhältnis unter 1 kommt in dieser Welt nicht vor. |
| **Gamma-verteilte Anfälligkeit** (Experiment 2) und **lognormal** wirkendes zweites Merkmal (Experiment 3) | Annahmen der Simulation. Ob echte Hafendaten so streuen, zeigt die Demo nicht. |
| **Zufällige Zuteilung** | Gegeben: kein Störfaktor, keine Auswahl in die Gruppen. Mit einem Störfaktor kommt eine weitere Verzerrung dazu, die hier nicht gemessen ist. |
| **Eine feste, vorab gewählte Zeit τ** (Experiment 4) | Der Regler lädt zum Probieren ein; wer τ nach Ansicht der Daten wählt, verändert das Niveau eines Tests. Das zeigt diese Demo nicht. |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* je Container die Gruppe $x \in \{0, 1\}$ (Maßnahme ja oder nein), die Zeit bis zur Abholung $T_i$ und eine unbeobachtete Anfälligkeit $Z_i$ (oder ein zweites Merkmal $z_i$).
*Gesucht:* was das Cox-Hazard-Verhältnis $e^{\hat\beta}$ über die Gruppen aussagt, wenn jeder Container dieselbe Wirkung $f$ erfährt, und welche Größen stattdessen berichtet werden können.

| Symbol | Bedeutung |
|---|---|
| $x$ | Gruppe: 1 Maßnahme, 0 Kontrolle (zufällig, je 50 %) |
| $f$ | Faktor der Maßnahme je Container (Rate mal $f$); $\beta = \ln f$ ist die Wahrheit je Container |
| $\lambda$ | Grundrate je Tag (hier 0,1) |
| $Z$ | unbeobachtete Anfälligkeit eines Containers; gamma-verteilt mit Mittel 1 und Varianz $v$ |
| $r_k,\ w_k$ | Rate und Anteil der Klasse $k$ (Experiment 1) |
| $S_x(t),\ f_x(t),\ h_x(t)$ | Überlebensfunktion, Dichte und Rate der Gruppe $x$ (marginal, über alle Container der Gruppe) |
| $\mathrm{HR}(t)$ | Verhältnis der marginalen Raten $h_1(t)/h_0(t)$ |
| $\gamma,\ z$ | Wirkung und Wert des zweiten Merkmals (Experiment 3), $z \sim N(0, 1)$ |
| $\hat S(\tau),\ \widehat{\mathrm{RMST}}(\tau)$ | Kaplan–Meier-Schätzer bei $\tau$ und mittlere Verweilzeit bis $\tau$ (Fläche unter der Kurve) |

**Marginale Rate einer Mischung** (Klassen mit Raten $r_k$, Anteilen $w_k$):

$$h(t) = \frac{\sum_k w_k r_k e^{-r_k t}}{\sum_k w_k e^{-r_k t}}, \qquad \mathrm{HR}(0) = f \text{ für jede Aufteilung}$$

**Gamma-Anfälligkeit.** Die Laplace-Transformierte der Gamma-Verteilung (Mittel 1, Varianz $v$) ist $\mathrm{E}\,e^{-sZ} = (1 + vs)^{-1/v}$. Mit der Rate $\lambda f^x Z$ folgt:

$$S_x(t) = \big(1 + v\lambda f^x t\big)^{-1/v}, \qquad h_x(t) = \frac{\lambda f^x}{1 + v\lambda f^x t}, \qquad \mathrm{HR}(t) = \frac{f\,(1 + v\lambda t)}{1 + f v\lambda t}$$

$\mathrm{HR}(0) = f$, $\mathrm{HR}(t) \to 1$ für $t \to \infty$ bei $v > 0$ und $\mathrm{HR}(t) = f$ bei $v = 0$. Die mittlere Anfälligkeit der Übrigen zur Zeit $t$ ist $\mathrm{E}[Z \mid T > t, x] = 1 / (1 + v\lambda f^x t)$,
also $\mathrm{HR}(t) / f = \mathrm{E}[Z \mid T > t, 1] \,/\, \mathrm{E}[Z \mid T > t, 0]$. Die mittlere Verweilzeit bis $\tau$ ist $\int_0^\tau S_x(t)\,dt = \dfrac{(1 + v\lambda f^x \tau)^{1 - 1/v} - 1}{v\lambda f^x\,(1 - 1/v)}$
(bei $v = 1$: $\ln(1 + \lambda f^x \tau) / (\lambda f^x)$; bei $v = 0$: $(1 - e^{-\lambda f^x \tau}) / (\lambda f^x)$).

**Grenzwert des Cox-Schätzers** (ein binäres Merkmal, gleich große Gruppen, Ereignisse im Fenster $(a, b]$, Container, die bei $a$ noch da sind, werden bei $b$ zensiert). Der erwartete Score je Container ist

$$U(\beta) = \int_a^b \Big[ f_1(t) - \big(f_0(t) + f_1(t)\big)\, \frac{S_1(t)\,e^{\beta}}{S_0(t) + S_1(t)\,e^{\beta}} \Big]\, dt, \qquad \hat\beta_\infty:\; U(\hat\beta_\infty) = 0$$

(die Risikomenge der Gruppe $x$ hat zur Zeit $t$ den Umfang $\tfrac{n}{2} S_x(t)$). Die Demo löst die Gleichung per Bisektion auf einer zusammengesetzten Gauß–Legendre-Quadratur und zeigt sie als „Grenzwert“.

**Zweites Merkmal (Experiment 3).** Mit der Rate $\lambda f^x e^{\gamma z}$ ist die Rate der Gruppe $x$ ohne $z$: $S_x(t) = \mathrm{E}_z\, \exp\!\big(-\lambda f^x e^{\gamma z} t\big)$, $h_x = -S_x'/S_x$.
Das Verhältnis $h_1(t)/h_0(t)$ ist dann nicht mehr konstant gleich $f$, obwohl die Zuteilung unabhängig von $z$ ist. Mit $x$ und $z$ im Modell gilt $\hat\beta \to \ln f$.

**Überlebenswahrscheinlichkeit und mittlere Verweilzeit (Experiment 4).** Kaplan–Meier $\hat S(\tau) = \prod_{t_i \le \tau} (1 - d_i / n_i)$ mit Greenwood-Varianz
$\widehat{\mathrm{Var}}\,\hat S(\tau) = \hat S(\tau)^2 \sum_{t_i \le \tau} d_i / (n_i (n_i - d_i))$; mittlere Verweilzeit $\widehat{\mathrm{RMST}}(\tau) = \int_0^\tau \hat S(t)\,dt$ mit
$\widehat{\mathrm{Var}} = \sum_{t_i < \tau} \big(\int_{t_i}^\tau \hat S\big)^2 d_i / (n_i (n_i - d_i))$. Die Differenz zweier unabhängiger Gruppen hat die Summe der Varianzen; das 95-%-Intervall ist Differenz $\pm 1{,}96\,\mathrm{SE}$.

**Voraussetzungen und Konvention.** Zensierung unabhängig vom Ereignis (hier nur feste Zeiten); Risikomenge $R_k = \{ i : t_i \ge t_{(k)} \}$, bei gleicher Zeit zählt das Ereignis vor der Zensierung; Gleichstände nach Breslow.
Die Gamma-Formeln gelten nur für die gamma-verteilte Anfälligkeit; ein anderes Anfälligkeitsgesetz hat andere Formeln (hier nur mit dem zweiten Merkmal numerisch gerechnet).

**Literatur** (nach Lesestufe, siehe Abschnitt „Der Streit“): Hernán (2010), Epidemiology 21(1):13–15 und Abrahamowicz u. a. (2025), Eur. J. Epidemiol. 40(6):611–629 jeweils Volltext über Abruf; Ying und Xu, arXiv 2307.11971 Volltext selbst gelesen;
Royston und Parmar (2013), BMC Med. Res. Methodol. 13:152 Volltext über Abruf. Marsaglia und Tsang (2000, Gamma-Zufallszahlen) und Greenwood (1926) sind aus dem Gedächtnis genannt, die Originale sind nicht eingesehen.
Die Formeln oben sind gegen unabhängige Rechnungen geprüft (`tests/test_oracle_hazard_verhaeltnis.py`).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
