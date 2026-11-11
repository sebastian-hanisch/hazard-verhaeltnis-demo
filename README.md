# Was das Hazard-Verhältnis misst – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-hazard-verhaeltnis-demo.streamlit.app/)**

Teil der Linie **Überlebenszeit** (Zeit bis zum Ereignis bei Beobachtungen, die irgendwann abbrechen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Der Aufhänger ist die Abholung von Containern: Eine **Maßnahme** (etwa eine andere Abfertigungsregel) vervielfacht die Abholrate **jedes einzelnen** Containers mit demselben Faktor. Das **Hazard-Verhältnis**
(HR) vergleicht die Raten der beiden Gruppen unter den Containern, die zu diesem Zeitpunkt noch da sind, und die Cox-Regression gibt dafür eine einzige Zahl aus. Die Demo zeigt in Simulationen, wann diese Zahl
etwas anderes misst als den Faktor, den jeder Container erfährt (unterschiedliche Anfälligkeit; ein weggelassenes Merkmal bei zufälliger Zuteilung), liest dieselben Daten als Überlebenswahrscheinlichkeit
und mittlere Verweilzeit bis zu einem festen Zeitpunkt τ und stellt den **fachlichen Streit** um das Hazard-Verhältnis als Streit dar, mit der Lesestufe jeder Quelle.

Einordnung in der Linie: Auf die Prüfung der Proportionalität folgt diese Demo als Kontrast: Auch ohne Kreuzung der Raten ist ein Hazard-Verhältnis schwer zu deuten. Nachbarn:
[cox-regression-demo](https://github.com/sebastian-hanisch/cox-regression-demo) (das Verfahren, das hier gedeutet wird), [proportionalitaet-demo](https://github.com/sebastian-hanisch/proportionalitaet-demo) (Prüfung, ob ein
einziges Hazard-Verhältnis zu den Daten passt), [log-rank-demo](https://github.com/sebastian-hanisch/log-rank-demo) (mittlere Lebenszeit bis τ als zweite Antwort beim Gruppenvergleich).
Gleicher Aufhänger Containerverweilzeit wie [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo).

**Annahme in dieser Demo, sichtbar in der App und hier: Die Zensierung ist unabhängig vom Ereignis.** Es gibt hier keine zufällige Zensierung; zensiert wird nur an festen, vorab gewählten Zeiten (Ende einer Phase
oder der Nachbeobachtung), unabhängig von jedem Container. Die Annahme ist damit erfüllt, geprüft wird sie hier nicht.

## Warum dieses Problem

Ein Hazard-Verhältnis von 2 klingt nach „jeder behandelte Container wird doppelt so schnell abgeholt“. Gemessen wird aber ein Verhältnis zweier **Mischungen**: der Container, die zu diesem Zeitpunkt noch da sind.
Sind die Container unterschiedlich anfällig (unbeobachtet), verlassen die anfälligen eine Gruppe früher, in der Maßnahme-Gruppe schneller als in der Kontrolle; das Verhältnis der Übrigen verändert sich dann mit
der Zeit, obwohl jeder Container zu jedem Zeitpunkt denselben Faktor erfährt. In der Welt dieser Demo (Faktor 2 je Container, gamma-verteilte Anfälligkeit mit Varianz 1, 2000 Container, 200 Läufe, Zufalls-Seed 12)
schätzt Cox in den Tagen 0 bis 3 im Mittel β̂ = 0,592 (Hazard-Verhältnis 1,81), in den Tagen 3 bis 8 β̂ = 0,400 (1,49) und nach Tag 8 β̂ = 0,169 (1,18), gegen die Wahrheit je Container ln 2 = 0,693
(Hazard-Verhältnis 2). Die **Kernfrage** ist: Misst das Hazard-Verhältnis den Faktor, den jeder Container erfährt? Antwort in den simulierten Welten dieser Demo mit Wirkung (Faktor über 1): nur ohne Streuung der Anfälligkeit und ohne
weggelassene Merkmale; ohne Wirkung (Faktor 1) liegt es auch mit Streuung bei 1 (Befund 2); sonst hängt der Wert an der Zeit, an der Nachbeobachtung und an den mitgeführten Merkmalen. Wie stark das in echten Studien ist, zeigt die Demo nicht (siehe „Der Streit“).

## Modell

Rate eines Containers: 0,1 je Tag · Z · f^x mit der Gruppe x (1 Maßnahme, 0 Kontrolle; zufällig, je 50 %), dem Faktor f und der unbeobachteten Anfälligkeit Z (Experiment 2: gamma-verteilt, Mittel 1, Varianz v;
Experiment 3: statt Z ein zweites Merkmal, standardnormal, Rate · e^(γz)). Die Verweilzeit ist bei gegebener Rate exponentiell verteilt. Bei gamma-verteilter Anfälligkeit gilt geschlossen
HR(t) = f (1 + vλt) / (1 + f vλt) (Herleitung im Expander „📐 Mathematische Formulierung“ der App: Laplace-Transformierte der Gamma-Verteilung, marginale Überlebensfunktion (1 + vλf^x t)^(−1/v)); dort stehen auch
der **Grenzwert des Cox-Schätzers** für unendlich viele Container (Nullstelle des erwarteten Scores, numerisch integriert und mit Bisektion gelöst), die Formeln für Experiment 3 und für Kaplan–Meier mit Greenwood-Band und die mittlere
Verweilzeit bis τ. Der Kern (`hzv_core.py`) ist eine Kopie des gemeinsamen Kerns `uz_core.py` der Überlebenszeit-Linie (Stand 2026-11-10, mit der Trennungserkennung auch bei einem einzigen Ereignis):
SplitMix64-Zufall, Eingabeprüfung, Kaplan–Meier mit Greenwood-Band, mittlere Lebenszeit bis τ mit Varianz und z-Test, Cox mit Newton-Schrittweitenhalbierung und Konvergenzflag. Die Gamma-Zufallszahlen
(Marsaglia und Tsang) kommen aus dem SplitMix64-Strom (`hzv_scenario.py`).

## Methodik

Die App hat vier Experimente und einen Abschnitt zum Streit.

1. **Zwei Klassen von Hand.** Klasse A (langsam) und Klasse B (schnell) mit Raten je Tag, Anteil der Klasse B und dem Faktor (Seitenleiste) als Regler. Die App zeigt die Raten bei t = 0 in Brüchen
   (Handbeispiel: 5/2, 5, Verhältnis 2), eine Tabelle der marginalen Raten und des Verhältnisses zu festen Zeiten mit dem Anteil der Klasse B unter den Übrigen, die Kurven über die Zeit und einen frei wählbaren Zeitpunkt.
   Kein Zufall.
2. **Streuung der Anfälligkeit** (Kernfrage). 200 Läufe mit festen Zufalls-Seeds; Regler: Varianz v (0 bis 3), in der Seitenleiste Faktor, Container n (400 bis 3000) und Zufalls-Seed. Das Cox-β̂ wird in drei Phasen getrennt
   geschätzt (früh: Tage 0 bis 3; mittel: Tage 3 bis 8 für Container, die bei Tag 3 noch da sind, Zeit ab 3, bei 8 zensiert; spät: Container, die bei Tag 8 noch da sind, Zeit ab 8), dazu ein einziges Hazard-Verhältnis bei
   Nachbeobachtung bis Tag 2, 5, 10, 20, 50 und bis zur Abholung aller. Gegen die Läufe stehen die geschlossene Formel, der Grenzwert des Cox-Schätzers für unendlich viele Container und die mittlere Anfälligkeit der Übrigen
   (nur die Simulation kennt Z).
3. **Zufällige Zuteilung, zweites Merkmal.** Cox nur mit der Maßnahme gegen Cox mit Maßnahme und zweitem Merkmal, 200 Läufe; Regler γ (0 bis 2).
4. **Was stattdessen.** Dieselben Daten wie in Experiment 2 (Lauf 1), gelesen als Anteil, der nach τ Tagen noch da ist, und als mittlere Verweilzeit bis τ, je Gruppe mit Differenz und 95-%-Intervall und gegen die Wahrheit.
   Die Kaplan–Meier-Kurven haben ein punktweises log-log-Greenwood-Band.

Jede Nicht-Zahl-Meldung der App kommt aus einem Status: Die schnelle Cox-Anpassung (`cox_fast` in `hzv_model.py`, vektorisierte Hesse-Matrix, tests halten sie gegen den Kern und eine Nullstellensuche ohne Newton)
gibt einen Schätzer oder `ok = False` mit Grund (ohne Ereignis, Information 0, keine Konvergenz, häufig perfekte Trennung) und nie eine stille Zahl; solche Läufe stehen als Hinweis mit Zahl da und fehlen im Mittel.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus dem Code der Demo (SplitMix64, `hzv_model.py`) mit festen Zufalls-Seeds und stehen eingefroren in `tools/measure_hazard_result.json` (erzeugt mit `tools/measure_hazard.py`).
Welt, wenn nichts anderes steht: Faktor 2, Anfälligkeit gamma-verteilt mit Mittel 1 und Varianz v = 1, Grundrate 0,1 je Tag, n = 2000 Container, alle bis zur Abholung beobachtet, 200 Läufe; der Zufalls-Seed der Läufe ist 12,
Lauf r hat den Seed 12 · 1 000 003 + r. „Mittel ± Standardfehler“ ist das Mittel des Cox-β̂ über die Läufe mit dem Standardfehler dieses Mittels. Die Tabellen unten werden von `tests/test_claims.py` Wort für Wort gegen
`tools/measure_hazard.py` geprüft (siehe „Tests“).

**1. Das Verhältnis der Übrigen sinkt mit der Streuung (Messung 1, Experiment 2).** Wahrheit je Container ln 2 = 0,693 in jeder Phase. Die letzte Spalte ist der Grenzwert des Cox-Schätzers für unendlich viele Container
(numerisch aus der geschlossenen Formel, ohne Zufall).

| Varianz v | früh (Tag 0 bis 3) | mittel (Tag 3 bis 8) | spät (nach Tag 8) | Grenzwerte früh / mittel / spät |
|---|---|---|---|---|
| 0,0 | 0,703 ± 0,006 | 0,684 ± 0,006 | 0,690 ± 0,006 | 0,693 / 0,693 / 0,693 |
| 0,5 | 0,639 ± 0,006 | 0,511 ± 0,006 | 0,302 ± 0,005 | 0,634 / 0,509 / 0,298 |
| 1,0 | 0,592 ± 0,006 | 0,400 ± 0,007 | 0,169 ± 0,005 | 0,589 / 0,405 / 0,164 |
| 2,0 | 0,541 ± 0,006 | 0,297 ± 0,007 | 0,075 ± 0,004 | 0,525 / 0,290 / 0,071 |
| 3,0 | 0,485 ± 0,007 | 0,227 ± 0,008 | 0,039 ± 0,004 | 0,481 / 0,227 / 0,040 |

Ohne Streuung (v = 0) liegt β̂ in allen drei Phasen innerhalb von 2 Standardfehlern der Wahrheit (0,703, 0,684 und 0,690). Mit Streuung sinkt es von Phase zu Phase: bei v = 1 von 0,592 (Hazard-Verhältnis 1,81) über 0,400 (1,49) auf 0,169 (1,18), bei
v = 2 auf 0,541, 0,297 und 0,075. Das Mittel der Läufe liegt in 15 von 15 Zellen (5 Varianzen mal 3 Phasen) innerhalb von 3 Standardfehlern des Grenzwerts; die größte Abweichung beträgt 2,7 Standardfehler (v = 2, Phase früh). Die Hazard-Verhältnisse fallen gegen 1, nicht darunter: Hier beschleunigt die Maßnahme in jedem Container die Abholung.

**Stabilität über Zufalls-Seeds** (Mittel der 200 Läufe je Zufalls-Seed, n = 2000, v = 1 bzw. v = 2): Bei v = 1 liegt das Mittel der späten Phase in den fünf Zufalls-Seeds 1 bis 5 zwischen 0,158 und 0,171 (Mittel 0,164; Grenzwert 0,164; Zufalls-Seed 12 aus Befund 1: 0,169). Bei v = 2 liegt das Mittel der frühen Phase zwischen 0,519 und 0,538 (Mittel 0,528; Grenzwert 0,525; Zufalls-Seed 12 aus Befund 1: 0,541).

v = 1:

| Zufalls-Seed | früh | mittel | spät |
|---|---|---|---|
| 1 | 0,592 | 0,406 | 0,158 |
| 2 | 0,573 | 0,404 | 0,163 |
| 3 | 0,587 | 0,418 | 0,161 |
| 4 | 0,596 | 0,403 | 0,166 |
| 5 | 0,582 | 0,392 | 0,171 |

v = 2:

| Zufalls-Seed | früh | mittel | spät |
|---|---|---|---|
| 1 | 0,531 | 0,296 | 0,069 |
| 2 | 0,519 | 0,292 | 0,072 |
| 3 | 0,524 | 0,286 | 0,072 |
| 4 | 0,538 | 0,293 | 0,059 |
| 5 | 0,529 | 0,298 | 0,070 |

**Andere Stichprobengrößen und Faktoren (v = 1).**

| n | früh | mittel | spät | Läufe ohne Schätzer in der späten Phase |
|---|---|---|---|---|
| 400 | 0,601 ± 0,013 | 0,410 ± 0,014 | 0,167 ± 0,012 | 0 von 200 |
| 1000 | 0,603 ± 0,008 | 0,406 ± 0,010 | 0,177 ± 0,006 | 0 von 200 |
| 2000 | 0,592 ± 0,006 | 0,400 ± 0,007 | 0,169 ± 0,005 | 0 von 200 |
| 3000 | 0,587 ± 0,005 | 0,408 ± 0,005 | 0,169 ± 0,004 | 0 von 200 |

| Faktor | Wahrheit ln f | früh | mittel | spät | Grenzwerte früh / mittel / spät |
|---|---|---|---|---|---|
| 1,5 | 0,405 | 0,352 ± 0,007 | 0,245 ± 0,007 | 0,108 ± 0,004 | 0,351 / 0,251 / 0,103 |
| 2,0 | 0,693 | 0,592 ± 0,006 | 0,400 ± 0,007 | 0,169 ± 0,005 | 0,589 / 0,405 / 0,164 |
| 3,0 | 1,099 | 0,906 ± 0,006 | 0,590 ± 0,007 | 0,241 ± 0,005 | 0,907 / 0,591 / 0,234 |

**2. Ohne Wirkung gibt es keinen Drift.** Faktor 1, Zufalls-Seed 12, 200 Läufe:

| Faktor | Varianz v | früh | mittel | spät |
|---|---|---|---|---|
| 1,0 | 1,0 | 0,002 ± 0,007 | −0,005 ± 0,007 | 0,004 ± 0,004 |
| 1,0 | 2,0 | 0,019 ± 0,007 | 0,000 ± 0,008 | 0,005 ± 0,004 |

Der Grenzwert ist in jeder Phase genau 0. Das Mittel der Läufe liegt in 6 von 6 Zellen (zwei Varianzen mal 3 Phasen) innerhalb von 3 Standardfehlern von 0; die größte Abweichung beträgt 2,8 Standardfehler (v = 2, Phase früh). **Ohne Wirkung gibt es keinen Drift:** Wo die Maßnahme keine Rate ändert, ist nichts da, was die Auswahl der Übrigen verzerren könnte (eigene Lesart, siehe Hernán 2010 in der Tabelle unten).

**3. Warum? Die mittlere Anfälligkeit der Übrigen** (v = 1, Faktor 2, 200 Läufe, Mittel über die Läufe der mittleren Anfälligkeit Z unter den Containern, die bei Zeit t noch da sind; in Klammern die Formel
1/(1 + vλ f^x t), das Verhältnis gegen HR(t)/f). Bei gamma-verteilter Anfälligkeit ist das Hazard-Verhältnis der Übrigen genau f mal dem Verhältnis der mittleren Anfälligkeit der Übrigen.

| Zeit t (Tage) | mittleres Z der Übrigen, Kontrolle (Formel) | mittleres Z der Übrigen, Maßnahme (Formel) | Verhältnis (Formel HR(t) / f) |
|---|---|---|---|
| 2 | 0,831 (0,833) | 0,713 (0,714) | 0,858 (0,857) |
| 5 | 0,664 (0,667) | 0,499 (0,500) | 0,752 (0,750) |
| 10 | 0,497 (0,500) | 0,333 (0,333) | 0,671 (0,667) |
| 20 | 0,332 (0,333) | 0,199 (0,200) | 0,602 (0,600) |

Die Mittel der Läufe weichen von der Formel um höchstens 0,003 (Kontrolle), 0,002 (Maßnahme) und 0,004 (Verhältnis) ab. Die Maßnahme-Gruppe hat zu jedem Zeitpunkt ab Tag 2 eine niedrigere mittlere Anfälligkeit unter den Übrigen als die Kontrolle (Verhältnis unter 1): Das ist die gemessene Seite des Mechanismus.

**4. Ein einziges Hazard-Verhältnis hängt an der Länge der Nachbeobachtung (v = 1, Faktor 2).** Cox auf denselben 2000 Containern, bei Tag F zensiert (administrativ, für alle gleich):

| Nachbeobachtung | Cox β̂ (Mittel ± Standardfehler) | Hazard-Verhältnis exp(β̂) | Grenzwert des Hazard-Verhältnisses |
|---|---|---|---|
| bis Tag 2 | 0,618 ± 0,008 | 1,86 | 1,85 |
| bis Tag 5 | 0,548 ± 0,005 | 1,73 | 1,73 |
| bis Tag 10 | 0,489 ± 0,004 | 1,63 | 1,63 |
| bis Tag 20 | 0,438 ± 0,004 | 1,55 | 1,55 |
| bis Tag 50 | 0,391 ± 0,003 | 1,48 | 1,48 |
| bis zum Ende | 0,349 ± 0,003 | 1,42 | 1,41 |

Das einzelne Hazard-Verhältnis fällt von 1,86 (Nachbeobachtung bis Tag 2) auf 1,42 (bis zur Abholung aller Container), obwohl jeder Container zu jedem Zeitpunkt den Faktor 2 erfährt. Die Grenzwerte stimmen mit den Mitteln überein (größter Abstand 0,6 Standardfehler). Jede dieser Zahlen stimmt mit dem Grenzwert des Cox-Schätzers für die jeweils noch vorhandenen Container und die jeweilige Beobachtungsdauer überein; sie unterscheiden sich, weil die Mischung der Übrigen eine andere ist.

**5. Zufällige Zuteilung, zweites Merkmal (Experiment 3, n = 2000, Faktor 2, 200 Läufe).** Rate 0,1 · 2^x · e^(γz), z standardnormal, x zufällig und von z unabhängig:

| γ | Cox nur mit x (Mittel ± Standardfehler) | Grenzwert nur mit x | Cox mit x und z | Koeffizient von z (Wahrheit γ) |
|---|---|---|---|---|
| 0,0 | 0,695 ± 0,003 | 0,693 | 0,695 ± 0,003 | 0,003 |
| 0,5 | 0,583 ± 0,003 | 0,582 | 0,695 ± 0,003 | 0,506 |
| 1,0 | 0,438 ± 0,003 | 0,438 | 0,695 ± 0,003 | 1,009 |
| 1,5 | 0,338 ± 0,003 | 0,339 | 0,694 ± 0,003 | 1,510 |
| 2,0 | 0,272 ± 0,003 | 0,274 | 0,694 ± 0,003 | 2,012 |

Bei γ = 0 sind beide Schätzer gleich (0,695 und 0,695). Mit wachsendem γ fällt der Cox-Schätzer nur mit x auf 0,583, 0,438, 0,338, 0,272, während der Schätzer mit x und z in allen fünf Zeilen zwischen 0,694 und 0,695 bleibt (Wahrheit je Container 0,693); der Koeffizient von z liegt nahe bei γ (Spalte rechts; im Mittel um bis zu 0,012 darüber, ein kleines systematisches Übermaß bei n = 2000). Der Grenzwert nur mit x stimmt mit den Mitteln überein (größter Abstand 0,5 Standardfehler). Das gilt bei zufälliger Zuteilung, also ohne Störfaktor. Ohne Wirkung der Maßnahme (Faktor 1, γ = 1): Cox nur mit x −0,001 ± 0,003, mit x und z −0,002 ± 0,003.

**6. Was stattdessen (Experiment 4, v = 1, Faktor 2, n = 2000, 200 Läufe mit je einem Datensatz).** Differenz Maßnahme minus Kontrolle; die Wahrheit ist aus der geschlossenen Formel gerechnet:

| τ (Tage) | Differenz Anteil noch da (Mittel ± Standardfehler) | Wahrheit | Intervall enthält die Wahrheit | Differenz mittlere Verweilzeit (Tage) | Wahrheit | Intervall enthält die Wahrheit |
|---|---|---|---|---|---|---|
| 2 | −11,88 ± 0,15 Prozentpunkte | −11,90 Prozentpunkte | 182 von 200 | −0,142 ± 0,002 | −0,141 | 187 von 200 |
| 5 | −16,58 ± 0,16 Prozentpunkte | −16,67 Prozentpunkte | 184 von 200 | −0,589 ± 0,006 | −0,589 | 185 von 200 |
| 10 | −16,62 ± 0,16 Prozentpunkte | −16,67 Prozentpunkte | 187 von 200 | −1,435 ± 0,013 | −1,438 | 182 von 200 |
| 20 | −13,45 ± 0,14 Prozentpunkte | −13,33 Prozentpunkte | 189 von 200 | −2,941 ± 0,025 | −2,939 | 185 von 200 |

Die Mittel der Differenzen über die Läufe liegen bei allen vier τ innerhalb von 0,9 (Anteil) bzw. 0,6 (mittlere Verweilzeit) Standardfehlern der Wahrheit. Das 95-%-Intervall enthält die Wahrheit in den Zellen mit 200 Läufen in 182 bis 189 von 200 Läufen (Anteil) bzw. 182 bis 187 von 200 (mittlere Verweilzeit); über die fünf Zufalls-Seeds bei τ = 10 in 957 von 1000 bzw. 953 von 1000 Läufen (95,7 % und 95,3 %). Das einzelne Cox-Hazard-Verhältnis derselben Datensätze über alle Zeiten (es hängt nicht von τ ab) ist im Mittel der Läufe 1,42 (je Container: 2); die Differenz des Anteils dagegen hängt von τ ab (Tabelle), und zu jedem τ liefert die Formel ihre Wahrheit.

Überdeckung bei τ = 10 über fünf Zufalls-Seeds (je 200 Läufe):

| Zufalls-Seed | Anteil noch da: Intervall enthält die Wahrheit | mittlere Verweilzeit: Intervall enthält die Wahrheit |
|---|---|---|
| 1 | 192 von 200 | 194 von 200 |
| 2 | 192 von 200 | 190 von 200 |
| 3 | 194 von 200 | 189 von 200 |
| 4 | 188 von 200 | 190 von 200 |
| 5 | 191 von 200 | 190 von 200 |

**7. Zwei Klassen von Hand (Experiment 1).** Standardbeispiel (Raten 1 und 4 je Tag, je 50 %, Faktor 2): Das Verhältnis der marginalen Raten startet bei 2, hat seinen Tiefpunkt 1,47 bei Tag 0,445 (Gitter 0,001 Tage) und liegt bei Tag 6 wieder bei 2,00. Raten 1 und 1,5: Tiefpunkt 1,87 bei Tag 2,248, bei Tag 6 1,96.

**Handbeispiel (Raten 1 und 4 je Tag, je 50 %, Maßnahme verdoppelt jede Rate; von Hand gerechnet):**

| Zeit t (Tage) | Rate Kontrolle | Rate Maßnahme | Verhältnis |
|---|---|---|---|
| 0 | 2,5000 | 5,0000 | 2,0000 |
| 0,5 | 1,5473 | 2,2846 | 1,4765 |
| 1 | 1,1423 | 2,0148 | 1,7639 |
| 2 | 1,0074 | 2,0000 | 1,9853 |
| 3 | 1,0004 | 2,0000 | 1,9993 |

Bei t = 0 sind die Raten 1/2 · 1 + 1/2 · 4 = 5/2 und 1/2 · 2 + 1/2 · 8 = 5, das Verhältnis ist 2 für jede Aufteilung der Klassen. Danach sinkt es (Tiefpunkt 1,47 bei Tag 0,445) und kehrt auf 2 zurück, weil zuletzt nur die
langsamere Klasse übrig ist. Das ist ein Unterschied zur gamma-verteilten Anfälligkeit (Experiment 2): Bei nur zwei Klassen kehrt das Verhältnis auf f zurück, bei gamma-verteilter Anfälligkeit fällt es gegen 1.

**Zahl → Test.**

| Zahlen | Wo geprüft |
|---|---|
| Handtabelle der zwei Klassen, Brüche bei t = 0, Tiefpunkt, Formelwerte (HR(5,5) = 1,48), geschlossene RMST für v = 0 und 1 | `tests/test_model.py`, auf 5e-5 bis 1e-12 |
| Marginale Rate, Gamma-Formeln, mittlere Lebenszeit, Gamma-Zufallszahlen, Grenzwert des Cox-Schätzers, schnelle Cox-Anpassung, Kaplan–Meier und Greenwood | `tests/test_oracle_hazard_verhaeltnis.py` (andere Rechenwege, siehe „Tests“) |
| Eigenschaften über das Reglerraster (v = 0 konstant; für 144 Zellen v > 0, f > 1: fällt streng, bleibt über 1; Überlebenskurven kreuzen sich nicht; Grenzwert fällt mit Phase und Streuung) | `tests/test_eigenschaften_hazard.py` |
| Jede Zahl der Verdikt-Texte der App | `tests/test_verdicts.py` an handgesetzten Werten |
| Messung 1 bis 7 (alle Tabellen und Prosa-Zahlen oben) | eingefroren in `tools/measure_hazard_result.json`; `tests/test_claims.py` vergleicht jede Tabelle wörtlich und rechnet die Zellen Standard (v = 1), Experiment 3 (γ = 1), Experiment 4 (τ = 10) und die beiden Klassen-Beispiele neu (Band 1e-6; Zählungen exakt) |
| Zellen v = 0, 0,5, 2, 3, Seeds, n = 400, 1000, 3000, Faktoren, andere γ und τ | nur eingefroren, nicht im Test neu gerechnet |

**Erste Messung zum Vergleich** (vor dem Bau der Demo mit einem einfachen Skript gemessen, NumPy-Zufall, 200 Läufe): v = 0: früh 0,695, spät 0,694; v = 1: früh 0,589, mittel 0,406, spät 0,174 ± 0,005;
v = 2: früh 0,519, spät 0,074 ± 0,004; zufällige Zuteilung, Wirkung 0,70: Cox nur mit x 0,441 ± 0,003, mit x und z 0,699 ± 0,003. Die Werte dieser Demo weichen zufallsbedingt ab (anderer Zufallsgenerator,
Faktor 2 statt Wirkung 0,70 bei der zufälligen Zuteilung); siehe „Befunde und Korrekturen gegenüber der ersten Messung“.

## Der Streit um das Hazard-Verhältnis

Die Demo stellt die Positionen **als Streit** dar, nicht als Urteil. Die Tabelle gibt sie in eigenen Worten wieder, mit der **Lesestufe** der Quelle. *Volltext selbst gelesen:* Text aus der PDF gelesen.
*Volltext über Abruf:* Volltextseite vom Abrufwerkzeug zusammengefasst, Rohtext nicht selbst gesehen; Zahlen und Satzbau sind vor einer Veröffentlichung im PDF gegenzulesen. *Abstract:* nur Kurzfassung.

| Position | Lesestufe | Aussage in eigenen Worten | Was diese Demo dazu zeigt und was nicht |
|---|---|---|---|
| **Hernán (2010)**, Epidemiology 21(1):13–15 (Kommentar) | Volltext über Abruf | Ein über die Nachbeobachtung gemitteltes HR hängt von deren Dauer ab. Periodenspezifische HR enthalten eine eingebaute Auswahlverzerrung: Anfälligere der behandelten Gruppe scheiden früher aus, späte HR sinken Richtung oder unter 1. Empfohlen: angepasste Überlebenskurven und eine Reihe gemittelter HR. Der Text enthält weder eine Simulation noch eine Herleitung. *Eigene Lesart (nicht wörtlich bei Hernán belegt):* Die Verzerrung setzt eine Wirkung der Behandlung voraus (etwa frühe Schädigung); bei strenger Nullwirkung gibt es sie nicht; bei Ying und Xu steht dazu sinngemäß, die Gruppen nach dem Überleben seien nur unter der scharfen Nullhypothese identisch (nicht wörtlich geprüft). | Der Mechanismus tritt in Experiment 2 auf (HR der Übrigen sinkt gegen 1), die Abhängigkeit von der Nachbeobachtung steht in Befund 4, und ohne Wirkung (Faktor 1) gibt es keinen Drift (Befund 2). Nicht gezeigt: ein Fall unter 1 (hier beschleunigt die Maßnahme in jedem Container die Abholung) und wie stark der Effekt in echten Studien ist. |
| **Abrahamowicz u. a. (2025)**, Eur. J. Epidemiol. 40(6):611–629 (Simulationen, Fallstudien) | Volltext über Abruf (rund 100 000 von 122 000 Zeichen; Referenzen und Zusatzmaterial fehlen) | Bestätigen den Mechanismus (Erschöpfung Anfälliger, Verzerrung zur Null). Halten ihn **für die nachgebaute WHI-Studie unter den geprüften Annahmen** für eine sehr unwahrscheinliche Erklärung der beobachteten Kreuzung; bei sehr starker Anfälligkeit (HR der Anfälligkeit 25) bleibt ein Bias bis etwa 58 %, bei HR der Anfälligkeit höchstens 5 unter 20 % (Gesamt-HR). Nach Ansicht der Autoren habe Hernán seine Behauptung weder algebraisch noch per Simulation gestützt. Ihr Gesamtschluss, nur für die nachgebaute WHI-Konstellation unter den geprüften Annahmen: Als Alternativen zur Anfälligkeit nennen sie Nichteinnahme und echte zeitabhängige Wirkung; Bedenken gegen das HR könnten übertrieben sein. | Die Größe des Effekts hängt an Stärke und Streuung der Anfälligkeit (Experiment 2: Regler v). Die Welt der Demo (gamma-verteilte Anfälligkeit, kein WHI-Nachbau) ist nicht ihre; die Zahlen sind nicht vergleichbar, und die Demo entscheidet nicht, ob echte Daten so streuen. |
| **Ying und Xu**, arXiv 2307.11971 (v2, Okt. 2023), „On Defense of the Hazard Ratio“ | Volltext selbst gelesen | Streit um den **Kausalbegriff**, nicht um den Mechanismus. Ihr Hauptergebnis: Die Hazard-Funktion habe in den drei Rahmen von Rubin, Robins und Pearl eine kausale Deutung. Nach der strengen Definition von Rubin sei das HR über Risikomengen nicht kausal (außer bei der scharfen Nullhypothese), nach Robins und Pearl schon. Die Erschöpfung der Anfälligen sei die bekannte unbeobachtete Heterogenität. Ob ein späteres niedrigeres Hazard Wirkung oder Auswahl ist, lasse sich an den Überlebenskurven prüfen (im reinen Auswahlfall kreuzen sie sich nicht; kreuzen sie sich, deute das auf eine echte Wirkung; zusätzlich können Sensitivitätsanalysen nötig sein); man solle den ganzen Verlauf des HR ansehen, nicht einen späten Zeitpunkt allein. | Die Überlebenskurven kreuzen sich in dieser Welt nicht (Experiment 1 und Formel in Experiment 2), und die Wirkung je Container ist konstant: Das Absinken ist hier reine Auswahl. Zur Kausalfrage selbst sagt die Demo nichts. |
| **Royston und Parmar (2013)**, BMC Med. Res. Methodol. 13:152 | Volltext über Abruf (rund 200 000 von 218 000 Zeichen) | Die mittlere Lebenszeit bis τ (RMST) als Alternative: Verlust an Lebenszeit bis τ, gültig ohne Proportionalität, relatives und absolutes Maß. Für die Hauptanalyse legen sie ein **vorab festgelegtes** τ zugrunde; in ihrer Simulation hat der RMST-Test bei einem nicht proportionalen Szenario eine höhere Trennschärfe als der Log-Rank-Test. Nachteil laut Tabelle 5: τ muss gewählt werden. | Experiment 4 liest dieselben Daten als Überlebenswahrscheinlichkeit und mittlere Verweilzeit bis τ. Die Trennschärfe vergleicht diese Demo nicht. |

Die in Abrahamowicz u. a. (2025) genannte Übersicht über 27 große Studien (Strobel u. a. 2023) wurde nicht gefunden und ist hier nicht verwendet; eine „GMDS-Auswertung von 28 Studien“ ist nicht belegt und kommt nicht vor.
Hernán (2010), Abrahamowicz u. a. (2025) und Royston und Parmar (2013) sind nur über Abruf-Zusammenfassungen gelesen. Hernáns Empfehlung nennt die mittlere Lebenszeit nicht (im gelesenen Text nicht gefunden); die Demo übernimmt diese Alternative von Royston und Parmar (2013). Marsaglia und Tsang (2000) und Greenwood (1926) sind
aus dem Gedächtnis genannt, die Originale sind nicht eingesehen.

## Ehrliche Grenzen

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

**Weitere Einschränkungen (Negativbefunde bleiben stehen)**
- Die Streuung der Anfälligkeit, bei der der Effekt auftritt, ist ein Regler (v bis 3), keine Schätzung aus Daten: Zwischen v = 0,5 und v = 3,0 liegt das Hazard-Verhältnis exp(β̂) der späten Phase im Mittel der Läufe zwischen 1,35 und 1,04 (Faktor je Container 2). Welches v zu echten Hafendaten passt, zeigt die Demo nicht.
- Der Grenzwert des Cox-Schätzers ist eine Rechnung unter dem Modell dieser Demo (gleiche Gruppengrößen, alle Container bis zur Abholung oder bis zum Phasenende beobachtet); er gilt nicht für andere Zuteilungsverhältnisse
  oder zufällige Zensierung.
- Die Überdeckung der 95-%-Intervalle in Experiment 4 schwankt: im Zufalls-Seed 12 liegt sie in den vier Zellen mit 200 Läufen bei 182 bis 189 von 200 (91,0 % bis 94,5 %), in den Zufalls-Seeds 1 bis 5 bei τ = 10 bei 188 bis 194 von 200 (Anteil) bzw. 189 bis 194 von 200 (mittlere Verweilzeit). Der Binomial-Standardfehler eines Anteils um 95 % ist bei 200 Läufen 1,5 Prozentpunkte; ein Unter- oder Überschreiten von 95 % zeigen diese Messungen nicht sicher.
- Ein einzelnes τ ist selbst eine Wahl: Hernán (2010) weist darauf hin, dass auch eine einzelne Überlebenswahrscheinlichkeit irreführen kann (Volltext über Abruf). Die Demo zeigt τ von 1 bis 40 Tagen am Regler, aber keine
  Messung, wie sich ein nach den Daten gewähltes τ auf ein Testniveau auswirkt.
- Die Maßnahme beschleunigt in jedem Container die Abholung (Faktor größer als 1, oder 1 ohne Wirkung): Ein Hazard-Verhältnis unter 1 kommt in dieser Welt nicht vor.

## Befunde und Korrekturen gegenüber der ersten Messung

Die erste Messung ist ein einfaches Skript, das vor dem Bau der Demo lief (NumPy-Zufall, 200 Läufe); die Demo rechnet mit SplitMix64. Dieser Abschnitt nennt, wo die Demo von der ersten Messung und der ursprünglichen Erwartung abweicht.

- **Zahlen (Experiment 2):** erste Messung (NumPy-Zufall, v = 1): früh 0,589, mittel 0,406, spät 0,174 ± 0,005. Hier: 0,592, 0,400 und 0,169 (± 0,005); die späte Phase liegt über fünf weitere Zufalls-Seeds zwischen 0,158 und 0,171 (Grenzwert 0,164): Der Wert 0,174 der ersten Messung war ein Ausschlag nach oben. Bei v = 2 nannte die erste Messung früh 0,519, mittel 0,272, spät 0,074; hier 0,541, 0,297 und 0,075, wobei die frühe Phase in den fünf weiteren Zufalls-Seeds zwischen 0,519 und 0,538 liegt (Grenzwert 0,525). Es gilt die Zahl dieser Demo.
- **Zahlen (zufällige Zuteilung, Experiment 3):** erste Messung Cox nur mit x 0,441 ± 0,003 und mit x und z 0,699 ± 0,003 (Wirkung 0,70 je Container); hier 0,438 ± 0,003 und 0,695 ± 0,003 (Wirkung ln 2 = 0,693 je Container, weil die App den Faktor 2 statt 0,70 nutzt).
- **Prüfung der Formel:** Die erste Messung prüfte die geschlossene Formel an einem Punkt (HR(5,5) = 1,48, ln = 0,39, gegen die mittlere Phase 0,406). Eine Punktprüfung ist grob, weil Cox in einem Fenster über viele Zeiten mittelt; hier steht stattdessen der Grenzwert des Cox-Schätzers im Fenster (Integration über die Formel), gegen den die Läufe in 15 Zellen gezählt werden (Befund 1). Der Punkt der ersten Messung bleibt als Handwert in `tests/test_model.py`.
- **Neu gegenüber der ersten Messung:** die Zellen ohne Wirkung (Befund 2), die gemessene mittlere Anfälligkeit der Übrigen (Befund 3), die Abhängigkeit des einzelnen Hazard-Verhältnisses von der Nachbeobachtung (Befund 4), v = 3 und andere Faktoren und Stichprobengrößen.
- **Ursprünglich offen: „RMST-Differenz gegen Hazard bei denselben Daten“.** Gebaut als Experiment 4 (Befund 6), gemessen als Differenz und Überdeckung des Intervalls, nicht als Trennschärfe-Vergleich zweier Tests.
- **Offen bleibt:** Die Gamma-Verteilung der Anfälligkeit ist eine Annahme; ob echte Hafendaten so streuen, zeigt die Demo nicht.
- **Schnelle Cox-Anpassung:** Zuerst war nur der Kern vorgesehen. Die Schleife des Kerns über die Ereigniszeiten ist für 200 Läufe mit je neun Anpassungen und n = 2000 zu langsam für eine App; `cox_fast` (vektorisierte Hesse-Matrix) ersetzt sie für die Läufe und wird in den Tests gegen den Kern (40 Instanzen, bis 1e-9) und gegen eine Nullstellensuche ohne Newton gehalten. Der Kern bleibt für die Einzelanpassung mit Status zuständig (Handbeispiele, Tests).
- **Streit als Streit, ohne Urteil:** Die Positionen stehen mit ihren Lesestufen nebeneinander. Die App und diese README urteilen nicht über das Hazard-Verhältnis selbst, sie sagen nur „in dieser Annahme verändert es sich so“ (ein Test sucht Urteilswörter in beiden Texten).
- **Regler der Klassen:** Die Raten der Klasse A (0,2 bis 1,0) und der Klasse B (1,5 bis 10) überlappen sich nicht, damit B immer die schnellere Klasse ist; die erste Messung nannte nur das Beispiel 1 und 4 mit Anteil 50 %.
- **Kern:** Die Kopie enthält die Verbesserung vom 2026-11-10 (Trennungserkennung auch bei einem einzigen Ereignis, große Standardfehler oder |β̂| > 20 gelten als nicht konvergiert); die Demo braucht keinen eigenen Wrapper dafür.
- **Kein PDF-Export:** Die App zeigt alle Zahlen auf der Seite.

## Tests

163 Tests (`pytest --collect-only -q tests`), alle grün. Den größten Teil der Laufzeit machen die AppTest-Läufe (`test_app.py`, `test_layout.py`), der Orakel-Test mit der großen Simulation und die Neuberechnung der Messzellen (`test_claims.py`) aus.

| Datei | Tests | Inhalt |
|---|---|---|
| `test_core.py` | 13 | kopierte Kernfunktionen: Kaplan–Meier-Handbeispiel, Greenwood und mittlere Lebenszeit von Hand, Eingabeprüfung, Cox-Handbeispiel, Eintritt, Trennung (auch bei einem einzigen Ereignis) und Singularität als Status, SplitMix64-Referenzwert; Vergleich der Funktionstexte mit der Quelle `uz_core.py`, falls vorhanden |
| `test_model.py` | 24 | Einheiten am Handbeispiel der zwei Klassen mit Brüchen, Tiefpunkt, geschlossene Formel an Handwerten und Randfällen (v → 0, extrem große v), mittlere Lebenszeit, Grenzwert (v = 0, Faktor 1, leeres Fenster, Quadratur), schnelle Cox-Anpassung (Handbeispiel, Status), Zerlegung in Phasen und Nachbeobachtung von Hand, Spalten, Greenwood-Standardfehler, Experiment 4 |
| `test_oracle_hazard_verhaeltnis.py` | 15 | **Orakel** mit anderen Rechenwegen: numerische Ableitung der Mischung und Optimierung des Tiefpunkts; Integration über die Gamma-Dichte (`scipy.integrate.quad`) für Überlebensfunktion, Rate, Verhältnis und mittlere Lebenszeit; Kolmogorow–Smirnow-Test und Laplace-Transformierte der Gamma-Zufallszahlen; Grenzwert des Cox-Schätzers gegen adaptive Quadratur mit `brentq`, gegen eine Simulation mit 300 000 Containern und gegen eine Gauß–Hermite-Mischung; mittlere Anfälligkeit der Übrigen gegen die Formel; schnelle Cox-Anpassung gegen den Kern und gegen eine Nullstellensuche in reinen Schleifen; Kaplan–Meier, Greenwood und Fläche gegen Aufzählung mit Brüchen |
| `test_eigenschaften_hazard.py` | 10 | Eigenschaften über das Reglerraster: v = 0 konstant, v > 0 fällt streng und bleibt über 1 (144 Zellen), ohne Wirkung gleich 1, Überlebenskurven kreuzen sich nicht, Klassenmittel, Verhältnis bei t = 0 für 81 Kombinationen, Grenzwerte monoton in Phase, Faktor, Streuung und Nachbeobachtung |
| `test_scenario.py` | 7 | Gamma-Zufallszahlen (Momente, Reproduzierbarkeit, v = 0 ohne Zufallsverbrauch, Fehler), Welten von Experiment 2 und 3, Lauf-Seeds |
| `test_edge_cases.py` | 8 | kleinste und größte Reglerwerte, Phase ohne Container, Hinweistexte, ungültige Eingaben (`SurvivalError`), Klassen ohne Tiefpunkt, Experiment 4 an den Zeiträndern |
| `test_verdicts.py` | 15 | jedes Verdikt und jeder Hinweistext an handgesetzten Werten: Schwellen, Zahlen, Zweige |
| `test_claims.py` | 22 | Tabellen der README wörtlich gegen `tools/measure_hazard.py`, Prosa-Zahlen, Zählungen, Neuberechnung ausgewählter Zellen, Annahmen-Text der App in der README, Formulierungen zum Streit |
| `test_presets.py` | 8 | Schnellstart-Szenarien vollständig, in den Grenzen, Permalink-Begrenzung, Aussagen der Hilfetexte gegen die Rechnung |
| `test_layout.py` | 22 | Schnellstart in der Seitenleiste, fester Kopf, Hervorhebung des aktiven Szenarios, jede Regleränderung hebt sie auf |
| `test_app.py` | 16 | AppTest: jeder Button, jeder Regler an Minimum und Maximum, jede Varianz des Rasters mit ihrem Zweig, Faktor 1, Eckfall-Matrix, kleinste Stichprobe, Zufalls-Seed-Ränder, Annahme sichtbar, Streit ohne Urteil, fünf Diagramme |
| `test_footer.py` | 2 | Footer in App und README |
| `test_runtime_imports.py` | 1 | Laufzeit-Importe gegen `requirements.txt` |

Die Orakel-Tests brauchen SciPy (`requirements-dev.txt`, nicht in der Laufzeit); fehlt es, überspringen sie sich (`pytest.importorskip`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Streit, Annahmen, Formeln |
| `hzv_core.py` | Kopie des gemeinsamen Kerns `uz_core.py` der Überlebenszeit-Linie (Stand 2026-11-10; SplitMix64, Eingabeprüfung, Kaplan–Meier mit Greenwood-Band, mittlere Lebenszeit, Cox) |
| `hzv_scenario.py` | Welten (Zwei Klassen, Gamma-Anfälligkeit, zweites Merkmal), Gamma-Zufallszahlen aus SplitMix64 |
| `hzv_model.py` | Gesetze (Mischung, Gamma), Grenzwert des Cox-Schätzers, schnelle Cox-Anpassung, Studien, Auswertung von Experiment 4 |
| `hzv_evaluation.py` | Kennzahlen, Texte und Verdikte als reine Funktionen |
| `hzv_constants.py` | Regler-Grenzen, feste Annahmen, Schnellstart-Szenarien, Text „Annahmen, Grenzen und bekannte Probleme“ |
| `hzv_presets.py` | Permalink (Begrenzen), Schnellstart in der Seitenleiste mit hervorgehobenem aktiven Szenario, Zufalls-Seed-Button |
| `hzv_layout.py` | fester Kopf (Stylesheet) |
| `hzv_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/measure_hazard.py`, `tools/measure_hazard_result.json` | Messskript und eingefrorene Zahlen dieser README |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly); dev zusätzlich pytest und scipy (nur Orakel-Tests) |
| `tests/` | Kern, Modell, Orakel, Eigenschaften, Szenario, Randfälle, Verdikte, Claims, Schnellstart, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Nachbau der Women's-Health-Initiative-Studie** und jeder Vergleich mit den Zahlen von Hernán (2010) oder Abrahamowicz u. a. (2025): Die Demo hat andere Welten und vergleicht keine Zahlen.
- **Kreuzung unter 1, frühe Schädigung, Nichteinnahme, echte biologische Zeitabhängigkeit:** nicht gebaut (die Wirkung je Container ist konstant und beschleunigt die Abholung).
- **Kausale Diagramme und Störfaktoren:** die Demo teilt zufällig zu und zeigt keine Auswahl in die Gruppen; Auswahl und Kausalbegriff gehören in die Linie der kausalen Diagramme (nicht Teil dieser Demo).
- **Zufällige Zensierung und informatives Zensieren:** hier nur feste Zensierzeiten; die Annahme prüft die [informatives-zensieren-demo](https://github.com/sebastian-hanisch/informatives-zensieren-demo) der Linie.
- **Vergleich der Trennschärfe von RMST-Test und Log-Rank-Test, Niveau bei nach den Daten gewähltem τ:** nicht gemessen (zur mittleren Lebenszeit beim Gruppenvergleich siehe [log-rank-demo](https://github.com/sebastian-hanisch/log-rank-demo)).
- **PDF-Export.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; die in Streamlit Cloud eingestellte Python-Version steht noch nicht fest und ist beim Deploy hier einzutragen.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy; Orakel-Tests mit SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Überlebenszeit: zensierte Zeiten lesen](https://sebastianhanisch.net/konzepte-ueberlebenszeit.html).
