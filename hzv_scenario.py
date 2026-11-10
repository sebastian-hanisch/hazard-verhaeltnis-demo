"""Szenario: Container mit unbeobachteter Anfälligkeit und einer zufällig zugeteilten Maßnahme.

Die Welten (DEMO-PLAYBOOK §1b: eine Regel je Funktion, Zufall nur über SplitMix64):
  * Experiment 2 (`FrailtyWorld`): Rate eines Containers = λ · Z · f^x mit x ∈ {0, 1} zufällig (je 50 %), Anfälligkeit Z gamma-verteilt mit
    Mittel 1 und Varianz v, Faktor f der Maßnahme. Die Verweilzeit ist bei gegebenem Z exponentiell verteilt: T = E / (Rate), E ~ Exponential(1).
  * Experiment 3 (`RandomWorld`): Rate = λ · f^x · exp(γ·z) mit z standardnormal (zweites Merkmal), sonst wie oben.
  * Experiment 1 (`ClassWorld`): zwei Klassen mit festen Raten (rechnerisch, ohne Zufall).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import hzv_constants as C
from hzv_core import SplitMix64


@dataclass(frozen=True)
class FrailtyWorld:
    """Welt von Experiment 2 und 4; alles, was die Daten bestimmt, steht hier (gehört in jeden Cache-Schlüssel)."""

    n: int = C.DEFAULT_N
    v: float = C.DEFAULT_V
    factor: float = C.DEFAULT_FACTOR
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class RandomWorld:
    """Welt von Experiment 3."""

    n: int = C.DEFAULT_N
    gamma: float = C.DEFAULT_GAMMA
    factor: float = C.DEFAULT_FACTOR
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class ClassWorld:
    """Zwei Klassen von Containern (Experiment 1): Raten je Tag, Anteil der Klasse B, Faktor der Maßnahme."""

    rate_a: float = C.DEFAULT_RATE_A
    rate_b: float = C.DEFAULT_RATE_B
    share_b: float = C.DEFAULT_SHARE_B
    factor: float = C.DEFAULT_FACTOR


@dataclass(frozen=True)
class Sample:
    time: np.ndarray   # Verweilzeit bis zur Abholung (alle beobachtet, keine zufällige Zensierung)
    x: np.ndarray      # 1: Maßnahme, 0: Kontrolle
    z: np.ndarray      # Anfälligkeit (Experiment 2) bzw. zweites Merkmal (Experiment 3); nur der Simulation bekannt


def run_seed(seed: int, run: int) -> int:
    """Seed des Laufs `run` aus dem Zufalls-Seed der Einstellung: eigener, fester Strom je Lauf."""
    return seed * 1_000_003 + run


def gamma_variates(rng: SplitMix64, size: int, shape: float) -> np.ndarray:
    """Gamma(shape, 1)-verteilte Zufallszahlen nach Marsaglia und Tsang (2000); für shape < 1 mit der Verstärkung G(a) = G(a + 1) · U^(1/a).
    Alle Zufallszahlen kommen aus dem SplitMix64-Strom; die Zahl der Runden hängt nur vom Strom ab (reproduzierbar)."""
    if shape <= 0:
        raise ValueError("Die Form muss positiv sein.")
    if shape < 1.0:
        g = gamma_variates(rng, size, shape + 1.0)
        u = rng.uniform(size)
        return g * (1.0 - u) ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    out = np.empty(size)
    todo = np.arange(size)
    for _ in range(10_000):
        if todo.size == 0:
            return out
        x = rng.normal(todo.size)
        u = rng.uniform(todo.size)
        v = (1.0 + c * x) ** 3
        with np.errstate(invalid="ignore", divide="ignore"):
            accept = (v > 0) & (np.log1p(-u) < 0.5 * x * x + d - d * v + d * np.log(np.where(v > 0, v, 1.0)))
        out[todo[accept]] = d * v[accept]
        todo = todo[~accept]
    raise RuntimeError("gamma_variates: keine Annahme nach 10 000 Runden")  # pragma: no cover


def frailty_values(rng: SplitMix64, size: int, v: float) -> np.ndarray:
    """Anfälligkeit Z: gamma-verteilt mit Mittel 1 und Varianz v (Form 1/v, Skala v); bei v = 0 konstant 1 (verbraucht dann keine Zufallszahlen)."""
    if v < 0:
        raise ValueError("Die Varianz darf nicht negativ sein.")
    if v == 0:
        return np.ones(size)
    return v * gamma_variates(rng, size, 1.0 / v)


def simulate_frailty(world: FrailtyWorld, run: int = 0) -> Sample:
    """Ein Datensatz der Welt von Experiment 2. Zugstrom je Lauf: erst n Zuteilungen (gleichverteilt), dann die Anfälligkeiten, dann n Exponentialzufälle."""
    rng = SplitMix64(run_seed(world.seed, run))
    x = (rng.uniform(world.n) < 0.5).astype(float)
    z = frailty_values(rng, world.n, world.v)
    e = rng.exponential(world.n)
    t = e / (C.BASE_RATE * z * world.factor**x)
    return Sample(time=np.maximum(t, 1e-9), x=x, z=z)


def simulate_random(world: RandomWorld, run: int = 0) -> Sample:
    """Ein Datensatz der Welt von Experiment 3. Zugstrom je Lauf: Zuteilungen, zweites Merkmal (normal), Exponentialzufälle."""
    rng = SplitMix64(run_seed(world.seed, run))
    x = (rng.uniform(world.n) < 0.5).astype(float)
    z = rng.normal(world.n)
    e = rng.exponential(world.n)
    t = e / (C.BASE_RATE * world.factor**x * np.exp(world.gamma * z))
    return Sample(time=np.maximum(t, 1e-9), x=x, z=z)
