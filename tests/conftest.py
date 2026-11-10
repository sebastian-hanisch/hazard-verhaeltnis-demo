"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from hzv_scenario import ClassWorld  # noqa: E402


@pytest.fixture
def hand_world():
    """Das Handbeispiel von Experiment 1: zwei Klassen zu je 50 % mit den Raten 1 und 4 je Tag, die Maßnahme verdoppelt jede Rate."""
    return ClassWorld(rate_a=1.0, rate_b=4.0, share_b=0.5, factor=2.0)
