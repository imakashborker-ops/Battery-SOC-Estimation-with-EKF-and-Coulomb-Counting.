"""Synthetic test data and CSV loading.

The synthetic record is generated with "true" cell parameters that differ from
the parameters the EKF is given, so the filter has to cope with model mismatch
(wrong capacity and resistance), a biased current sensor and voltage noise.
"""
import csv
from dataclasses import replace
import numpy as np

from .battery import CellParams, simulate_cell


def make_current_profile(n: int, rng: np.random.Generator, target_ah: float = 2.0,
                         dt: float = 1.0) -> np.ndarray:
    """Random piecewise-constant load, smoothed, scaled to draw `target_ah` net."""
    levels = []
    while len(levels) < n:
        seg = int(rng.integers(20, 120))
        levels.extend([float(np.clip(rng.normal(1.0, 1.2), -1.5, 4.0))] * seg)
    i = np.array(levels[:n])
    i = np.convolve(i, np.ones(5) / 5, mode="same")
    return i * (target_ah * 3600.0 / dt) / i.sum()


def make_synthetic(seed: int = 0, duration_s: int = 7200, dt: float = 1.0,
                   current_bias: float = 0.05, v_noise: float = 0.005,
                   i_noise: float = 0.02):
    """Return (record dict, true CellParams, nominal CellParams given to the filter)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    true_p = CellParams(capacity_ah=2.42, r0=0.050, r1=0.022, c1=1900.0,
                        ocv_offset=0.010)   # 10 mV OCV error the filter does not know about
    nominal_p = CellParams()      # what the filter believes (2.5 Ah, 45 mOhm, ...)

    i_true = make_current_profile(n, rng, dt=dt)
    soc_true, _, vt_true = simulate_cell(true_p, i_true, dt, soc0=1.0)
    rec = {
        "t": np.arange(n) * dt,
        "current": i_true + current_bias + rng.normal(0.0, i_noise, n),   # measured
        "voltage": vt_true + rng.normal(0.0, v_noise, n),                 # measured
        "soc_true": soc_true,
        "dt": dt,
    }
    return rec, true_p, nominal_p


def load_csv(path: str):
    """CSV columns: time_s, current_a, voltage_v [, soc_true]  (current > 0 = discharge)."""
    t, i, v, s = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["time_s"]))
            i.append(float(row["current_a"]))
            v.append(float(row["voltage_v"]))
            if "soc_true" in row and row["soc_true"] != "":
                s.append(float(row["soc_true"]))
    t = np.array(t)
    rec = {"t": t, "current": np.array(i), "voltage": np.array(v),
           "soc_true": np.array(s) if len(s) == len(t) else None,
           "dt": float(np.median(np.diff(t)))}
    return rec
