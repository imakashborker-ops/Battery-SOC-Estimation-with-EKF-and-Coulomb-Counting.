"""Robustness experiment: SOC error versus current-sensor bias."""
import numpy as np

from .data import make_synthetic
from .ekf import run_ekf, coulomb_counting


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def bias_sweep(biases=(0.0, 0.05, 0.1, 0.2), seeds=range(5), soc_guess=0.5,
               settle_s: float = 600.0):
    """RMSE (% points, after `settle_s`) of EKF and coulomb counting per current bias.

    The EKF starts from `soc_guess`; coulomb counting is given the TRUE start.
    Returns dict {"biases", "ekf": array[n_bias, n_seed], "cc": array[n_bias, n_seed]}.
    """
    ekf_r = np.zeros((len(biases), len(list(seeds))))
    cc_r = np.zeros_like(ekf_r)
    for bi, b in enumerate(biases):
        for si, s in enumerate(seeds):
            rec, _, nominal = make_synthetic(seed=s, current_bias=b)
            late = rec["t"] >= settle_s
            e = run_ekf(nominal, rec["dt"], rec["current"], rec["voltage"], soc0=soc_guess)
            c = coulomb_counting(nominal, rec["dt"], rec["current"], rec["soc_true"][0])
            ekf_r[bi, si] = _rmse((e["soc"] - rec["soc_true"])[late]) * 100
            cc_r[bi, si] = _rmse((c - rec["soc_true"])[late]) * 100
    return {"biases": np.array(biases), "ekf": ekf_r, "cc": cc_r}
