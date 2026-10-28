"""First-order Thevenin equivalent-circuit cell model (1 RC pair).

States:  SOC (0..1) and V1 (voltage over the RC pair)
Input:   current I [A], positive = discharge
Output:  terminal voltage  Vt = OCV(SOC) - V1 - R0 * I
"""
from dataclasses import dataclass
import numpy as np

# Open-circuit voltage curve of a generic NMC cell (illustrative values,
# NOT measured data). Replace with a curve identified from a real cell.
_SOC_PTS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
_OCV_PTS = np.array([3.00, 3.35, 3.50, 3.59, 3.64, 3.69, 3.75, 3.83, 3.92, 4.01, 4.10, 4.20])


def ocv(soc):
    """Open-circuit voltage [V] for SOC in 0..1."""
    return np.interp(soc, _SOC_PTS, _OCV_PTS)


def docv_dsoc(soc, h: float = 0.02):
    """Slope of the OCV curve [V per unit SOC] (central finite difference)."""
    lo = np.clip(soc - h, 0.0, 1.0)
    hi = np.clip(soc + h, 0.0, 1.0)
    return (ocv(hi) - ocv(lo)) / np.maximum(hi - lo, 1e-9)


@dataclass
class CellParams:
    capacity_ah: float = 2.5   # Ah
    r0: float = 0.045          # ohm, ohmic resistance
    r1: float = 0.020          # ohm, polarisation resistance
    c1: float = 2000.0         # F, polarisation capacitance  (tau = R1*C1 = 40 s)
    eta: float = 1.0           # coulombic efficiency
    ocv_offset: float = 0.0    # V, shifts the OCV curve (used to make the "true" cell
                               # differ from the filter's OCV model; the filter uses 0)

    @property
    def tau(self) -> float:
        return self.r1 * self.c1


def simulate_cell(p: CellParams, current: np.ndarray, dt: float, soc0: float):
    """Noise-free forward simulation. Returns (soc, v1, vt) arrays."""
    n = len(current)
    soc = np.zeros(n)
    v1 = np.zeros(n)
    vt = np.zeros(n)
    a = np.exp(-dt / p.tau)
    s, v = soc0, 0.0
    for k in range(n):
        i = current[k]
        soc[k], v1[k] = s, v
        vt[k] = ocv(s) + p.ocv_offset - v - p.r0 * i
        s = s - p.eta * i * dt / (3600.0 * p.capacity_ah)
        v = a * v + p.r1 * (1.0 - a) * i
    return soc, v1, vt
