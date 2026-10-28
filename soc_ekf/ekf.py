"""Extended Kalman Filter for SOC estimation + coulomb-counting baseline."""
import numpy as np

from .battery import CellParams, ocv, docv_dsoc


class SocEKF:
    """EKF with state x = [SOC, V1].

    Prediction : coulomb counting + RC-pair dynamics (linear in the states)
    Update     : terminal voltage via the nonlinear OCV(SOC) curve (linearised)
    """

    def __init__(self, params: CellParams, dt: float, soc0: float,
                 p0_soc_std: float = 0.2, q_soc: float = 1e-8, q_v1: float = 1e-6,
                 r_meas: float = 0.008 ** 2):
        self.p = params
        self.dt = dt
        self.x = np.array([soc0, 0.0])
        self.P = np.diag([p0_soc_std ** 2, 0.02 ** 2])
        self.Q = np.diag([q_soc, q_v1])
        self.R = r_meas
        self.a = np.exp(-dt / params.tau)

    def step(self, current: float, v_meas: float):
        p, dt, a = self.p, self.dt, self.a
        # --- predict
        soc = self.x[0] - p.eta * current * dt / (3600.0 * p.capacity_ah)
        v1 = a * self.x[1] + p.r1 * (1.0 - a) * current
        F = np.array([[1.0, 0.0], [0.0, a]])
        x_pred = np.array([soc, v1])
        P_pred = F @ self.P @ F.T + self.Q

        # --- update
        v_pred = float(ocv(soc)) - v1 - p.r0 * current
        H = np.array([[float(docv_dsoc(soc)), -1.0]])
        S = (H @ P_pred @ H.T)[0, 0] + self.R
        K = (P_pred @ H.T) / S                       # 2x1
        innovation = v_meas - v_pred
        self.x = x_pred + (K * innovation).ravel()
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + self.R * (K @ K.T)   # Joseph form
        return self.x[0], self.x[1], v_pred, float(np.sqrt(self.P[0, 0]))


def run_ekf(params: CellParams, dt: float, current, v_meas, soc0: float, **kw):
    """Run the EKF over a whole record. Returns dict of arrays."""
    ekf = SocEKF(params, dt, soc0, **kw)
    n = len(current)
    soc, v1, v_pred, std = (np.zeros(n) for _ in range(4))
    for k in range(n):
        soc[k], v1[k], v_pred[k], std[k] = ekf.step(current[k], v_meas[k])
    return {"soc": soc, "v1": v1, "v_pred": v_pred, "std": std}


def coulomb_counting(params: CellParams, dt: float, current, soc0: float):
    """Baseline: integrate the (measured) current only."""
    return soc0 - np.cumsum(params.eta * np.asarray(current) * dt) / (3600.0 * params.capacity_ah)
