"""SOC estimation with an Extended Kalman Filter on a Thevenin cell model.

    python main.py                          # synthetic data
    python main.py --soc-guess 0.3          # start the filters from a worse guess
    python main.py --current-bias 0.1       # stronger current-sensor bias
    python main.py --csv my_data.csv        # columns: time_s,current_a,voltage_v[,soc_true]
"""
import argparse
import os

import numpy as np

from soc_ekf.battery import CellParams
from soc_ekf.data import make_synthetic, load_csv
from soc_ekf.ekf import run_ekf, coulomb_counting
from soc_ekf.experiments import bias_sweep
from soc_ekf.plots import plot_estimation, plot_convergence, plot_bias_sweep


def converged_at(err, t, tol=0.02):
    """First time after which |err| stays below `tol` for the rest of the record."""
    bad = np.where(np.abs(err) > tol)[0]
    if len(bad) == 0:
        return t[0]
    if bad[-1] + 1 >= len(t):
        return None
    return t[bad[-1] + 1]


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soc-guess", type=float, default=0.5, help="initial SOC given to the filters")
    ap.add_argument("--current-bias", type=float, default=0.05, help="current sensor offset [A]")
    ap.add_argument("--csv", help="use your own data instead of the synthetic record")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.csv:
        rec = load_csv(args.csv)
        nominal = CellParams()
    else:
        rec, _true, nominal = make_synthetic(seed=args.seed, current_bias=args.current_bias)
    dt = rec["dt"]

    ekf = run_ekf(nominal, dt, rec["current"], rec["voltage"], soc0=args.soc_guess)
    # Fair baseline: coulomb counting gets the TRUE initial SOC when it is known
    # (it can never recover from a wrong start; the EKF can).
    cc_start = rec["soc_true"][0] if rec["soc_true"] is not None else args.soc_guess
    cc = coulomb_counting(nominal, dt, rec["current"], soc0=cc_start)

    os.makedirs(args.out, exist_ok=True)
    plot_estimation(rec, ekf, cc, args.soc_guess, os.path.join(args.out, "soc_estimation.png"))

    if rec["soc_true"] is not None:
        t = rec["t"]
        e_ekf = ekf["soc"] - rec["soc_true"]
        e_cc = cc - rec["soc_true"]
        late = t >= 600.0           # ignore the first 10 minutes (filter convergence)
        tc = converged_at(e_ekf, t)
        print(f"EKF initial guess: {args.soc_guess:.0%}  (true: {rec['soc_true'][0]:.0%})")
        print(f"Record length:     {t[-1] / 60:.0f} min at {dt:g} s sampling")
        print(f"EKF  RMSE after 10 min:               {rmse(e_ekf[late]) * 100:.2f} % points, "
              f"max |err| {np.abs(e_ekf[late]).max() * 100:.2f} %")
        print(f"Coulomb counting (true start) RMSE:   {rmse(e_cc[late]) * 100:.2f} % points, "
              f"max |err| {np.abs(e_cc[late]).max() * 100:.2f} %")
        print(f"Coulomb counting error at end of run: {e_cc[-1] * 100:+.2f} % points "
              f"(EKF: {e_ekf[-1] * 100:+.2f})")
        print("EKF within 2 % points from: " +
              (f"{tc:.0f} s on" if tc is not None else "never"))

        if not args.csv:
            guesses = {"20 %": 0.2, "50 %": 0.5, "80 %": 0.8}
            errs, init_errs = {}, {}
            for label, g in guesses.items():
                r = run_ekf(nominal, dt, rec["current"], rec["voltage"], soc0=g)
                errs[label] = r["soc"] - rec["soc_true"]
                init_errs[label] = g - rec["soc_true"][0]
            plot_convergence(t, errs, init_errs, os.path.join(args.out, "convergence.png"))

            sweep = bias_sweep(soc_guess=args.soc_guess)
            plot_bias_sweep(sweep, os.path.join(args.out, "bias_sweep.png"))
            print("\nRMSE after 10 min vs current-sensor bias (mean of 5 seeds), % points:")
            print("  bias [A]   coulomb counting   EKF")
            for b, c, e in zip(sweep["biases"], sweep["cc"].mean(1), sweep["ekf"].mean(1)):
                print(f"  {b:7.2f}   {c:16.2f}   {e:5.2f}")
    else:
        print("No soc_true column in the CSV: plots written, error metrics skipped.")


if __name__ == "__main__":
    main()
