"""Matplotlib figures (same light-surface style as the EV range simulator)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=11)


def plot_estimation(rec, ekf, cc, guess, path):
    t = rec["t"] / 60.0
    has_truth = rec["soc_true"] is not None
    cc_label = "Coulomb counting (given true start)" if has_truth else "Coulomb counting"
    nrows = 3 if has_truth else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 2.7 * nrows + 0.8), sharex=True)

    ax = axes[0]
    if has_truth:
        ax.plot(t, rec["soc_true"] * 100, color=INK2, lw=1.4, ls="--", label="True SOC")
    ax.plot(t, cc * 100, color=ORANGE, lw=1.6, label=cc_label)
    ax.plot(t, ekf["soc"] * 100, color=BLUE, lw=1.6, label=f"EKF (initial guess {guess:.0%})")
    ax.set_ylabel("SOC [%]")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    _title(ax, "SOC estimate")

    k = 1
    if has_truth:
        ax = axes[k]; k += 1
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(t, (cc - rec["soc_true"]) * 100, color=ORANGE, lw=1.4, label="Coulomb counting")
        ax.plot(t, (ekf["soc"] - rec["soc_true"]) * 100, color=BLUE, lw=1.4, label="EKF")
        ax.fill_between(t, -2 * ekf["std"] * 100, 2 * ekf["std"] * 100, color=BLUE,
                        alpha=0.10, lw=0, label="EKF +/-2 sigma")
        ax.set_ylabel("SOC error [% points]")
        ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
        _title(ax, "Estimation error (estimate minus true SOC)")

    ax = axes[k]
    ax.plot(t, rec["current"], color=AQUA, lw=1.0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Current [A]")
    ax.set_xlabel("Time [min]")
    _title(ax, "Measured load current (positive = discharge)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bias_sweep(sweep, path):
    b = sweep["biases"]
    x = np.arange(len(b))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for off, key, col, lab in ((-w / 2, "cc", ORANGE, "Coulomb counting (given true start)"),
                               (+w / 2, "ekf", BLUE, "EKF (starts from a wrong guess)")):
        m = sweep[key].mean(axis=1)
        lo = m - sweep[key].min(axis=1)
        hi = sweep[key].max(axis=1) - m
        bars = ax.bar(x + off, m, w - 0.03, color=col, edgecolor=SURFACE, linewidth=1.5,
                      label=lab)
        ax.errorbar(x + off, m, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1,
                    capsize=3)
        for xi, mi in zip(x + off, m):
            ax.text(xi, mi + 0.35, f"{mi:.1f}", ha="center", va="bottom", fontsize=9,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g} A" for v in b])
    ax.set_xlabel("Current-sensor bias")
    ax.set_ylabel("SOC RMSE [% points]")
    ax.grid(axis="x", visible=False)
    n = sweep["ekf"].shape[1]
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Error vs. current-sensor bias (mean of {n} noise seeds, whiskers = min/max)",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_convergence(t_s, errors, init_errors, path):
    """errors: {label: error array}; init_errors: {label: error of the initial guess}.

    t = 0 is the initial guess; the following points are the estimates after each
    measurement update.
    """
    cols = [BLUE, ORANGE, AQUA]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    dt = float(t_s[1] - t_s[0])
    for (label, err), c in zip(errors.items(), cols):
        x = np.concatenate([[0.0], t_s + dt])
        y = np.concatenate([[init_errors[label]], err]) * 100
        a1.plot(x, y, color=c, lw=1.6, marker="o", ms=3, label=label)
        a2.plot(t_s / 60.0, err * 100, color=c, lw=1.4)
    for ax in (a1, a2):
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_ylabel("SOC error [% points]")
    a1.set_xlim(-0.5, 20)
    a1.set_xlabel("Time [s]")
    a1.legend(frameon=False, title="Initial SOC guess\n(true start: 100 %)",
              title_fontsize=9, loc="lower right")
    a1.set_title("First 20 s", loc="left", color=INK, fontsize=11)
    a2.set_xlabel("Time [min]")
    a2.set_title("Whole record", loc="left", color=INK, fontsize=11)
    fig.suptitle("EKF convergence from different initial guesses", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)
