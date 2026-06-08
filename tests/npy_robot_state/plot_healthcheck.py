#!/usr/bin/env python3
"""
Plot an RB-Y1 health-check .npz (saved by rby1_state_healthcheck.py).

Lays out, in one figure:
  - timing jitter: dt histogram + dt over time (spots bursts/spikes)
  - right & left arm force (Fx/Fy/Fz) and torque (Tx/Ty/Tz) traces
Each F/T legend label shows that channel's std (the noise floor).

Usage:
  pip install matplotlib            # once, in this venv
  python plot_healthcheck.py                      # uses rby1_healthcheck.npz
  python plot_healthcheck.py my.npz --show        # open a window instead of saving
"""

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="rby1_healthcheck.npz")
    ap.add_argument("--save", default="rby1_healthcheck_plots.png")
    ap.add_argument("--show", action="store_true", help="open a window (needs a display)")
    ap.add_argument("--rate", type=float, default=100.0, help="nominal rate for the dt marker")
    args = ap.parse_args()

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")          # headless / over SSH
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed. Run:  pip install matplotlib")

    try:
        d = np.load(args.npz)
    except FileNotFoundError:
        sys.exit(f"File not found: {args.npz}")

    host_t = d["host_t"]
    ftR, ftL = d["ftR"], d["ftL"]
    t = host_t - host_t[0]
    dt = np.diff(host_t) * 1e3                 # ms
    nominal = 1e3 / args.rate

    f_lbl = ["Fx", "Fy", "Fz"]
    t_lbl = ["Tx", "Ty", "Tz"]

    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle(f"RB-Y1 health check — {args.npz}  ({len(host_t)} samples)", fontsize=13)

    # --- timing: histogram ---
    ax[0, 0].hist(dt, bins=80)
    ax[0, 0].axvline(nominal, color="k", ls="--", lw=1, label=f"nominal {nominal:.1f} ms")
    ax[0, 0].set(xlabel="inter-sample dt (ms)", ylabel="count",
                 title=f"Timing jitter  (std {dt.std():.2f} ms, "
                       f"p99 {np.percentile(dt, 99):.2f} ms)")
    ax[0, 0].legend(fontsize=8)

    # --- timing: dt over time (reveals bursts/spikes) ---
    ax[0, 1].plot(t[1:], dt, lw=0.5)
    ax[0, 1].axhline(nominal, color="k", ls="--", lw=1)
    ax[0, 1].set(xlabel="time (s)", ylabel="dt (ms)", title="dt over time")

    # --- F/T panels ---
    def ft_panel(axis, data, labels, base, ylabel, title):
        for i, lab in enumerate(labels):
            col = data[:, base + i]
            axis.plot(t, col, lw=0.6, label=f"{lab} (σ={col.std():.3f})")
        axis.set(xlabel="time (s)", ylabel=ylabel, title=title)
        axis.legend(ncol=3, fontsize=8)

    ft_panel(ax[1, 0], ftR, f_lbl, 0, "force (N)",  "Right arm — force")
    ft_panel(ax[1, 1], ftR, t_lbl, 3, "torque (Nm)", "Right arm — torque")
    ft_panel(ax[2, 0], ftL, f_lbl, 0, "force (N)",  "Left arm — force")
    ft_panel(ax[2, 1], ftL, t_lbl, 3, "torque (Nm)", "Left arm — torque")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.show:
        plt.show()
    else:
        fig.savefig(args.save, dpi=120)
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
