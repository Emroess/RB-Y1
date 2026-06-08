#!/usr/bin/env python3
"""
RB-Y1 RobotState health check (observe-only).

Streams RobotState at a target rate while the robot is POWERED BUT STATIONARY,
then characterizes timing fidelity, F/T noise floor/bias, and data integrity.
Does NOT enable servos and does NOT command motion.

v2: numpy-array-aware vector accessors; correct EMO/collision interpretation
(state enum / distance, not list length); one-time structure dump so you can
see how your installed wheel actually exposes each field.

API names move between SDK versions. If a call fails, check the structure dump
and your installed wheel's Python API reference, then adjust the marked lines.
Pin numpy<2 to match the SDK wheel's ABI.
"""

import sys
import time
import math
import argparse

import numpy as np

try:
    import rby1_sdk as rby
except ImportError:
    sys.exit("rby1_sdk not importable in this environment. Run on the U-PC.")


# ----------------------------- accessors ------------------------------------
def _vec3(v):
    """Vec3 may be a numpy array, list, or an object with .x/.y/.z."""
    try:
        return [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        pass
    try:
        return [float(v.x), float(v.y), float(v.z)]
    except Exception:
        return [math.nan, math.nan, math.nan]


def ft_xyz(ft):
    """Return [Fx,Fy,Fz,Tx,Ty,Tz] from an FTSensorData."""
    try:
        return _vec3(ft.force) + _vec3(ft.torque)
    except Exception:
        return [math.nan] * 6


def device_time_seconds(ts):
    if ts is None:
        return math.nan
    if hasattr(ts, "timestamp") and callable(getattr(ts, "timestamp")):
        try:
            return float(ts.timestamp())
        except Exception:
            pass
    for s, n in (("seconds", "nanos"), ("tv_sec", "tv_nsec")):
        if hasattr(ts, s):
            return float(getattr(ts, s)) + float(getattr(ts, n, 0)) * 1e-9
    try:
        return float(ts)
    except Exception:
        return math.nan


def dump_structure(state):
    """Print how this wheel exposes the fields we care about (first frame only)."""
    print("\n--- one-time RobotState structure dump ---")
    for name in ("timestamp", "ft_sensor_right", "ft_sensor_left",
                 "emo_states", "collisions", "is_ready"):
        try:
            v = getattr(state, name)
        except Exception as e:
            print(f"  {name}: <no attr> ({e})")
            continue
        print(f"  {name}: type={type(v).__name__}  value={repr(v)[:90]}")
    try:
        f = state.ft_sensor_right.force
        print(f"  ft_sensor_right.force: type={type(f).__name__}  value={repr(f)[:60]}")
    except Exception as e:
        print(f"  ft_sensor_right.force: <error> {e}")
    try:
        e0 = state.emo_states[0]
        print(f"  emo_states[0]: type={type(e0).__name__}  "
              f"state={getattr(e0, 'state', '<no .state>')}")
    except Exception as e:
        print(f"  emo_states[0]: <error> {e}")
    print("--- end dump ---\n")


# ----------------------------- collection -----------------------------------
def collect(address, model, duration_s, rate_hz):
    robot = rby.create_robot(address, model)
    if not robot.connect():
        sys.exit(f"Could not connect to robot at {address}")
    print("Connected. Robot info:")
    print(f"  {robot.get_robot_info()}\n")

    host_t, dev_t, ftR, ftL, pos = [], [], [], [], []
    diag = {"not_ready": 0, "emo_values": set(),
            "coll_count_max": 0, "coll_min_dist": math.inf, "seen": False}

    def on_state(state):
        if not diag["seen"]:
            diag["seen"] = True
            dump_structure(state)
        host_t.append(time.perf_counter())
        dev_t.append(device_time_seconds(getattr(state, "timestamp", None)))
        ftR.append(ft_xyz(state.ft_sensor_right))
        ftL.append(ft_xyz(state.ft_sensor_left))
        try:
            pos.append(list(state.position))
        except Exception:
            pos.append([math.nan])
        try:
            if not all(state.is_ready):
                diag["not_ready"] += 1
        except Exception:
            pass
        try:
            for e in state.emo_states:                 # read the STATE, not the count
                diag["emo_values"].add(str(getattr(e, "state", e)))
        except Exception:
            pass
        try:
            cs = state.collisions
            diag["coll_count_max"] = max(diag["coll_count_max"], len(cs))
            for c in cs:                               # read DISTANCE, not the count
                d = getattr(c, "distance", None)
                if d is not None:
                    diag["coll_min_dist"] = min(diag["coll_min_dist"], float(d))
        except Exception:
            pass

    print(f"Streaming for {duration_s}s at target {rate_hz} Hz. Keep the robot STILL.")
    robot.start_state_update(on_state, rate_hz)
    t0 = time.perf_counter()
    time.sleep(duration_s)
    robot.stop_state_update()
    elapsed = time.perf_counter() - t0
    robot.disconnect()

    return {"elapsed": elapsed, "rate_hz": rate_hz,
            "host_t": np.array(host_t), "dev_t": np.array(dev_t),
            "ftR": np.array(ftR), "ftL": np.array(ftL),
            "pos": np.array(pos) if pos and len(pos[0]) > 1 else None,
            "diag": diag}


# ----------------------------- analysis --------------------------------------
def analyze(log):
    n = len(log["host_t"])
    nominal_dt = 1.0 / log["rate_hz"]
    eff_rate = n / log["elapsed"] if log["elapsed"] else float("nan")
    verdict = []

    print("================ TIMING ================")
    print(f"samples collected : {n}")
    print(f"elapsed           : {log['elapsed']:.3f} s")
    print(f"effective rate    : {eff_rate:.2f} Hz  (nominal {log['rate_hz']} Hz)")
    if n >= 2:
        dt = np.diff(log["host_t"]) * 1e3
        p = np.percentile(dt, [50, 95, 99])
        late = np.mean(dt > 1.5 * nominal_dt * 1e3) * 100
        print(f"dt mean / std     : {dt.mean():.2f} / {dt.std():.2f} ms")
        print(f"dt min / max      : {dt.min():.2f} / {dt.max():.2f} ms")
        print(f"dt p50/p95/p99    : {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f} ms")
        print(f"samples >1.5x dt  : {late:.2f} %")
        if eff_rate < 0.9 * log["rate_hz"]:
            verdict.append("LOW effective rate")
        if dt.std() > 0.5 * nominal_dt * 1e3:
            verdict.append("HIGH jitter")
        dev = log["dev_t"][np.isfinite(log["dev_t"])]
        if dev.size > 2:
            drift = (dev[-1] - dev[0]) - (log["host_t"][-1] - log["host_t"][0])
            print(f"device/host drift : {1e3*drift:+.1f} ms over capture")

    print("\n================ F/T NOISE FLOOR (at rest) ================")
    axes = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    units = ["N", "N", "N", "Nm", "Nm", "Nm"]
    for name, arr in (("RIGHT", log["ftR"]), ("LEFT", log["ftL"])):
        print(f"  {name} arm:")
        if arr.size == 0:
            print("    <no data>")
            continue
        for i, (ax, u) in enumerate(zip(axes, units)):
            col = arr[:, i][np.isfinite(arr[:, i])]
            if col.size:
                print(f"    {ax:>2}: bias {col.mean():+8.3f} {u:<2}  "
                      f"noise(std) {col.std():.4f} {u}")
        if not np.all(np.isfinite(arr)):
            verdict.append(f"{name} F/T has NaN/Inf (check accessor vs structure dump)")
        elif np.allclose(arr, 0.0):
            verdict.append(f"{name} F/T all zero (sensor/flange may be unpowered)")

    print("\n================ INTEGRITY ================")
    d = log["diag"]
    print(f"frames not-ready  : {d['not_ready']}")
    print(f"EMO state(s) seen : {d['emo_values'] or '<none read>'}")
    cd = "n/a" if d["coll_min_dist"] is math.inf else f"{d['coll_min_dist']:.4f}"
    print(f"collision entries : up to {d['coll_count_max']} / frame, min distance {cd}")
    if d["not_ready"]:
        verdict.append("robot reported not-ready frames")
    if any("PRESS" in v.upper() for v in d["emo_values"]):
        verdict.append("EMO appears PRESSED during capture")

    print("\n================ VERDICT ================")
    if not verdict:
        print("HEALTHY: contiguous low-jitter stream, F/T live, no active faults.")
    else:
        print("ATTENTION:")
        for v in verdict:
            print(f"  - {v}")
    return verdict


def save_outputs(log, prefix):
    np.savez_compressed(f"{prefix}.npz",
                        host_t=log["host_t"], dev_t=log["dev_t"],
                        ftR=log["ftR"], ftL=log["ftL"],
                        pos=log["pos"] if log["pos"] is not None else np.array([]))
    print(f"\nRaw log saved to {prefix}.npz")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7))
        if len(log["host_t"]) > 1:
            dt = np.diff(log["host_t"]) * 1e3
            ax[0].hist(dt, bins=60)
            ax[0].axvline(1e3 / log["rate_hz"], color="k", ls="--", label="nominal")
            ax[0].set(xlabel="inter-sample dt (ms)", ylabel="count",
                      title="Timing jitter (host clock)")
            ax[0].legend()
        if log["ftR"].size:
            t = log["host_t"] - log["host_t"][0]
            for i, lbl in enumerate(["Fx", "Fy", "Fz"]):
                ax[1].plot(t, log["ftR"][:, i], label=f"R {lbl}")
            ax[1].set(xlabel="time (s)", ylabel="force (N)",
                      title="Right-arm F/T at rest (noise floor)")
            ax[1].legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{prefix}.png", dpi=120)
        print(f"Plots saved to {prefix}.png")
    except ImportError:
        print("(matplotlib not available; skipped plots)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default="192.168.30.1:50051")
    ap.add_argument("--model", default="a")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--prefix", default="rby1_healthcheck")
    args = ap.parse_args()
    log = collect(args.address, args.model, args.duration, args.rate)
    analyze(log)
    save_outputs(log, args.prefix)


if __name__ == "__main__":
    main()
