#!/usr/bin/env python3
"""
Read RB-Y1 state via the SDK and log decoded values to JSONL.

This is the decoded counterpart to capture_wire.sh: instead of raw wire bytes,
the SDK hands you native objects, and you serialize the fields you want. JSONL
(one JSON object per line) is human-readable on purpose — open it in any editor
and compare against the opaque .pcap to see what decoding buys you.

Usage:
  python read_state_sdk.py --address 192.168.30.1:50051 --duration 30 --rate 100 --out state.jsonl

Inspect:
  head -n 1 state.jsonl | python -m json.tool
  wc -l state.jsonl
"""

import sys
import json
import time
import math
import argparse


def as_list(x):
    try:
        return [float(v) for v in x]
    except Exception:
        return []


def dev_seconds(ts):
    if ts is None:
        return math.nan
    if hasattr(ts, "timestamp") and callable(ts.timestamp):
        try:
            return float(ts.timestamp())
        except Exception:
            pass
    try:
        return float(ts)
    except Exception:
        return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default="192.168.30.1:50051")
    ap.add_argument("--model", default="a")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--out", default="state.jsonl")
    args = ap.parse_args()

    try:
        import rby1_sdk as rby
    except ImportError:
        sys.exit("rby1_sdk not importable. Activate the venv that has it.")

    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        sys.exit(f"Could not connect at {args.address}")
    print(f"Connected: {robot.get_robot_info()}")

    f = open(args.out, "w")
    n = {"count": 0}

    def on_state(state):
        rec = {
            "host_time": time.perf_counter(),
            "dev_time": dev_seconds(getattr(state, "timestamp", None)),
            "position": as_list(getattr(state, "position", [])),
            "velocity": as_list(getattr(state, "velocity", [])),
            "torque": as_list(getattr(state, "torque", [])),
            "temperature": as_list(getattr(state, "temperature", [])),
        }
        try:
            r = state.ft_sensor_right
            rec["ft_right"] = as_list(r.force) + as_list(r.torque)
            l = state.ft_sensor_left
            rec["ft_left"] = as_list(l.force) + as_list(l.torque)
        except Exception:
            rec["ft_right"] = rec["ft_left"] = []
        f.write(json.dumps(rec) + "\n")
        n["count"] += 1

    print(f"Streaming {args.duration}s @ {args.rate} Hz -> {args.out}")
    robot.start_state_update(on_state, args.rate)
    time.sleep(args.duration)
    robot.stop_state_update()
    robot.disconnect()
    f.flush()
    f.close()
    print(f"Done. {n['count']} records written.")


if __name__ == "__main__":
    main()
