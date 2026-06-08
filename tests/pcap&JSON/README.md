# RB-Y1 state acquisition: SDK read vs wire capture

Two scripts that observe the same robot state from **opposite ends of the stack**.
They exist as a learning/validation pair: one shows you what the data *means*
(decoded), the other shows you what actually crossed the network (raw bytes).

| | `read_state_sdk.py` | `capture_wire.sh` |
|---|---|---|
| Layer | Application (decoded SDK objects) | Transport (TCP/HTTP2/gRPC bytes) |
| Output | `state.jsonl` (human-readable) | `*.pcap` (opaque until dissected) |
| Content | Only the channels you extract (lossy) | Every byte on the wire (complete) |
| Needs sudo? | No | Yes (raw packet capture) |
| Touches robot? | Reads only, no motion | Passive sniff, no connection of its own |
| Role | **Your actual data path** | **A diagnostic instrument** |

The single most important framing: **the JSONL is the data you will build datasets
from; the pcap is never training data.** The pcap's only job is to tell you whether
the transport is healthy, independent of your code.

---

## `read_state_sdk.py` — decoded state via the SDK

### What it does
Connects to the R-PC over gRPC, subscribes to the RobotState stream at a target
rate, and writes one JSON object per sample to a `.jsonl` file. The SDK has already
decoded the wire bytes into native objects (numpy arrays, enums), so the output is
directly readable. Channels logged: `host_time`, `dev_time`, `position`, `velocity`,
`torque`, `temperature`, `ft_right` (`[Fx,Fy,Fz,Tx,Ty,Tz]`), `ft_left`.

It reads only — it never enables servos or commands motion.

### Usage
```bash
source ~/rby1env/bin/activate      # venv that has rby1_sdk (v0.10.0)

python read_state_sdk.py \
    --address 192.168.30.1:50051 \
    --duration 30 \
    --rate 100 \
    --out state.jsonl
```

| Flag | Default | Meaning |
|---|---|---|
| `--address` | `192.168.30.1:50051` | R-PC gRPC endpoint (sim: `localhost:50051`) |
| `--model` | `a` | robot model |
| `--duration` | `30` | seconds to stream |
| `--rate` | `100` | Hz (this gRPC channel caps at ~100) |
| `--out` | `state.jsonl` | output file |

### Inspect the output
```bash
head -n 1 state.jsonl | python -m json.tool   # pretty-print first record
wc -l state.jsonl                              # ~3000 lines for 30 s @ 100 Hz
```

### Why JSONL here
It's human-readable on purpose, to contrast with the opaque pcap. It is **lossy**:
it contains only the channels above. Anything not extracted (EMO state, collisions,
odometry, battery) is gone and not recoverable — if you might want a channel later,
add it before collecting real data. JSONL is fine for inspection/learning; for the
analysis ahead use `.npz`, and for actual demonstration episodes graduate to
HDF5 / the LeRobot layout (which can hold ragged, typed, video-bearing data).

---

## `capture_wire.sh` — raw gRPC wire capture

### What it does
Runs `tcpdump` to record the actual packets between this U-PC and the R-PC. It
auto-detects the interface toward the robot (`ip route get`), filters to the gRPC
host/port, and writes a `.pcap`. It generates no traffic itself — run it
**alongside** the SDK reader. On exit it prints decode hints.

The bytes it captures are layered:
`TCP -> HTTP/2 frames -> gRPC framing (1 flag byte + 4-byte big-endian length) -> protobuf`.

### Usage (two terminals, both in the venv)
```bash
# Terminal 1 — start the capture FIRST
sudo bash capture_wire.sh 192.168.30.1 50051 rby1_wire.pcap

# Terminal 2 — generate traffic
python read_state_sdk.py --address 192.168.30.1:50051 --duration 30 --rate 100

# back in Terminal 1: Ctrl-C to stop
```
Args are positional: `[ROBOT_IP] [PORT] [OUTFILE]`. If `./capture_wire.sh` gives
"command not found", invoke it as `sudo bash capture_wire.sh ...` (sidesteps the
exec bit and PATH-under-sudo quirks).

### Reading the pcap in Wireshark
The capture is plain TCP until you tell Wireshark it's HTTP/2 (cleartext h2c on
50051 is not auto-detected):

1. Right-click a port-50051 packet -> **Decode As…** -> TCP port `50051` -> `HTTP2`.
2. Analyze -> **Enabled Protocols**: ensure `HTTP2`, `GRPC`, `Protobuf` are ticked.
3. Preferences -> Protocols -> HTTP2 -> enable "reassemble bodies spanning multiple frames".
4. For *named* protobuf fields, point Preferences -> Protobuf -> search paths at
   Rainbow's `.proto`. Without it you still see field numbers + wire types.

Headless equivalent:
```bash
tshark -r rby1_wire.pcap -d tcp.port==50051,http2 -Y http2 -O http2,grpc,protobuf
```

### Two gotchas that cost time today
- **Start the capture before the reader.** Wireshark can only lock onto cleartext
  HTTP/2 if it sees the connection preface at the start of the TCP stream. Miss the
  handshake and the stream won't dissect even with Decode As.
- **Only works on a plaintext link.** Apply filter `tls`; if it matches anything,
  the payload is encrypted and the protobuf can't be dissected.

### What to actually look for
This is a transport health check, not data:
- **I/O graph** (`grpc`/`http2`, or just packets): flat ~200 pkt/s for a 100 Hz
  stream (one DATA + one ACK per sample) = steady emission, no stalls.
- Filter `tcp.analysis.retransmission` and `tcp.analysis.zero_window`: both should
  be ~empty. Retransmissions = link packet loss; zero-window = your host couldn't
  drain the socket fast enough.

---

## How the two fit together

**The cross-check is the payoff of having both.** If the JSONL shows gaps or jitter
but the pcap shows clean, flat 100 Hz with zero retransmits, the problem is on *your*
side (Python GC, scheduling, a slow callback). If the pcap itself is jittery or
retransmitting, it's the *link*. That comparison localizes any data-quality issue to
host-vs-network — the concrete version of the determinism question.

**Language note (Python vs C++).** For both of these, Python is the better design,
not worse: the SDK reader runs at <=100 Hz doing trivial per-sample work (GIL/GC are
far below the noise floor), and offline pcap decoding isn't latency-sensitive. The
SDK's heavy lifting is already compiled C++ behind a pybind shell, so rewriting these
in C++ buys nothing and costs you the NumPy/SciPy/PyTorch ecosystem. C++ earns its
place only on the **500 Hz realtime control loop** (the `ControlState`/`control()`
path), where the 2 ms budget makes GIL contention and GC pauses real jitter sources.
Choose per component by timing requirement, not globally.

---

## Where this sits in the plan

- **Step 0 — validate acquisition (DONE).** Timing characterized (≈100 Hz, dt std
  ~0.19 ms, 0% missed), F/T noise floor measured (~0.33–0.41 N, ~0.004–0.023 Nm per
  axis) and bias recorded, accessors locked against the v0.10.0 wheel, transport
  confirmed clean via the pcap. These two scripts closed the "do I trust the stream"
  question.
- **Step 1 — force-bandwidth experiment (NEXT).** Teleop the leader arms through
  representative carbon-fiber layup contacts (debulk press, ply-conformance sweep,
  wrinkle-out stroke), log both F/T sensors at 100 Hz to `.npz`, then compute the PSD
  of each force/torque channel. If informative energy sits below ~40 Hz, 100 Hz
  RobotState is sufficient. If it climbs toward Nyquist, you need the 500 Hz
  control-loop tap (hybrid pipeline). **This result gates the policy-architecture
  choice (ACT vs diffusion vs …).**
- **Step 2 — multimodal sync** (cameras + proprioception + leader-arm commands on one
  clock; anchor on the R-PC timestamp, since device/host clocks drift a few ms).
- **Step 3 — demonstration capture** in a structured format (HDF5 / LeRobot).
- **Step 4 — policy choice and training.**

### Storage progression
`JSONL`/`pcap` were for learning and inspection. Use `.npz` for the Step 1 analysis
(columnar, math-ready). Move to HDF5 / LeRobot for demonstration episodes (typed,
ragged, video-capable, schema-stable).
