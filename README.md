# RBY1 Teleoperation Recording & Replay

A paired set of scripts for **recording** teleoperation episodes and **replaying** them on the Rainbow Robotics RBY1 humanoid. Designed for imitation learning data collection — all data is saved as **MCAP files** with ROS2-compatible CDR-encoded messages, directly compatible with UR5 ROS bag training pipelines.

> **Platform:** Runs on the UPC (onboard NVIDIA Jetson) connected to the RBY1's master arm and grippers.

---

## Scripts

| Script | Purpose |
|---|---|
| `teleop_episode_recording.py` | Two-arm teleoperation with episode recording |
| `teleop_episode_replay.py` | Replays recorded MCAP episodes on the robot |

---

## Requirements

```
rby1_sdk
numpy
mcap
rosbags
```

The `models/master_arm/model.urdf` file must be present in the same directory as the recording script.

---

## 1. Recording — `teleop_episode_recording.py`

Records dual-arm teleoperation sessions as per-episode MCAP files.

### Usage

```bash
python3 teleop_episode_recording.py --address <ROBOT_IP:PORT>
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--address` | *(required)* | Robot IP and port (e.g. `192.168.30.1:50051`) |
| `--model` | `a` | Robot model name (`a` or `m`) |
| `--power` | `.*` | Regex pattern for power device names |
| `--servo` | `torso_.*\|right_arm_.*\|left_arm_.*` | Regex pattern for servo names |
| `--mode` | `position` | Control mode: `position` or `impedance` |

### Startup Flow

1. Connects to the robot and powers on servos
2. Initializes and homes the grippers
3. Initializes the master arm
4. **Prompts for a session folder name** — you can:
   - Type a **number** to resume an existing session
   - Type a **name** to create a new session
   - Press **Enter** for an auto-generated timestamped name
5. Starts the master arm control loop
6. Moves the robot arms to match the master arm's current position
7. Enters the teleop idle state, waiting for keyboard input

### Keyboard Controls

| Key | Action |
|---|---|
| **↑ Up Arrow** | Enable teleoperation (both arms) — starts a new recording episode |
| **↓ Down Arrow** | Disable teleoperation (both arms) — ends and saves the current episode |
| **→ Right Arrow** | Toggle right gripper open/closed |
| **← Left Arrow** | Toggle left gripper open/closed |
| **Ctrl+C** | Shut down — saves any active episode, powers off the robot |

### Episode Lifecycle

Each press of **↑ Up** starts a new episode file. Each press of **↓ Down** ends and saves it.

```
TeleOpRecorderData/
└── my_session/
    ├── episode_001_20260512_182030.mcap
    ├── episode_002_20260512_182145.mcap
    └── episode_003_20260512_182300.mcap
```

### MCAP Topics Recorded

All topics use **CDR serialization** with the `ros2` MCAP profile — compatible with `ros2 bag play`, Foxglove Studio, and the `rosbags` Python library.

| Topic | Message Type | Description |
|---|---|---|
| `/right/joint_states` | `sensor_msgs/msg/JointState` | Right arm position, velocity, effort |
| `/left/joint_states` | `sensor_msgs/msg/JointState` | Left arm position, velocity, effort |
| `/right/joint_commands` | `sensor_msgs/msg/JointState` | Right arm commanded positions |
| `/left/joint_commands` | `sensor_msgs/msg/JointState` | Left arm commanded positions |
| `/right/force_torque_sensor_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` | Right arm 6-axis force/torque |
| `/left/force_torque_sensor_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` | Left arm 6-axis force/torque |
| `/right_hand/joint_states` | `sensor_msgs/msg/JointState` | Right gripper state |
| `/left_hand/joint_states` | `sensor_msgs/msg/JointState` | Left gripper state |
| `/master_arm/joint_states` | `sensor_msgs/msg/JointState` | Master arm position, velocity, gravity term |
| `/teleop/status` | `std_msgs/msg/Float64MultiArray` | Collision flag, enable flags, currents, torques |

> Topic names are designed to match the UR5 ROS bag format for cross-robot training pipeline compatibility.

---

## 2. Replay — `teleop_episode_replay.py`

Reads recorded MCAP episodes and replays the joint commands on the physical robot.

### Usage

```bash
# Interactive mode — select session and episode from a menu
python3 teleop_episode_replay.py --address 192.168.30.1:50051

# Half speed (recommended for first-time testing)
python3 teleop_episode_replay.py --address 192.168.30.1:50051 --speed 0.5

# Direct path to a specific episode file
python3 teleop_episode_replay.py --address 192.168.30.1:50051 --mcap /path/to/episode.mcap
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--address` | *(required)* | Robot IP and port |
| `--model` | `a` | Robot model name |
| `--power` | `.*` | Regex pattern for power device names |
| `--servo` | `torso_.*\|right_arm_.*\|left_arm_.*` | Regex pattern for servo names |
| `--mode` | `position` | Control mode: `position` or `impedance` |
| `--speed` | `1.0` | Playback speed multiplier (e.g. `0.5` = half speed, `2.0` = double) |
| `--mcap` | *(none)* | Direct path to an `.mcap` file (skips interactive selection) |

### Replay Flow

1. Connects to the robot, powers on, homes grippers
2. **Episode selection** — interactive menu or `--mcap` direct path
   - Select `all` to play every episode in a session sequentially
3. Loads the MCAP file and extracts command frames
4. **Moves to the first frame's position** over 5 seconds (safe approach)
5. **Waits for ENTER** — manual confirmation before replay starts
6. **Replays at original timing** — adjustable with `--speed`
7. After replay completes, returns to the ready pose
8. **Robot stays powered on and idle** — press **Ctrl+C** to shut down

### Safety Features

- **Slow approach** to the episode's start position (5 seconds)
- **Manual confirmation** required before playback begins
- **Collision detection** — frames that would cause self-collision are skipped
- **Speed control** — use `--speed 0.5` for cautious first runs
- **Joint limits** — all commanded positions are clamped to the robot's joint limits
- **Ctrl+C** cleanly powers off the robot at any point

---

## Reading MCAP Data in Python

You can load recorded episodes for training without ROS2 installed:

```python
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS2_HUMBLE)

with open("episode_001_20260512_182030.mcap", "rb") as f:
    reader = make_reader(f)
    for schema, channel, message in reader.iter_messages():
        msg = typestore.deserialize_cdr(message.data, schema.name)

        if channel.topic == "/right/joint_commands":
            positions = msg.position  # numpy array of joint angles
            print(f"t={message.log_time}  right_cmd: {positions}")
```

---

## Directory Structure

```
TeleOp Episode Rec. Script and MCAP/
├── teleop_episode_recording.py      # Recording script
├── teleop_episode_replay.py         # Replay script
├── models/
│   └── master_arm/
│       └── model.urdf               # Master arm URDF (required for recording)
├── README.md                         # This file
└── TeleOpRecorderData/               # Created automatically
    ├── pick_and_place/
    │   ├── episode_001_20260512_182030.mcap
    │   └── episode_002_20260512_182145.mcap
    └── gripper_calibration/
        └── episode_001_20260512_190500.mcap
```
