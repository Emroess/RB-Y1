# -*- coding: utf-8 -*-
"""
Teleoperation Example

Run this example on UPC to which the master arm and hands are connected
"""

import rby1_sdk as rby
import numpy as np
import os
import sys
import time
import logging
import argparse
import signal
import threading
import datetime
from mcap.writer import Writer as McapWriter
from rosbags.typesys import Stores, get_typestore
from typing import *
from dataclasses import dataclass

# ===== KEYBOARD TOGGLE STATE =====
right_arm_enabled = False
left_arm_enabled = False
right_gripper_open = False
left_gripper_open = False
ARROW_KEYS = {"A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT"}

# ===== ROS2 TYPE SYSTEM (standalone -- no ROS2 install needed) =====
_typestore = get_typestore(Stores.ROS2_HUMBLE)
_JointState = _typestore.types['sensor_msgs/msg/JointState']
_WrenchStamped = _typestore.types['geometry_msgs/msg/WrenchStamped']
_Wrench = _typestore.types['geometry_msgs/msg/Wrench']
_Vector3 = _typestore.types['geometry_msgs/msg/Vector3']
_Float64MultiArray = _typestore.types['std_msgs/msg/Float64MultiArray']
_MultiArrayLayout = _typestore.types['std_msgs/msg/MultiArrayLayout']
_Header = _typestore.types['std_msgs/msg/Header']
_Time = _typestore.types['builtin_interfaces/msg/Time']

# Full .msg definitions with dependency resolution (needed for ros2 bag play compat)
_JOINT_STATE_SCHEMA = (
    "std_msgs/Header header\n"
    "string[] name\n"
    "float64[] position\n"
    "float64[] velocity\n"
    "float64[] effort\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
)

_WRENCH_STAMPED_SCHEMA = (
    "std_msgs/Header header\n"
    "geometry_msgs/Wrench wrench\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    "\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Wrench\n"
    "geometry_msgs/Vector3 force\n"
    "geometry_msgs/Vector3 torque\n"
    "\n"
    "================================================================================\n"
    "MSG: geometry_msgs/Vector3\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n"
)

_FLOAT64_MULTI_ARRAY_SCHEMA = (
    "std_msgs/MultiArrayLayout layout\n"
    "float64[] data\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/MultiArrayLayout\n"
    "std_msgs/MultiArrayDimension[] dim\n"
    "uint32 data_offset\n"
    "\n"
    "================================================================================\n"
    "MSG: std_msgs/MultiArrayDimension\n"
    "string label\n"
    "uint32 size\n"
    "uint32 stride\n"
)


def _make_header(timestamp):
    """Create a std_msgs/Header from a Unix timestamp (float seconds)."""
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    return _Header(stamp=_Time(sec=sec, nanosec=nanosec), frame_id='')


def _make_joint_state(header, names, position, velocity, effort):
    """Create a sensor_msgs/JointState message."""
    return _JointState(
        header=header,
        name=np.array(names, dtype=object),
        position=np.array(position, dtype=np.float64),
        velocity=np.array(velocity, dtype=np.float64),
        effort=np.array(effort, dtype=np.float64),
    )


def _make_wrench_stamped(header, force_xyz, torque_xyz):
    """Create a geometry_msgs/WrenchStamped message."""
    return _WrenchStamped(
        header=header,
        wrench=_Wrench(
            force=_Vector3(x=force_xyz[0], y=force_xyz[1], z=force_xyz[2]),
            torque=_Vector3(x=torque_xyz[0], y=torque_xyz[1], z=torque_xyz[2]),
        ),
    )


def _make_float64_multi_array(values):
    """Create a std_msgs/Float64MultiArray message."""
    return _Float64MultiArray(
        layout=_MultiArrayLayout(dim=np.array([], dtype=object), data_offset=0),
        data=np.array(values, dtype=np.float64),
    )


# ===== IMITATION LEARNING DATA RECORDER =====
class TeleopRecorder:
    """Records two-arm teleop data to per-episode MCAP files for imitation learning.

    Episode lifecycle:
      - new_episode() : called when teleop is ENABLED  (UP arrow)
      - end_episode()  : called when teleop is DISABLED (DOWN arrow) or on shutdown

    Each episode is saved as  <save_dir>/<session_name>/episode_<NNN>_<timestamp>.mcap

    Topics written (UR5 ROS bag compatible, all ROS2 CDR-encoded):
      /right/joint_states                               sensor_msgs/msg/JointState       (right arm q, vel, effort)
      /left/joint_states                                sensor_msgs/msg/JointState       (left arm q, vel, effort)
      /right/joint_commands                             sensor_msgs/msg/JointState       (right arm commanded q)
      /left/joint_commands                              sensor_msgs/msg/JointState       (left arm commanded q)
      /right/force_torque_sensor_broadcaster/wrench     geometry_msgs/msg/WrenchStamped  (right arm F/T)
      /left/force_torque_sensor_broadcaster/wrench      geometry_msgs/msg/WrenchStamped  (left arm F/T)
      /right_hand/joint_states                          sensor_msgs/msg/JointState       (right gripper state)
      /left_hand/joint_states                           sensor_msgs/msg/JointState       (left gripper state)
      /master_arm/joint_states                          sensor_msgs/msg/JointState       (ma_q, vel, gravity_term)
      /teleop/status                                    std_msgs/msg/Float64MultiArray   (flags + extras)
    """

    # Topic definitions:  (topic_name, ros2_type_string, schema_text)
    TOPICS = [
        # Per-arm joint states (matches UR5 bag)
        ('/right/joint_states',                            'sensor_msgs/msg/JointState',          _JOINT_STATE_SCHEMA),
        ('/left/joint_states',                             'sensor_msgs/msg/JointState',          _JOINT_STATE_SCHEMA),
        # Per-arm joint commands
        ('/right/joint_commands',                          'sensor_msgs/msg/JointState',          _JOINT_STATE_SCHEMA),
        ('/left/joint_commands',                           'sensor_msgs/msg/JointState',          _JOINT_STATE_SCHEMA),
        # Per-arm force/torque (matches UR5 topic name)
        ('/right/force_torque_sensor_broadcaster/wrench',  'geometry_msgs/msg/WrenchStamped',     _WRENCH_STAMPED_SCHEMA),
        ('/left/force_torque_sensor_broadcaster/wrench',   'geometry_msgs/msg/WrenchStamped',     _WRENCH_STAMPED_SCHEMA),
        # Per-hand gripper state (matches UR5 bag)
        ('/right_hand/joint_states',                       'sensor_msgs/msg/JointState',          _JOINT_STATE_SCHEMA),
        ('/left_hand/joint_states',                        'sensor_msgs/msg/JointState',          _JOINT_STATE_SCHEMA),
        # RBY1-specific (no UR5 equivalent, but useful for training)
        ('/master_arm/joint_states',                       'sensor_msgs/msg/JointState',          _JOINT_STATE_SCHEMA),
        ('/teleop/status',                                 'std_msgs/msg/Float64MultiArray',      _FLOAT64_MULTI_ARRAY_SCHEMA),
    ]

    def __init__(self, save_dir=None, session_name=None, right_arm_idx=None, left_arm_idx=None):
        if save_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            save_dir = os.path.join(script_dir, "TeleOpRecorderData")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if session_name is None:
            session_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{session_stamp}"
        self.session_dir = os.path.join(save_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)
        # Start episode count from existing episodes (so resumed sessions continue numbering)
        self.episode_count = len([f for f in os.listdir(self.session_dir) if f.endswith('.mcap')])
        self._file = None
        self._writer = None
        self._channels = {}   # topic_name -> channel_id
        self._lock = threading.Lock()
        # Store arm joint indices for slicing full robot state
        self._right_arm_idx = right_arm_idx
        self._left_arm_idx = left_arm_idx
        logging.info(f"[Recorder] Session directory: {self.session_dir}")

    # ---- episode lifecycle ----
    def new_episode(self):
        """Flush any active episode and start a fresh one."""
        with self._lock:
            self._close_writer_locked()
            self.episode_count += 1
            ep_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{self.episode_count:03d}_{ep_stamp}.mcap"
            path = os.path.join(self.session_dir, filename)
            self._file = open(path, 'wb')
            self._writer = McapWriter(self._file)
            self._writer.start(profile='ros2', library='teleop_episode_recorder')
            self._channels = {}
            # Register all schemas and channels
            schema_cache = {}
            for topic, msg_type, schema_text in self.TOPICS:
                if msg_type not in schema_cache:
                    schema_cache[msg_type] = self._writer.register_schema(
                        name=msg_type,
                        encoding='ros2msg',
                        data=schema_text.encode('utf-8'),
                    )
                self._channels[topic] = self._writer.register_channel(
                    topic=topic,
                    message_encoding='cdr',
                    schema_id=schema_cache[msg_type],
                )
            logging.info(f"[Recorder] >> Episode {self.episode_count} started -> {path}")

    def end_episode(self):
        """End the current episode and write it to disk."""
        with self._lock:
            self._close_writer_locked()

    def close(self):
        """End any active episode -- call on shutdown."""
        self.end_episode()
        logging.info(f"[Recorder] Session closed. {self.episode_count} episode(s) total.")

    # ---- per-tick recording ----
    def step(self, *,
             timestamp,
             # Robot state
             robot_q, robot_velocity, robot_torque, robot_current, robot_ft,
             # Commanded targets
             right_cmd_q, left_cmd_q,
             right_gripper, left_gripper,
             # Master arm state
             ma_q, ma_qvel, ma_gravity_term,
             ma_torque_joint, ma_current_joint,
             # Computed signals
             commanded_torque, is_collision,
             right_enabled, left_enabled):
        """Serialize and write one timestep to the active MCAP episode.  No-op if idle."""
        with self._lock:
            if self._writer is None:
                return

            ts_ns = int(timestamp * 1e9)
            header = _make_header(timestamp)

            # Slice full robot state into per-arm arrays
            r_idx = self._right_arm_idx
            l_idx = self._left_arm_idx

            # 1) /right/joint_states — right arm state
            msg = _make_joint_state(
                header,
                names=[f'right_joint_{i}' for i in range(len(r_idx))],
                position=robot_q[r_idx],
                velocity=robot_velocity[r_idx],
                effort=robot_torque[r_idx],
            )
            self._write_msg('/right/joint_states', ts_ns,
                            msg, 'sensor_msgs/msg/JointState')

            # 2) /left/joint_states — left arm state
            msg = _make_joint_state(
                header,
                names=[f'left_joint_{i}' for i in range(len(l_idx))],
                position=robot_q[l_idx],
                velocity=robot_velocity[l_idx],
                effort=robot_torque[l_idx],
            )
            self._write_msg('/left/joint_states', ts_ns,
                            msg, 'sensor_msgs/msg/JointState')

            # 3) /right/joint_commands — right arm commanded positions
            msg = _make_joint_state(
                header,
                names=[f'right_joint_{i}' for i in range(len(right_cmd_q))],
                position=right_cmd_q,
                velocity=np.zeros(len(right_cmd_q)),
                effort=np.zeros(len(right_cmd_q)),
            )
            self._write_msg('/right/joint_commands', ts_ns,
                            msg, 'sensor_msgs/msg/JointState')

            # 4) /left/joint_commands — left arm commanded positions
            msg = _make_joint_state(
                header,
                names=[f'left_joint_{i}' for i in range(len(left_cmd_q))],
                position=left_cmd_q,
                velocity=np.zeros(len(left_cmd_q)),
                effort=np.zeros(len(left_cmd_q)),
            )
            self._write_msg('/left/joint_commands', ts_ns,
                            msg, 'sensor_msgs/msg/JointState')

            # 5) /right/force_torque_sensor_broadcaster/wrench — right arm F/T
            ft = np.array(robot_ft, dtype=np.float64)
            force = ft[:3] if len(ft) >= 3 else np.zeros(3)
            torque_ft = ft[3:6] if len(ft) >= 6 else np.zeros(3)
            msg = _make_wrench_stamped(header, force, torque_ft)
            self._write_msg('/right/force_torque_sensor_broadcaster/wrench', ts_ns,
                            msg, 'geometry_msgs/msg/WrenchStamped')

            # 6) /left/force_torque_sensor_broadcaster/wrench — left arm F/T
            #    (RBY1 has a single F/T sensor; duplicate for format compatibility)
            msg = _make_wrench_stamped(header, force, torque_ft)
            self._write_msg('/left/force_torque_sensor_broadcaster/wrench', ts_ns,
                            msg, 'geometry_msgs/msg/WrenchStamped')

            # 7) /right_hand/joint_states — right gripper state
            msg = _make_joint_state(
                header,
                names=['right_gripper'],
                position=[right_gripper],
                velocity=[0.0],
                effort=[0.0],
            )
            self._write_msg('/right_hand/joint_states', ts_ns,
                            msg, 'sensor_msgs/msg/JointState')

            # 8) /left_hand/joint_states — left gripper state
            msg = _make_joint_state(
                header,
                names=['left_gripper'],
                position=[left_gripper],
                velocity=[0.0],
                effort=[0.0],
            )
            self._write_msg('/left_hand/joint_states', ts_ns,
                            msg, 'sensor_msgs/msg/JointState')

            # 9) /master_arm/joint_states — master arm state (RBY1-specific)
            ma_names = [f'ma_joint_{i}' for i in range(len(ma_q))]
            msg = _make_joint_state(header, ma_names, ma_q, ma_qvel, ma_gravity_term)
            self._write_msg('/master_arm/joint_states', ts_ns,
                            msg, 'sensor_msgs/msg/JointState')

            # 10) /teleop/status — scalar flags & extra arrays packed together
            status_data = np.concatenate([
                [is_collision, right_enabled, left_enabled],
                np.array(robot_current, dtype=np.float64),
                np.array(ma_torque_joint, dtype=np.float64),
                np.array(ma_current_joint, dtype=np.float64),
                np.array(commanded_torque, dtype=np.float64),
            ])
            msg = _make_float64_multi_array(status_data)
            self._write_msg('/teleop/status', ts_ns,
                            msg, 'std_msgs/msg/Float64MultiArray')

    # ---- internal helpers ----
    def _write_msg(self, topic, ts_ns, msg, typename):
        """CDR-serialize a rosbags message and write it to the MCAP channel."""
        data = _typestore.serialize_cdr(msg, typename)
        self._writer.add_message(
            channel_id=self._channels[topic],
            log_time=ts_ns,
            publish_time=ts_ns,
            data=data,
        )

    def _close_writer_locked(self):
        """Finalize and close the current MCAP writer if active."""
        if self._writer is None:
            return
        try:
            self._writer.finish()
        except Exception as e:
            logging.warning(f"[Recorder] Error finishing MCAP: {e}")
        self._file.close()
        self._writer = None
        self._file = None
        self._channels = {}
        logging.info(f"[Recorder] -- Episode {self.episode_count} saved")


def _getch():
    """Read a keypress on Linux. Arrow keys return 'UP','DOWN','LEFT','RIGHT'."""
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            # Arrow keys send \x1b[X or \x1bOX (application mode)
            rest = os.read(fd, 2).decode("utf-8", errors="ignore")
            if len(rest) == 2 and rest[0] in ("[", "O"):
                return ARROW_KEYS.get(rest[1], "")
            return ""
        return ch.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Global reference set by main() so keyboard_listener can drive episode lifecycle
_recorder: TeleopRecorder = None


def keyboard_listener():
    """Background thread: Up/Down toggle arm teleop, Left/Right toggle grippers, Ctrl+C to quit."""
    global right_arm_enabled, left_arm_enabled, right_gripper_open, left_gripper_open
    while True:
        try:
            key = _getch()
            if key == "\x03":  # Ctrl+C in raw mode
                os.kill(os.getpid(), signal.SIGINT)
                break
            elif key == "UP":
                right_arm_enabled = True
                left_arm_enabled = True
                logging.info("Teleop ENABLED (both arms)")
                if _recorder is not None:
                    _recorder.new_episode()
            elif key == "DOWN":
                right_arm_enabled = False
                left_arm_enabled = False
                logging.info("Teleop DISABLED (both arms)")
                if _recorder is not None:
                    _recorder.end_episode()
            elif key == "RIGHT":
                right_gripper_open = not right_gripper_open
                logging.info(f"Right gripper {'OPEN' if right_gripper_open else 'CLOSED'}")
            elif key == "LEFT":
                left_gripper_open = not left_gripper_open
                logging.info(f"Left gripper {'OPEN' if left_gripper_open else 'CLOSED'}")
        except Exception:
            break

GRIPPER_DIRECTION = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class Pose:
    toros: np.typing.NDArray
    right_arm: np.typing.NDArray
    left_arm: np.typing.NDArray


class Settings:
    master_arm_loop_period = 1 / 100

    impedance_stiffness = 50
    impedance_damping_ratio = 1.0
    impedance_torque_limit = 30.0


READY_POSE = {
    "A": Pose(
        toros=np.deg2rad([0.0, 0.0, -0.0, 0.0, 0.0, 0.0]),
        right_arm=np.deg2rad([0.0, -5.0, 0.0, -80.0, 0.0, 60.0, 0.0]),
        left_arm=np.deg2rad([0.0, 5.0, 0.0, -80.0, 0.0, 60.0, 0.0]),
    ),
    "M": Pose(
        toros=np.deg2rad([0.0, 45.0, 0.0, 0.0, 0.0, 0.0]),
        right_arm=np.deg2rad([0.0, -5.0, 0.0, -80.0, 0.0, 60.0, 0.0]),
        left_arm=np.deg2rad([0.0, 5.0, 0.0, -80.0, 0.0, 60.0, 0.0]),
    ),
}


class Gripper:
    """
    Class for gripper
    """

    def __init__(self):
        self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self.bus.open_port()
        self.bus.set_baud_rate(2_000_000)
        self.bus.set_torque_constant([1, 1])
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q: np.typing.NDArray = None
        self._running = False
        self._thread = None

    def initialize(self, verbose=False):
        rv = True
        for dev_id in [0, 1]:
            if not self.bus.ping(dev_id):
                if verbose:
                    logging.error(f"Dynamixel ID {dev_id} is not active")
                rv = False
            else:
                if verbose:
                    logging.info(f"Dynamixel ID {dev_id} is active")
        if rv:
            logging.info("Servo on gripper")
            self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
        return rv

    def set_operating_mode(self, mode):
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])

    def homing(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        direction = 0
        q = np.array([0, 0], dtype=np.float64)
        prev_q = np.array([0, 0], dtype=np.float64)
        counter = 0
        while direction < 2:
            self.bus.group_sync_write_send_torque(
                [(dev_id, 0.5 * (1 if direction == 0 else -1)) for dev_id in [0, 1]]
            )
            rv = self.bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            self.min_q = np.minimum(self.min_q, q)
            self.max_q = np.maximum(self.max_q, q)
            if np.array_equal(prev_q, q):
                counter += 1
            prev_q = q
            if counter >= 30:
                direction += 1
                counter = 0
            time.sleep(0.1)
        return True

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def loop(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        while self._running:
            if self.target_q is not None:
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                )
            time.sleep(0.1)

    def set_target(self, normalized_q):
        # self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            logging.error("Cannot set target. min_q or max_q is not valid.")
            return

        if GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q


def joint_position_command_builder(
    pose: Pose, minimum_time, control_hold_time=0, position_mode=True
):
    right_arm_builder = (
        rby.JointPositionCommandBuilder()
        if position_mode
        else rby.JointImpedanceControlCommandBuilder()
    )
    (
        right_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(pose.right_arm)
        .set_minimum_time(minimum_time)
    )
    if not position_mode:
        (
            right_arm_builder.set_stiffness(
                [Settings.impedance_stiffness] * len(pose.right_arm)
            )
            .set_damping_ratio(Settings.impedance_damping_ratio)
            .set_torque_limit([Settings.impedance_torque_limit] * len(pose.right_arm))
        )

    left_arm_builder = (
        rby.JointPositionCommandBuilder()
        if position_mode
        else rby.JointImpedanceControlCommandBuilder()
    )
    (
        left_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(pose.left_arm)
        .set_minimum_time(minimum_time)
    )
    if not position_mode:
        (
            left_arm_builder.set_stiffness(
                [Settings.impedance_stiffness] * len(pose.left_arm)
            )
            .set_damping_ratio(Settings.impedance_damping_ratio)
            .set_torque_limit([Settings.impedance_torque_limit] * len(pose.left_arm))
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
                )
                .set_position(pose.toros)
                .set_minimum_time(minimum_time)
            )
            .set_right_arm_command(right_arm_builder)
            .set_left_arm_command(left_arm_builder)
        )
    )


def move_j(robot: Union[rby.Robot_A, rby.Robot_M], pose: Pose, minimum_time=5.0):
    handler = robot.send_command(joint_position_command_builder(pose, minimum_time))
    return handler.get() == rby.RobotCommandFeedback.FinishCode.Ok


def main(address, model, power, servo, control_mode):
    # ===== SETUP ROBOT =====
    robot = rby.create_robot(address, model)
    if not robot.connect():
        logging.error(f"Failed to connect robot {address}")
        exit(1)
    supported_model = ["A", "M"]
    supported_control_mode = ["position", "impedance"]
    model = robot.model()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state([], model.robot_joint_names)
    robot_q = None
    robot_max_q = dyn_model.get_limit_q_upper(dyn_state)
    robot_min_q = dyn_model.get_limit_q_lower(dyn_state)
    robot_max_qdot = dyn_model.get_limit_qdot_upper(dyn_state)
    robot_max_qddot = dyn_model.get_limit_qddot_upper(dyn_state)

    if control_mode == "impedance":
        robot_max_qdot[model.right_arm_idx[-1]] *= 10
        robot_max_qdot[model.left_arm_idx[-1]] *= 10

    if not model.model_name in supported_model:
        logging.error(
            f"Model {model.model_name} not supported (Current supported model is {supported_model})"
        )
        exit(1)
    if not control_mode in supported_control_mode:
        logging.error(
            f"Control mode {control_mode} not supported (Current supported control mode is {supported_control_mode})"
        )
        exit(1)
    position_mode = control_mode == "position"
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            logging.error(f"Failed to turn power ({power}) on")
            exit(1)
    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            logging.error(f"Failed to servo ({servo}) on")
            exit(1)
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        logging.error(f"Failed to enable control manager")
        exit(1)
    for arm in ["right", "left"]:
        if not robot.set_tool_flange_output_voltage(arm, 12):
            logging.error(f"Failed to set tool flange output voltage ({arm}) as 12v")
            exit(1)
    robot.set_parameter("joint_position_command.cutoff_frequency", "3")
    move_j(robot, READY_POSE[model.model_name], 5)

    # Extended robot state for recording
    robot_velocity = None
    robot_torque = None
    robot_current = None
    robot_ft = None

    def robot_state_callback(state: rby.RobotState_A):
        nonlocal robot_q, robot_velocity, robot_torque, robot_current, robot_ft
        robot_q = state.position
        # Capture force/torque data if the SDK exposes it (safe fallbacks)
        robot_velocity = getattr(state, 'velocity', np.zeros_like(robot_q))
        robot_torque   = getattr(state, 'torque',   np.zeros_like(robot_q))
        robot_current  = getattr(state, 'current',  np.zeros_like(robot_q))
        robot_ft       = getattr(state, 'ft',       np.zeros(6))

    robot.start_state_update(robot_state_callback, 1 / Settings.master_arm_loop_period)

    # ===== SETUP GRIPPER =====
    gripper = Gripper()
    if not gripper.initialize():
        logging.error("Failed to initialize gripper")
        robot.stop_state_update()
        robot.power_off("12v")
        exit(1)
    gripper.homing()
    gripper.start()

    # ===== SETUP MASTER ARM =====
    rby.upc.initialize_device(rby.upc.MasterArmDeviceName)
    master_arm_model = f"{os.path.dirname(os.path.realpath(__file__))}/models/master_arm/model.urdf"
    master_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
    master_arm.set_model_path(master_arm_model)
    master_arm.set_control_period(Settings.master_arm_loop_period)
    active_ids = master_arm.initialize(verbose=False)
    #if len(active_ids) != rby.upc.MasterArm.DeviceCount:
        #logging.error(
            #f"Mismatch in the number of devices detected for RBY Master Arm (active devices: {active_ids})"
        #)
        #exit(1)

    ma_q_limit_barrier = 0.5
    ma_min_q = np.deg2rad(
        [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
    )
    ma_max_q = np.deg2rad(
        [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
    )
    ma_torque_limit = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
    ma_viscous_gain = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)

    right_q = None
    left_q = None
    ma_state_ready = threading.Event()   # set once we get the first master arm callback
    initial_move_done = False            # gates control loop until initial positioning finishes

    right_minimum_time = 1.0
    left_minimum_time = 1.0
    stream = robot.create_command_stream(priority=1)  # TODO

    # ===== SETUP RECORDER =====
    global _recorder
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, "TeleOpRecorderData")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # List existing session folders
    existing = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    print("\n===== SESSION SETUP =====")
    if existing:
        print("Existing sessions:")
        for i, name in enumerate(existing, 1):
            ep_count = len([f for f in os.listdir(os.path.join(data_dir, name)) if f.endswith('.mcap')])
            print(f"  [{i}] {name}  ({ep_count} episode{'s' if ep_count != 1 else ''})")
        print(f"\nEnter a number to resume a session, or type a new name to create one.")
    else:
        print("No existing sessions found.")
        print("Enter a name for the new session folder.")

    choice = input("\n> ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(existing):
        session_name = existing[int(choice) - 1]
        print(f"Resuming session: {session_name}")
    elif choice:
        session_name = choice
        print(f"Creating new session: {session_name}")
    else:
        session_name = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Creating new session: {session_name}")
    recorder = TeleopRecorder(
        session_name=session_name,
        right_arm_idx=model.right_arm_idx,
        left_arm_idx=model.left_arm_idx,
    )
    _recorder = recorder      # expose to keyboard_listener for episode lifecycle

    log_count = 0

    def master_arm_control_loop(state: rby.upc.MasterArm.State):
        nonlocal position_mode, right_q, left_q, right_minimum_time, left_minimum_time, log_count

        # Capture the first master arm state for initial robot positioning
        if right_q is None:
            right_q = np.clip(
                state.q_joint[0:7],
                robot_min_q[model.right_arm_idx],
                robot_max_q[model.right_arm_idx],
            )
        if left_q is None:
            left_q = np.clip(
                state.q_joint[7:14],
                robot_min_q[model.left_arm_idx],
                robot_max_q[model.left_arm_idx],
            )
        if not ma_state_ready.is_set():
            ma_state_ready.set()

        ma_input = rby.upc.MasterArm.ControlInput()

        log_count += 1
        if log_count % round(1 / Settings.master_arm_loop_period) == 0:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            r_arm = "ON " if right_arm_enabled else "OFF"
            l_arm = "ON " if left_arm_enabled else "OFF"
            r_grip = "OPEN  " if right_gripper_open else "CLOSED"
            l_grip = "OPEN  " if left_gripper_open else "CLOSED"
            print(
                f"\r[{ts}] R-arm:{r_arm} L-arm:{l_arm} | R-grip:{r_grip} L-grip:{l_grip}   ",
                end="", flush=True,
            )
            log_count = 0
        gripper.set_target(
            np.array(
                [float(right_gripper_open), float(left_gripper_open)]
            )
        )
        # ===== CALCULATE MASTER ARM COMMAND =====
        torque = (
            state.gravity_term
            + ma_q_limit_barrier
            * (
                np.maximum(ma_min_q - state.q_joint, 0)
                + np.minimum(ma_max_q - state.q_joint, 0)
            )
            + ma_viscous_gain * state.qvel_joint
        )
        torque = np.clip(torque, -ma_torque_limit, ma_torque_limit)
        if right_arm_enabled:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[0:7] = torque[0:7] * 0.6
            right_q = state.q_joint[0:7]
        else:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[0:7] = ma_torque_limit[0:7]
            ma_input.target_position[0:7] = right_q

        if left_arm_enabled:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[7:14] = torque[7:14] * 0.6
            left_q = state.q_joint[7:14]
        else:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[7:14] = ma_torque_limit[7:14]
            ma_input.target_position[7:14] = left_q

        # Check whether target configure is in collision
        q = robot_q.copy()
        q[model.right_arm_idx] = right_q
        q[model.left_arm_idx] = left_q
        dyn_state.set_q(q)
        dyn_model.compute_forward_kinematics(dyn_state)
        is_collision = (
            dyn_model.detect_collisions_or_nearest_links(dyn_state, 1)[0].distance
            < 0.02
        )

        # ===== BUILD ROBOT COMMAND =====
        rc = rby.BodyComponentBasedCommandBuilder()
        if right_arm_enabled and not is_collision:
            right_minimum_time -= Settings.master_arm_loop_period
            right_minimum_time = max(
                right_minimum_time, Settings.master_arm_loop_period * 1.01
            )
            right_arm_builder = (
                rby.JointPositionCommandBuilder()
                if position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                right_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(
                    np.clip(
                        right_q,
                        robot_min_q[model.right_arm_idx],
                        robot_max_q[model.right_arm_idx],
                    )
                )
                .set_velocity_limit(robot_max_qdot[model.right_arm_idx])
                .set_acceleration_limit(robot_max_qddot[model.right_arm_idx] * 30)
                .set_minimum_time(right_minimum_time)
            )
            if not position_mode:
                (
                    right_arm_builder.set_stiffness(
                        [Settings.impedance_stiffness] * len(model.right_arm_idx)
                    )
                    .set_damping_ratio(Settings.impedance_damping_ratio)
                    .set_torque_limit(
                        [Settings.impedance_torque_limit] * len(model.right_arm_idx)
                    )
                )
            rc.set_right_arm_command(right_arm_builder)
        else:
            right_minimum_time = 0.8

        if left_arm_enabled and not is_collision:
            left_minimum_time -= Settings.master_arm_loop_period
            left_minimum_time = max(
                left_minimum_time, Settings.master_arm_loop_period * 1.01
            )
            left_arm_builder = (
                rby.JointPositionCommandBuilder()
                if position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                left_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(
                    np.clip(
                        left_q,
                        robot_min_q[model.left_arm_idx],
                        robot_max_q[model.left_arm_idx],
                    )
                )
                .set_velocity_limit(robot_max_qdot[model.left_arm_idx])
                .set_acceleration_limit(robot_max_qddot[model.left_arm_idx] * 30)
                .set_minimum_time(left_minimum_time)
            )
            if not position_mode:
                (
                    left_arm_builder.set_stiffness(
                        [Settings.impedance_stiffness] * len(model.left_arm_idx)
                    )
                    .set_damping_ratio(Settings.impedance_damping_ratio)
                    .set_torque_limit(
                        [Settings.impedance_torque_limit] * len(model.left_arm_idx)
                    )
                )
            rc.set_left_arm_command(left_arm_builder)
        else:
            left_minimum_time = 0.8
        if initial_move_done:
            stream.send_command(
                rby.RobotCommandBuilder().set_command(
                    rby.ComponentBasedCommandBuilder().set_body_command(rc)
                )
            )

        # ===== RECORD DATA FOR IMITATION LEARNING =====
        recorder.step(
            timestamp=time.time(),
            # Robot state (with safe fallbacks for missing fields)
            robot_q=robot_q,
            robot_velocity=robot_velocity if robot_velocity is not None else np.zeros_like(robot_q),
            robot_torque=robot_torque     if robot_torque   is not None else np.zeros_like(robot_q),
            robot_current=robot_current   if robot_current  is not None else np.zeros_like(robot_q),
            robot_ft=robot_ft             if robot_ft       is not None else np.zeros(6),
            # Commanded targets
            right_cmd_q=right_q,
            left_cmd_q=left_q,
            right_gripper=float(right_gripper_open),
            left_gripper=float(left_gripper_open),
            # Master arm state
            ma_q=state.q_joint.copy(),
            ma_qvel=state.qvel_joint.copy(),
            ma_gravity_term=state.gravity_term.copy(),
            ma_torque_joint=getattr(state, 'torque_joint', np.zeros(14)),
            ma_current_joint=getattr(state, 'current_joint', np.zeros(14)),
            # Computed signals
            commanded_torque=torque,
            is_collision=is_collision,
            right_enabled=right_arm_enabled,
            left_enabled=left_arm_enabled,
        )

        return ma_input

    # ===== START KEYBOARD LISTENER =====
    logging.info("Keyboard controls: UP = enable teleop, DOWN = disable teleop, LEFT = toggle left gripper, RIGHT = toggle right gripper")
    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    master_arm.start_control(master_arm_control_loop)

    # Wait for the first master arm callback to get the current arm positions
    logging.info("Waiting for master arm state...")
    ma_state_ready.wait(timeout=10)
    if right_q is not None and left_q is not None:
        initial_pose = Pose(
            toros=READY_POSE[model.model_name].toros,
            right_arm=right_q.copy(),
            left_arm=left_q.copy(),
        )
        stream.send_command(
            joint_position_command_builder(
                initial_pose,
                minimum_time=5,
                control_hold_time=1e6,
                position_mode=position_mode,
            )
        )
        time.sleep(5)  # wait for the initial move to complete
        initial_move_done = True
        logging.info("Robot arms moved to master arm start position.")
    else:
        logging.warning("Could not read master arm state, using default ready pose.")
        stream.send_command(
            joint_position_command_builder(
                READY_POSE[model.model_name],
                minimum_time=5,
                control_hold_time=1e6,
                position_mode=position_mode,
            )
        )
        time.sleep(5)
        initial_move_done = True

    # ===== SETUP SIGNAL =====
    def handler(signum, frame):
        recorder.close()  # flush any active episode before shutdown
        robot.stop_state_update()
        master_arm.stop_control()
        robot.cancel_control()
        time.sleep(0.5)

        robot.disable_control_manager()
        robot.power_off("12v")
        gripper.stop()
        exit(1)

    signal.signal(signal.SIGINT, handler)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="17_teleoperation_with_joint_mapping")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Regex pattern for power device names (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default="torso_.*|right_arm_.*|left_arm_.*",
        help="Regex pattern for servo names (default: 'torso_.*|right_arm_.*|left_arm_.*')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="position",
        choices=["position", "impedance"],
        help="Control mode to use: 'position' or 'impedance' (default: 'position')",
    )
    args = parser.parse_args()

    main(args.address, args.model, args.power, args.servo, args.mode)
