# -*- coding: utf-8 -*-
"""
MCAP Episode Replay for RBY1

Reads recorded MCAP episode files and replays the joint commands on the robot.
Pairs with teleop_episode_recording.py — uses the same robot setup, command
builders, and MCAP topic structure.

Usage:
    python3 teleop_episode_replay.py --address 192.168.30.1:50051
    python3 teleop_episode_replay.py --address 192.168.30.1:50051 --speed 0.5
    python3 teleop_episode_replay.py --address 192.168.30.1:50051 --mcap /path/to/episode.mcap
"""

import rby1_sdk as rby
import numpy as np
import os
import sys
import time
import logging
import argparse
import signal
import datetime
from mcap.reader import make_reader as McapReader
from rosbags.typesys import Stores, get_typestore
from typing import *
from dataclasses import dataclass

# ===== ROS2 TYPE SYSTEM (standalone -- no ROS2 install needed) =====
_typestore = get_typestore(Stores.ROS2_HUMBLE)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

GRIPPER_DIRECTION = False


@dataclass
class Pose:
    toros: np.typing.NDArray
    right_arm: np.typing.NDArray
    left_arm: np.typing.NDArray


class Settings:
    replay_loop_period = 1 / 100  # 100 Hz, matches recording rate

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
    """Class for gripper (same as teleop_episode_recording.py)"""

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
        import threading
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
        import threading
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


def move_j(robot, pose: Pose, minimum_time=5.0):
    handler = robot.send_command(joint_position_command_builder(pose, minimum_time))
    return handler.get() == rby.RobotCommandFeedback.FinishCode.Ok


# ===== MCAP EPISODE LOADER =====
def load_episode(mcap_path):
    """Load an MCAP episode and return time-sorted command frames.

    Returns a list of dicts, each with:
      - timestamp_ns: int
      - right_cmd_q: np.array (7,) or None
      - left_cmd_q: np.array (7,) or None
      - right_gripper: float or None
      - left_gripper: float or None
    """
    # Collect all messages grouped by timestamp
    messages = {}  # timestamp_ns -> {topic: msg}

    with open(mcap_path, 'rb') as f:
        reader = McapReader(f)
        for schema, channel, message in reader.iter_messages():
            ts = message.log_time
            if ts not in messages:
                messages[ts] = {}

            # Deserialize the CDR message
            msg = _typestore.deserialize_cdr(message.data, schema.name)
            messages[ts][channel.topic] = msg

    # Sort by timestamp and build command frames
    frames = []
    for ts_ns in sorted(messages.keys()):
        topic_msgs = messages[ts_ns]
        frame = {
            'timestamp_ns': ts_ns,
            'right_cmd_q': None,
            'left_cmd_q': None,
            'right_gripper': None,
            'left_gripper': None,
        }

        if '/right/joint_commands' in topic_msgs:
            msg = topic_msgs['/right/joint_commands']
            frame['right_cmd_q'] = np.array(msg.position, dtype=np.float64)

        if '/left/joint_commands' in topic_msgs:
            msg = topic_msgs['/left/joint_commands']
            frame['left_cmd_q'] = np.array(msg.position, dtype=np.float64)

        if '/right_hand/joint_states' in topic_msgs:
            msg = topic_msgs['/right_hand/joint_states']
            frame['right_gripper'] = float(msg.position[0]) if len(msg.position) > 0 else 0.0

        if '/left_hand/joint_states' in topic_msgs:
            msg = topic_msgs['/left_hand/joint_states']
            frame['left_gripper'] = float(msg.position[0]) if len(msg.position) > 0 else 0.0

        # Only include frames that have at least one command
        if frame['right_cmd_q'] is not None or frame['left_cmd_q'] is not None:
            frames.append(frame)

    return frames


def select_episode():
    """Interactive episode selection from TeleOpRecorderData."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, "TeleOpRecorderData")

    if not os.path.exists(data_dir):
        logging.error(f"No TeleOpRecorderData directory found at: {data_dir}")
        return None

    # List sessions
    sessions = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if not sessions:
        logging.error("No session folders found in TeleOpRecorderData/")
        return None

    print("\n===== EPISODE REPLAY =====")
    print("Available sessions:")
    for i, name in enumerate(sessions, 1):
        ep_count = len([f for f in os.listdir(os.path.join(data_dir, name)) if f.endswith('.mcap')])
        print(f"  [{i}] {name}  ({ep_count} episode{'s' if ep_count != 1 else ''})")

    choice = input("\nSelect a session number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(sessions)):
        logging.error("Invalid session selection.")
        return None

    session_dir = os.path.join(data_dir, sessions[int(choice) - 1])

    # List episodes in that session
    episodes = sorted([f for f in os.listdir(session_dir) if f.endswith('.mcap')])
    if not episodes:
        logging.error(f"No .mcap episodes found in {sessions[int(choice) - 1]}/")
        return None

    print(f"\nEpisodes in '{sessions[int(choice) - 1]}':")
    for i, ep in enumerate(episodes, 1):
        size_mb = os.path.getsize(os.path.join(session_dir, ep)) / (1024 * 1024)
        print(f"  [{i}] {ep}  ({size_mb:.1f} MB)")

    ep_choice = input("\nSelect an episode number (or 'all' to play all): ").strip()

    if ep_choice.lower() == 'all':
        return [os.path.join(session_dir, ep) for ep in episodes]
    elif ep_choice.isdigit() and 1 <= int(ep_choice) <= len(episodes):
        return [os.path.join(session_dir, episodes[int(ep_choice) - 1])]
    else:
        logging.error("Invalid episode selection.")
        return None


def main(address, model_name, power, servo, control_mode, speed_scale, mcap_path):
    # ===== SETUP ROBOT =====
    robot = rby.create_robot(address, model_name)
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

    if model.model_name not in supported_model:
        logging.error(f"Model {model.model_name} not supported")
        exit(1)
    if control_mode not in supported_control_mode:
        logging.error(f"Control mode {control_mode} not supported")
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
        logging.error("Failed to enable control manager")
        exit(1)

    for arm in ["right", "left"]:
        if not robot.set_tool_flange_output_voltage(arm, 12):
            logging.error(f"Failed to set tool flange output voltage ({arm}) as 12v")
            exit(1)

    robot.set_parameter("joint_position_command.cutoff_frequency", "3")

    # Track robot state for collision detection
    def robot_state_callback(state):
        nonlocal robot_q
        robot_q = state.position

    robot.start_state_update(robot_state_callback, 1 / Settings.replay_loop_period)

    # ===== SETUP GRIPPER =====
    gripper = Gripper()
    if not gripper.initialize():
        logging.error("Failed to initialize gripper")
        robot.stop_state_update()
        robot.power_off("12v")
        exit(1)
    gripper.homing()
    gripper.start()

    # ===== SIGNAL HANDLER =====
    def handler(signum, frame):
        logging.info("Shutting down replay...")
        robot.stop_state_update()
        robot.cancel_control()
        time.sleep(0.5)
        robot.disable_control_manager()
        robot.power_off("12v")
        gripper.stop()
        exit(0)

    signal.signal(signal.SIGINT, handler)

    # ===== SELECT EPISODE(S) =====
    if mcap_path:
        episode_paths = [mcap_path]
    else:
        episode_paths = select_episode()
        if not episode_paths:
            handler(None, None)
            return

    # ===== REPLAY EACH EPISODE =====
    for ep_idx, ep_path in enumerate(episode_paths):
        ep_name = os.path.basename(ep_path)
        logging.info(f"Loading episode {ep_idx + 1}/{len(episode_paths)}: {ep_name}")

        frames = load_episode(ep_path)
        if not frames:
            logging.warning(f"No command frames found in {ep_name}, skipping.")
            continue

        total_duration_s = (frames[-1]['timestamp_ns'] - frames[0]['timestamp_ns']) / 1e9
        logging.info(f"  {len(frames)} frames, {total_duration_s:.2f}s duration, "
                     f"speed scale: {speed_scale}x")

        # Move to the first frame's position slowly (safe approach)
        first_right = frames[0].get('right_cmd_q')
        first_left = frames[0].get('left_cmd_q')
        if first_right is None or first_left is None:
            # Find the first frame with both arms
            for f in frames:
                if first_right is None and f['right_cmd_q'] is not None:
                    first_right = f['right_cmd_q']
                if first_left is None and f['left_cmd_q'] is not None:
                    first_left = f['left_cmd_q']
                if first_right is not None and first_left is not None:
                    break

        if first_right is None:
            first_right = READY_POSE[model.model_name].right_arm
        if first_left is None:
            first_left = READY_POSE[model.model_name].left_arm

        start_pose = Pose(
            toros=READY_POSE[model.model_name].toros,
            right_arm=np.clip(first_right, robot_min_q[model.right_arm_idx],
                              robot_max_q[model.right_arm_idx]),
            left_arm=np.clip(first_left, robot_min_q[model.left_arm_idx],
                             robot_max_q[model.left_arm_idx]),
        )

        logging.info("  Moving to episode start position...")
        move_j(robot, start_pose, minimum_time=5.0)
        time.sleep(0.5)

        # Confirm before starting replay
        input(f"  Press ENTER to start replay (Ctrl+C to abort)...")
        logging.info(f"  Replaying...")

        # Create command stream
        stream = robot.create_command_stream(priority=1)

        # Send initial hold command
        stream.send_command(
            joint_position_command_builder(
                start_pose,
                minimum_time=0.5,
                control_hold_time=1e6,
                position_mode=position_mode,
            )
        )
        time.sleep(0.5)

        # ===== MAIN REPLAY LOOP =====
        collision_count = 0
        skipped_count = 0
        start_time = time.time()

        for i, frame in enumerate(frames):
            # Compute target wall-clock time for this frame
            frame_offset_s = (frame['timestamp_ns'] - frames[0]['timestamp_ns']) / 1e9
            target_time = start_time + (frame_offset_s / speed_scale)

            # Wait until it's time to send this frame
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)

            right_q = frame.get('right_cmd_q')
            left_q = frame.get('left_cmd_q')

            # Use previous values for any missing arm
            if right_q is None:
                right_q = start_pose.right_arm
            if left_q is None:
                left_q = start_pose.left_arm

            # Clip to joint limits
            right_q = np.clip(right_q, robot_min_q[model.right_arm_idx],
                              robot_max_q[model.right_arm_idx])
            left_q = np.clip(left_q, robot_min_q[model.left_arm_idx],
                             robot_max_q[model.left_arm_idx])

            # Collision detection
            if robot_q is not None:
                q = robot_q.copy()
                q[model.right_arm_idx] = right_q
                q[model.left_arm_idx] = left_q
                dyn_state.set_q(q)
                dyn_model.compute_forward_kinematics(dyn_state)
                is_collision = (
                    dyn_model.detect_collisions_or_nearest_links(dyn_state, 1)[0].distance
                    < 0.02
                )
                if is_collision:
                    collision_count += 1
                    skipped_count += 1
                    continue

            # Build and send command
            loop_period = Settings.replay_loop_period / speed_scale
            rc = rby.BodyComponentBasedCommandBuilder()

            right_arm_builder = (
                rby.JointPositionCommandBuilder()
                if position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                right_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(right_q)
                .set_velocity_limit(robot_max_qdot[model.right_arm_idx])
                .set_acceleration_limit(robot_max_qddot[model.right_arm_idx] * 30)
                .set_minimum_time(loop_period * 1.01)
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

            left_arm_builder = (
                rby.JointPositionCommandBuilder()
                if position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                left_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(left_q)
                .set_velocity_limit(robot_max_qdot[model.left_arm_idx])
                .set_acceleration_limit(robot_max_qddot[model.left_arm_idx] * 30)
                .set_minimum_time(loop_period * 1.01)
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

            stream.send_command(
                rby.RobotCommandBuilder().set_command(
                    rby.ComponentBasedCommandBuilder().set_body_command(rc)
                )
            )

            # Drive grippers
            r_grip = frame.get('right_gripper', 0.0)
            l_grip = frame.get('left_gripper', 0.0)
            if r_grip is not None and l_grip is not None:
                gripper.set_target(np.array([r_grip, l_grip]))

            # Status display (every 100 frames)
            if i % 100 == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / len(frames) * 100
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"\r[{ts}] Frame {i+1}/{len(frames)} ({progress:.0f}%) "
                    f"| Elapsed: {elapsed:.1f}s "
                    f"| Collisions skipped: {collision_count}   ",
                    end="", flush=True,
                )

        # Episode complete
        print()  # newline after progress
        actual_duration = time.time() - start_time
        logging.info(
            f"  Episode complete. {len(frames)} frames in {actual_duration:.2f}s "
            f"({collision_count} collision{'s' if collision_count != 1 else ''} skipped)"
        )

        # If more episodes to play, pause between them
        if ep_idx < len(episode_paths) - 1:
            input(f"\n  Press ENTER to continue to next episode (Ctrl+C to stop)...")

    # ===== RETURN TO READY POSE AND STAY ALIVE =====
    logging.info("Replay complete. Returning to ready pose...")
    move_j(robot, READY_POSE[model.model_name], 5)
    logging.info("Robot is idle. Press Ctrl+C to shut down.")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCAP Episode Replay for RBY1")
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
        help="Regex pattern for servo names",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="position",
        choices=["position", "impedance"],
        help="Control mode (default: 'position')",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0, use 0.5 for half speed)",
    )
    parser.add_argument(
        "--mcap",
        type=str,
        default=None,
        help="Direct path to an .mcap file (skips interactive selection)",
    )
    args = parser.parse_args()

    main(args.address, args.model, args.power, args.servo, args.mode, args.speed, args.mcap)
