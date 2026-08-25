import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional
from boot_manager import AutoMenuBootManager

WR_BENCHMARKS = {
    "Breakfast Bends (3 Laps)": 42500.0,   # 42.50s (~14.16s pro Runde)
    "Breakfast Bends (Best Lap)": 13950.0, # 13.95s Einzelrunden-Rekord
    "Garden Jumps (3 Laps)": 73420.0,      # 73.42s
    "Desktop Destruction": 51200.0,        # 51.20s
}

ACTION_NAMES = [
    "NOOP",          # 0
    "ACCEL (B)",     # 1: Vollgas
    "BRAKE (C)",     # 2: Bremse / Rückwärts
    "STEER_L",       # 3: Lenken Links
    "STEER_R",       # 4: Lenken Rechts
    "ACCEL + LEFT",  # 5: Power-Slide / Drift Links
    "ACCEL + RIGHT", # 6: Power-Slide / Drift Rechts
    "BRAKE + LEFT",  # 7: Hard Turn / Handbremse Links
    "BRAKE + RIGHT"  # 8: Hard Turn / Handbremse Rechts
]

ACTION_BUTTONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0: NOOP
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1: B (Gas)
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 2: C (Bremse)
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 3: LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 4: RIGHT
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 5: B + LEFT
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 6: B + RIGHT
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], # 7: C + LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], # 8: C + RIGHT
]

class MicroMachinesEnvWrapper:
    def __init__(
        self,
        retro_env,
        seq_len: int = 16,
        frame_skip: int = 4,
        target_wr_ms: float = 42500.0,
        game_mode: str = "SUPER_LEAGUE_HARD"
    ):
        self.env = retro_env
        self.seq_len = seq_len
        self.frame_skip = frame_skip
        self.target_wr_ms = target_wr_ms
        self.game_mode = game_mode
        self.boot_manager = AutoMenuBootManager(mode=game_mode)

        self.frame_buffer = []
        self.ram_buffer = []
        self.prev_progress = 0.0
        self.lap_start_step = 0
        self.total_frames_rendered = 0
        self.cached_race_start_state = None
        self.is_first_boot = True
        self.last_valid_obs = None

    def preprocess_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        if raw_frame is None:
            if self.last_valid_obs is not None:
                raw_frame = self.last_valid_obs
            else:
                return np.zeros((84, 84, 1), dtype=np.float32)
        
        self.last_valid_obs = raw_frame
        if len(raw_frame.shape) == 3 and raw_frame.shape[2] == 3:
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
        elif len(raw_frame.shape) == 3 and raw_frame.shape[2] == 1:
            gray = raw_frame[:, :, 0]
        else:
            gray = raw_frame
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0)[:, :, np.newaxis]
        return normalized

    def extract_ram_state(self, info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not isinstance(info, dict):
            info = {}
        speed = float(info.get("speed", 0.0))
        lap = int(info.get("lap", 1))
        checkpoint = int(info.get("checkpoint", 0))
        max_checkpoints = float(info.get("max_checkpoints", 16.0))
        off_track = float(info.get("off_track", 0.0))
        points = int(info.get("points", 0))
        won = bool(info.get("won", False))

        progress = checkpoint / max(1.0, max_checkpoints)
        lap_time_ms = float(info.get("lap_time_ms", (self.total_frames_rendered - self.lap_start_step) * 16.666))

        ram_vector = np.zeros(16, dtype=np.float32)
        ram_vector[0] = speed / 100.0
        ram_vector[1] = progress
        ram_vector[2] = lap_time_ms / 60000.0
        ram_vector[3] = off_track
        ram_vector[4] = float(lap) / 3.0
        ram_vector[5] = float(points) / 100.0

        telemetry = {
            "speed": speed,
            "lap": lap,
            "checkpoint": checkpoint,
            "progress": progress,
            "lap_time_ms": lap_time_ms,
            "off_track": off_track > 0,
            "points": points,
            "won": won
        }
        return ram_vector, telemetry

    def reset(self, streamer=None, tracker=None):
        # 1. Schneller Restore auf den Startplatz
        if hasattr(self.env, "em") and self.cached_race_start_state is not None:
            self.env.em.set_state(self.cached_race_start_state)
            obs = self.last_valid_obs
            info = {}
        elif self.is_first_boot:
            reset_res = self.env.reset()
            obs, info = self.boot_manager.execute_boot_sequence(self.env, streamer=streamer, tracker=tracker)
            if hasattr(self.env, "em"):
                try:
                    self.cached_race_start_state = self.env.em.get_state()
                except Exception:
                    pass
            self.is_first_boot = False
        else:
            reset_res = self.env.reset()
            if isinstance(reset_res, tuple) and len(reset_res) == 2:
                obs, info = reset_res
            else:
                obs = reset_res
                info = {}

        self.total_frames_rendered = 0
        self.lap_start_step = 0
        self.prev_progress = 0.0

        proc_frame = self.preprocess_frame(obs)
        ram_vec, telemetry = self.extract_ram_state(info)

        self.frame_buffer = [proc_frame] * self.seq_len
        self.ram_buffer = [ram_vec] * self.seq_len

        return (
            np.array(self.frame_buffer)[np.newaxis, ...],
            np.array(self.ram_buffer)[np.newaxis, ...],
            obs if obs is not None else self.last_valid_obs,
            telemetry
        )

    def step(self, action_idx: int):
        button_array = ACTION_BUTTONS[action_idx]
        accumulated_reward = 0.0
        done = False
        last_obs = None
        last_info = {}

        for _ in range(self.frame_skip):
            self.total_frames_rendered += 1
            step_res = self.env.step(button_array)
            if len(step_res) == 5:
                obs, r, d, tr, info = step_res
                d_flag = d or tr
            else:
                obs, r, d, info = step_res
                d_flag = d

            accumulated_reward += float(r)
            last_obs = obs
            last_info = info if isinstance(info, dict) else {}
            if d_flag:
                done = True
                break

        proc_frame = self.preprocess_frame(last_obs)
        ram_vec, telemetry = self.extract_ram_state(last_info)

        self.frame_buffer.pop(0)
        self.frame_buffer.append(proc_frame)
        self.ram_buffer.pop(0)
        self.ram_buffer.append(ram_vec)

        # REWARD SHAPING
        progress_delta = telemetry["progress"] - self.prev_progress
        self.prev_progress = telemetry["progress"]

        reward = (progress_delta * 15.0) + (telemetry["speed"] * 0.02)

        if telemetry["off_track"]:
            reward -= 1.5

        if last_info.get("lap_completed", False):
            lap_ms = telemetry["lap_time_ms"]
            if lap_ms < self.target_wr_ms:
                time_saved_ms = self.target_wr_ms - lap_ms
                reward += 1000.0 + (time_saved_ms * 0.2)
            else:
                reward += 200.0

        frames_seq = np.array(self.frame_buffer)[np.newaxis, ...]
        rams_seq = np.array(self.ram_buffer)[np.newaxis, ...]

        return frames_seq, rams_seq, reward, done, last_obs if last_obs is not None else self.last_valid_obs, telemetry
