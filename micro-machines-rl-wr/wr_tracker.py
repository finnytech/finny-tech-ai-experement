import os
import json
import time
import dataclasses
from typing import Optional, Dict, Any, List

@dataclasses.dataclass
class RunMetrics:
    run_id: int
    episode: int
    timestamp: float
    lap_time_ms: float
    lap_time_str: str
    target_wr_ms: float
    delta_to_wr_ms: float
    is_world_record: bool
    points: int
    wins: int
    total_reward: float
    avg_speed: float
    checkpoint_progress_pct: float
    reason: str  # 'completed_lap', 'collision_timeout', 'finished_race'

class WorldRecordTracker:
    def __init__(
        self,
        track_name: str = "Breakfast Bends",
        target_wr_ms: float = 48250.0,  # Benchmark e.g. 48.25s
        log_file: str = "wr_tracker_log.json",
        best_run_file: str = "best_world_record.json"
    ):
        self.track_name = track_name
        self.target_wr_ms = target_wr_ms
        self.log_file = log_file
        self.best_run_file = best_run_file
        
        self.current_best_lap_ms = float("inf")
        self.total_runs = 0
        self.total_wins = 0
        self.total_points = 0
        self.wr_broken_count = 0
        self.history: List[Dict[str, Any]] = []
        
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.best_run_file):
            try:
                with open(self.best_run_file, "r") as f:
                    data = json.load(f)
                    self.current_best_lap_ms = data.get("best_lap_time_ms", float("inf"))
                    self.wr_broken_count = data.get("wr_broken_count", 0)
            except Exception:
                pass

    @staticmethod
    def ms_to_time_str(ms: float) -> str:
        if ms == float("inf") or ms <= 0:
            return "--:--:--"
        total_sec = ms / 1000.0
        minutes = int(total_sec // 60)
        seconds = int(total_sec % 60)
        hundredths = int((total_sec - int(total_sec)) * 100)
        return f"{minutes:02d}:{seconds:02d}.{hundredths:02d}"

    def register_run(
        self,
        episode: int,
        lap_time_ms: float,
        points: int = 0,
        won: bool = False,
        total_reward: float = 0.0,
        avg_speed: float = 0.0,
        checkpoint_progress_pct: float = 0.0,
        reason: str = "completed_lap"
    ) -> RunMetrics:
        self.total_runs += 1
        if won:
            self.total_wins += 1
        self.total_points += points

        delta = lap_time_ms - self.target_wr_ms
        is_wr = (lap_time_ms > 0) and (lap_time_ms < self.target_wr_ms)
        is_personal_best = (lap_time_ms > 0) and (lap_time_ms < self.current_best_lap_ms)

        if is_personal_best:
            self.current_best_lap_ms = lap_time_ms
            if is_wr:
                self.wr_broken_count += 1
            self._save_best(episode, lap_time_ms, points, is_wr)

        metric = RunMetrics(
            run_id=self.total_runs,
            episode=episode,
            timestamp=time.time(),
            lap_time_ms=lap_time_ms,
            lap_time_str=self.ms_to_time_str(lap_time_ms),
            target_wr_ms=self.target_wr_ms,
            delta_to_wr_ms=delta,
            is_world_record=is_wr,
            points=points,
            wins=self.total_wins,
            total_reward=total_reward,
            avg_speed=avg_speed,
            checkpoint_progress_pct=checkpoint_progress_pct,
            reason=reason
        )

        self.history.append(dataclasses.asdict(metric))
        self._append_to_log(metric)
        return metric

    def _save_best(self, episode: int, lap_time_ms: float, points: int, is_wr: bool):
        best_data = {
            "track_name": self.track_name,
            "best_lap_time_ms": lap_time_ms,
            "best_lap_time_str": self.ms_to_time_str(lap_time_ms),
            "target_wr_ms": self.target_wr_ms,
            "target_wr_str": self.ms_to_time_str(self.target_wr_ms),
            "delta_to_wr_ms": lap_time_ms - self.target_wr_ms,
            "is_world_record": is_wr,
            "wr_broken_count": self.wr_broken_count,
            "episode_achieved": episode,
            "total_wins": self.total_wins,
            "total_points": self.total_points,
            "timestamp": time.time()
        }
        with open(self.best_run_file, "w") as f:
            json.dump(best_data, f, indent=2)

    def _append_to_log(self, metric: RunMetrics):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(dataclasses.asdict(metric)) + "\n")

    def get_summary_dict(self) -> Dict[str, Any]:
        return {
            "track_name": self.track_name,
            "total_runs": self.total_runs,
            "total_wins": self.total_wins,
            "total_points": self.total_points,
            "win_rate_pct": (self.total_wins / max(1, self.total_runs)) * 100.0,
            "target_wr_str": self.ms_to_time_str(self.target_wr_ms),
            "current_best_str": self.ms_to_time_str(self.current_best_lap_ms),
            "current_best_ms": self.current_best_lap_ms,
            "wr_beaten": self.current_best_lap_ms < self.target_wr_ms,
            "wr_broken_count": self.wr_broken_count,
        }
