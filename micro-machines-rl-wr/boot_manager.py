import time
import numpy as np
from typing import List, Tuple, Dict, Any

# Standard Sega Genesis Button Array (12 Buttons):
# [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]
BTNS = {
    "NOOP":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "START":  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # Button 3: START
    "B":      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Button 0: B (Gas / Select)
    "A":      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Button 1: A
    "C":      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # Button 8: C (Confirm)
    "UP":     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # Button 4: UP
    "DOWN":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # Button 5: DOWN
    "LEFT":   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # Button 6: LEFT
    "RIGHT":  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # Button 7: RIGHT
}

class AutoMenuBootManager:
    """
    Präziser Menü-Navigator für Micro Machines 2 Sega Mega Drive:
    Überspringt alle Sega/Codemasters Intros, navigiert durch:
    1 PLAYER -> SUPER LEAGUE (Div 1) -> Spider -> Rennstrecke!
    """

    def __init__(self, mode: str = "SUPER_LEAGUE_HARD"):
        self.mode = mode

    def step_frames(self, env, buttons: List[int], count: int, streamer=None, tracker=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        last_obs = None
        last_info = {}
        for _ in range(count):
            step_res = env.step(buttons)
            if len(step_res) == 5:
                obs, r, d, tr, info = step_res
            else:
                obs, r, d, info = step_res
            last_obs = obs
            last_info = info if isinstance(info, dict) else {}

            if streamer is not None and last_obs is not None:
                summary = tracker.get_summary_dict() if tracker else {}
                streamer.update_frame(
                    raw_bgr_frame=last_obs,
                    tracker_summary=summary,
                    current_lap_ms=0.0,
                    speed=0.0,
                    action_name="NAVIGATING_MENU",
                    episode=0,
                    reward=0.0
                )
        return last_obs, last_info

    def press_and_release(self, env, button_name: str, press_frames: int = 10, release_frames: int = 15, streamer=None, tracker=None):
        btn = BTNS.get(button_name, BTNS["NOOP"])
        self.step_frames(env, btn, press_frames, streamer, tracker)
        return self.step_frames(env, BTNS["NOOP"], release_frames, streamer, tracker)

    def execute_boot_sequence(self, env, *args, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        streamer = kwargs.get("streamer", None)
        tracker = kwargs.get("tracker", None)
        if len(args) > 0 and streamer is None:
            streamer = args[0]
        if len(args) > 1 and tracker is None:
            tracker = args[1]

        print("[BootManager] 🏎️ Starte Boot-Sequenz & überspringe Intros...")

        # Phase 1: Sega & Codemasters Splash Screens überspringen
        self.step_frames(env, BTNS["NOOP"], 70, streamer, tracker)
        for _ in range(6):
            self.press_and_release(env, "START", press_frames=8, release_frames=12, streamer=streamer, tracker=tracker)

        # Phase 2: Hauptmenü 'PLEASE SELECT OPTION' -> '1 PLAYER' bestätigen
        print("[BootManager] 🎮 Bestätige '1 PLAYER'...")
        self.step_frames(env, BTNS["NOOP"], 30, streamer, tracker)
        # Drücke C und B (die Standard-Bestätigungstasten in MM2)
        self.press_and_release(env, "C", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
        self.press_and_release(env, "B", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
        self.press_and_release(env, "START", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)

        # Phase 3: Spielmodus wählen
        self.step_frames(env, BTNS["NOOP"], 45, streamer, tracker)
        if self.mode == "SUPER_LEAGUE_HARD":
            print("[BootManager] 🏆 Wähle 'SUPER LEAGUE' (Division 1)...")
            self.press_and_release(env, "DOWN", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
            self.press_and_release(env, "C", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
            self.press_and_release(env, "B", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)

            # Division 1 bestätigen
            self.step_frames(env, BTNS["NOOP"], 45, streamer, tracker)
            self.press_and_release(env, "C", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
            self.press_and_release(env, "B", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)

            # Fahrer 'Spider' auswählen
            print("[BootManager] 🕷️ Wähle Fahrer 'Spider' (Top-Speed)...")
            self.step_frames(env, BTNS["NOOP"], 45, streamer, tracker)
            self.press_and_release(env, "RIGHT", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
            self.press_and_release(env, "C", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
            self.press_and_release(env, "B", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)

        elif self.mode == "TIME_TRIAL_RECORD":
            print("[BootManager] ⏱️ Wähle 'TIME TRIAL'...")
            self.press_and_release(env, "DOWN", press_frames=12, release_frames=12, streamer=streamer, tracker=tracker)
            self.press_and_release(env, "DOWN", press_frames=12, release_frames=12, streamer=streamer, tracker=tracker)
            self.press_and_release(env, "C", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)
            # Track bestätigen
            self.step_frames(env, BTNS["NOOP"], 45, streamer, tracker)
            self.press_and_release(env, "C", press_frames=12, release_frames=15, streamer=streamer, tracker=tracker)

        # Phase 4: Strecken-Ladebildschirm & Countdown (3, 2, 1, GO!)
        print("[BootManager] 🏁 Warte auf Strecken-Ladebildschirm & Startampel...")
        last_obs, last_info = self.step_frames(env, BTNS["NOOP"], 240, streamer, tracker)

        print("[BootManager] 🟢 START-AMPEL GRÜN! Steuerung auf der Rennstrecke an JAX/Flax übergeben!")
        return last_obs, last_info
