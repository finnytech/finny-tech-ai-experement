import time
import numpy as np
from typing import List, Tuple, Dict, Any

# Genesis Controller Buttons: [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]
BTNS = {
    "NOOP":        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "START":       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "B":           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "A":           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "C":           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "CONFIRM":     [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # B + START gleichzeitig (garantiert Menü-Bestätigung!)
    "UP":          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "DOWN":        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "LEFT":        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "RIGHT":       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}

class AutoMenuBootManager:
    """
    Präziser Menü-Navigator für Micro Machines 2 Sega Mega Drive:
    1. Überspringt Intro & Titelbildschirm
    2. Bestätigt '1 PLAYER'
    3. Wählt 'SUPER LEAGUE' (Division 1)
    4. Wählt den Fahrer 'Spider'
    5. Wartet die Rennstrecke und Startampel ab -> Übergabe an den Agenten!
    """

    def __init__(self, mode: str = "SUPER_LEAGUE_HARD"):
        self.mode = mode

    @staticmethod
    def get_macro_sequence(mode: str = "SUPER_LEAGUE_HARD") -> List[Tuple[List[int], int]]:
        seq = []
        # 1. Intro & Codemasters Splash Screen skippen (B + START drücken)
        seq.append((BTNS["NOOP"], 60))
        seq.append((BTNS["CONFIRM"], 15))
        seq.append((BTNS["NOOP"], 45))
        seq.append((BTNS["CONFIRM"], 15))
        seq.append((BTNS["NOOP"], 60))

        # 2. Im Hauptmenü '1 PLAYER' bestätigen (Cursor steht auf 1 PLAYER)
        seq.append((BTNS["CONFIRM"], 20))
        seq.append((BTNS["NOOP"], 60))
        seq.append((BTNS["B"], 20)) # Nochmal B zur Sicherheit
        seq.append((BTNS["NOOP"], 60))

        if mode == "SUPER_LEAGUE_HARD":
            # 3. 1-Player Untermenü: Runter auf 'SUPER LEAGUE'
            seq.append((BTNS["DOWN"], 15))
            seq.append((BTNS["NOOP"], 25))
            seq.append((BTNS["CONFIRM"], 20)) # SUPER LEAGUE bestätigen
            seq.append((BTNS["NOOP"], 60))

            # 4. Division 1 (Härtester Modus) bestätigen
            seq.append((BTNS["CONFIRM"], 20))
            seq.append((BTNS["NOOP"], 60))

            # 5. Fahrer 'Spider' auswählen (rechts)
            seq.append((BTNS["RIGHT"], 15))
            seq.append((BTNS["NOOP"], 25))
            seq.append((BTNS["CONFIRM"], 20)) # Spider bestätigen
            seq.append((BTNS["NOOP"], 120))   # Streckenlade-Bildschirm abwarten

        elif mode == "TIME_TRIAL_RECORD":
            # 3. 1-Player Untermenü: Runter auf 'TIME TRIAL'
            seq.append((BTNS["DOWN"], 15))
            seq.append((BTNS["NOOP"], 20))
            seq.append((BTNS["DOWN"], 15))
            seq.append((BTNS["NOOP"], 20))
            seq.append((BTNS["CONFIRM"], 20))
            seq.append((BTNS["NOOP"], 60))
            # Track bestätigen
            seq.append((BTNS["CONFIRM"], 20))
            seq.append((BTNS["NOOP"], 120))

        # 6. Countdown auf der Strecke ('3, 2, 1, GO!') abwarten
        seq.append((BTNS["NOOP"], 180))
        return seq

    def execute_boot_sequence(self, env, streamer=None, tracker=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        macro = self.get_macro_sequence(self.mode)
        last_obs = None
        last_info = {}

        print(f"[BootManager] 🏎️ Navigiere durch Startmenü (1 PLAYER -> {self.mode} -> Spider)...")
        for buttons, frame_count in macro:
            for _ in range(frame_count):
                step_res = env.step(buttons)
                if len(step_res) == 5:
                    obs, r, d, tr, info = step_res
                    done = d or tr
                else:
                    obs, r, d, info = step_res
                    done = d
                
                last_obs = obs
                last_info = info if isinstance(info, dict) else {}

                # Live-Stream während des Menü-Bypasses aktualisieren, damit du alles live siehst!
                if streamer is not None and last_obs is not None:
                    summary = tracker.get_summary_dict() if tracker else {}
                    streamer.update_frame(
                        raw_bgr_frame=last_obs,
                        tracker_summary=summary,
                        current_lap_ms=0.0,
                        speed=0.0,
                        action_name="MENU_BOOT",
                        episode=0,
                        reward=0.0
                    )

                if done:
                    reset_res = env.reset()
                    if isinstance(reset_res, tuple) and len(reset_res) == 2:
                        last_obs, last_info = reset_res
                    else:
                        last_obs = reset_res
                        last_info = {}

        print("[BootManager] 🟢 START-AMPEL GRÜN! Steuerung auf der Rennstrecke an JAX/Flax übergeben!")
        return last_obs, last_info
