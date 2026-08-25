import time
import numpy as np
from typing import List, Tuple, Dict, Any

# Genesis Controller Buttons: [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]
BTNS = {
    "NOOP":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "START": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "B":     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "A":     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "C":     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "UP":    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "DOWN":  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "LEFT":  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "RIGHT": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}

class AutoMenuBootManager:
    """
    Automatischer Menu-Skipper für Micro Machines 2 Genesis:
    1. Überspringt automatisch Intro-Screen & Hauptmenü.
    2. Wählt den Modus ('Super League - Division 1' oder 'Time Trial').
    3. Wählt den Fahrer ('Spider' / Formula 1).
    4. Übergibt das Spiel exakt im Moment des Startsignals an das RL-Modell!
    """

    def __init__(self, mode: str = "SUPER_LEAGUE_HARD"):
        self.mode = mode

    @staticmethod
    def get_macro_sequence(mode: str = "SUPER_LEAGUE_HARD") -> List[Tuple[List[int], int]]:
        seq = []
        # Intro Splash Screen skippen
        seq.append((BTNS["NOOP"], 30))
        seq.append((BTNS["START"], 6))
        seq.append((BTNS["NOOP"], 25))
        seq.append((BTNS["START"], 6))
        seq.append((BTNS["NOOP"], 35))

        if mode == "SUPER_LEAGUE_HARD":
            # Hauptmenü -> Super League
            seq.append((BTNS["DOWN"], 6))
            seq.append((BTNS["NOOP"], 10))
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 40))

            # Division 1 wählen
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 30))

            # Spider wählen
            seq.append((BTNS["RIGHT"], 6))
            seq.append((BTNS["NOOP"], 10))
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 80))

        elif mode == "TIME_TRIAL_RECORD":
            # Time Trial Mode
            seq.append((BTNS["DOWN"], 6))
            seq.append((BTNS["NOOP"], 8))
            seq.append((BTNS["DOWN"], 6))
            seq.append((BTNS["NOOP"], 8))
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 40))
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 70))

        # Countdown abwarten
        seq.append((BTNS["NOOP"], 40))
        return seq

    def execute_boot_sequence(self, env) -> Tuple[np.ndarray, Dict[str, Any]]:
        macro = self.get_macro_sequence(self.mode)
        last_obs = None
        last_info = {}

        print(f"[BootManager] Überspringe Startmenü -> Starte Modus: {self.mode}...")
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

                if done:
                    reset_res = env.reset()
                    if isinstance(reset_res, tuple) and len(reset_res) == 2:
                        last_obs, last_info = reset_res
                    else:
                        last_obs = reset_res
                        last_info = {}

        print("[BootManager] 🟢 START-AMPEL GRÜN! Steuerung übergeben!")
        return last_obs, last_info
