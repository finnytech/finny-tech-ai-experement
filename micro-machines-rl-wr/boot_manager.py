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
    Präziser Menü-Navigator für Micro Machines 2 Sega Mega Drive:
    1. Überspringt Intro & Titelbildschirm
    2. Wählt im Hauptmenü '1 PLAYER'
    3. Wählt 'SUPER LEAGUE' (Division 1) oder 'TIME TRIAL'
    4. Wählt 'Spider' als Fahrer
    5. Wartet die Ampel ('3, 2, 1, GO!') ab und übergibt die Kontrolle an das JAX-Transformer-Modell!
    """

    def __init__(self, mode: str = "SUPER_LEAGUE_HARD"):
        self.mode = mode

    @staticmethod
    def get_macro_sequence(mode: str = "SUPER_LEAGUE_HARD") -> List[Tuple[List[int], int]]:
        seq = []
        # 1. Intro & Codemasters Splash Screen skippen
        seq.append((BTNS["NOOP"], 45))
        seq.append((BTNS["START"], 10))
        seq.append((BTNS["NOOP"], 35))
        seq.append((BTNS["START"], 10))
        seq.append((BTNS["NOOP"], 45))

        # 2. Hauptmenü: '1 PLAYER' auswählen (Cursor steht bereits auf 1 PLAYER)
        seq.append((BTNS["START"], 10)) # Bestätige 1 PLAYER
        seq.append((BTNS["NOOP"], 45))

        if mode == "SUPER_LEAGUE_HARD":
            # 3. 1-Player Untermenü: Runter auf 'SUPER LEAGUE' navigieren
            seq.append((BTNS["DOWN"], 10))
            seq.append((BTNS["NOOP"], 15))
            seq.append((BTNS["START"], 10)) # Bestätige SUPER LEAGUE
            seq.append((BTNS["NOOP"], 50))

            # 4. Division 1 (Härtester Modus) auswählen
            seq.append((BTNS["START"], 10))
            seq.append((BTNS["NOOP"], 45))

            # 5. Charakterauswahl: 'Spider' (Fahrer mit Top-Speed & Drift-Präzision)
            seq.append((BTNS["RIGHT"], 10))
            seq.append((BTNS["NOOP"], 15))
            seq.append((BTNS["START"], 10)) # Bestätige Fahrer Spider
            seq.append((BTNS["NOOP"], 90)) # Strecken-Ladebildschirm abwarten

        elif mode == "TIME_TRIAL_RECORD":
            # 3. 1-Player Untermenü: Runter auf 'TIME TRIAL'
            seq.append((BTNS["DOWN"], 10))
            seq.append((BTNS["NOOP"], 12))
            seq.append((BTNS["DOWN"], 10))
            seq.append((BTNS["NOOP"], 12))
            seq.append((BTNS["START"], 10)) # Bestätige TIME TRIAL
            seq.append((BTNS["NOOP"], 50))
            # Track 1 (Breakfast Bends) auswählen
            seq.append((BTNS["START"], 10))
            seq.append((BTNS["NOOP"], 90))

        # 6. Ampel-Countdown auf der Rennstrecke ('3, 2, 1, GO!') abwarten
        seq.append((BTNS["NOOP"], 140))
        return seq

    def execute_boot_sequence(self, env) -> Tuple[np.ndarray, Dict[str, Any]]:
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

                if done:
                    reset_res = env.reset()
                    if isinstance(reset_res, tuple) and len(reset_res) == 2:
                        last_obs, last_info = reset_res
                    else:
                        last_obs = reset_res
                        last_info = {}

        print("[BootManager] 🟢 START-AMPEL GRÜN! Steuerung auf der Rennstrecke an JAX/Flax übergeben!")
        return last_obs, last_info
