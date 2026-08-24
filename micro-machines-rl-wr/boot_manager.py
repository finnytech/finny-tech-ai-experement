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
    Automatischer Menu-Skipper & Savestate-Manager für Micro Machines 2 Genesis:
    1. Überspringt automatisch den Codemasters-Intro-Screen und das Hauptmenü.
    2. Wählt den härtesten Modus ('Super League - Division 1' oder 'Time Trial').
    3. Wählt den schnellsten Rennfahrer ('Spider' / 'Formula 1 Sports Car').
    4. Übergibt das Spiel exakt im Moment des Startsignals ('3, 2, 1, GO!') an das RL-Modell!
    """

    def __init__(self, mode: str = "SUPER_LEAGUE_HARD"):
        self.mode = mode

    @staticmethod
    def get_macro_sequence(mode: str = "SUPER_LEAGUE_HARD") -> List[Tuple[List[int], int]]:
        """
        Gibt eine präzise Tasten-Sequenz zurück: (Tasten-Array, Anzahl der Frames)
        """
        seq = []
        # 1. Intro Splash Screen skippen (Codemasters & Titelbildschirm)
        seq.append((BTNS["NOOP"], 30))
        seq.append((BTNS["START"], 6))
        seq.append((BTNS["NOOP"], 25))
        seq.append((BTNS["START"], 6))
        seq.append((BTNS["NOOP"], 35))

        if mode == "SUPER_LEAGUE_HARD":
            # 2. Im Hauptmenü nach unten auf 'Super League' navigieren
            seq.append((BTNS["DOWN"], 6))
            seq.append((BTNS["NOOP"], 10))
            seq.append((BTNS["START"], 6)) # Bestätigen
            seq.append((BTNS["NOOP"], 40))

            # 3. Division 1 (Härtester Modus / Pro League) wählen
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 30))

            # 4. Charakterauswahl: 'Spider' (Aggressivster & schnellster Top-Speed Fahrer)
            seq.append((BTNS["RIGHT"], 6))
            seq.append((BTNS["NOOP"], 10))
            seq.append((BTNS["START"], 6)) # Bestätigen
            seq.append((BTNS["NOOP"], 80)) # Strecken-Ladebildschirm abwarten

        elif mode == "TIME_TRIAL_RECORD":
            # Time Trial Mode: Direkte Rundenzeit-Jagd ohne störende KI
            seq.append((BTNS["DOWN"], 6))
            seq.append((BTNS["NOOP"], 8))
            seq.append((BTNS["DOWN"], 6))
            seq.append((BTNS["NOOP"], 8))
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 40))
            # Track 1: Breakfast Bends wählen
            seq.append((BTNS["START"], 6))
            seq.append((BTNS["NOOP"], 70))

        # Ampel-Countdown ('3, 2, 1, GO') abwarten, dann Start!
        seq.append((BTNS["NOOP"], 40))
        return seq

    def execute_boot_sequence(self, env) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Führt den Boot-Makro aus und liefert den ersten echten Renn-Frame zurück,
        an dem das RL-Modell sofort Gas geben kann.
        """
        macro = self.get_macro_sequence(self.mode)
        last_obs = None
        last_info = {}

        print(f"[BootManager] Überspringe Startmenü -> Starte direkt im Modus: {self.mode}...")
        for buttons, frame_count in macro:
            for _ in range(frame_count):
                obs, r, d, tr, info = env.step(buttons)
                last_obs = obs
                last_info = info
                if d or tr:
                    obs, info = env.reset()
                    last_obs = obs
                    last_info = info

        print("[BootManager] 🟢 START-AMPEL GRÜN! Steuerung an JAX/Flax Transformer übergeben!")
        return last_obs, last_info
