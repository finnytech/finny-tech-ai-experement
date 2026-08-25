import time
import numpy as np
from typing import List, Tuple, Dict, Any

# Sega Genesis Controller Button Mapping (12 Buttons):
# Index 0: B (Accelerate / Menu Confirm)
# Index 1: A (Reverse)
# Index 2: MODE
# Index 3: START (Pause / Menu Start)
# Index 4: UP
# Index 5: DOWN
# Index 6: LEFT
# Index 7: RIGHT
# Index 8: C (Horn / Weapon)
BTNS = {
    "NOOP":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "B":      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # B Button (Select / Accelerate)
    "START":  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # START Button
    "DOWN":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # DOWN
    "RIGHT":  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # RIGHT
    "UP":     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # UP
    "LEFT":   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # LEFT
}

class AutoMenuBootManager:
    """
    Robuster Menü-Navigator für Micro Machines 2 Sega Mega Drive:
    Garantiert den Übergang von Boot/Titelbildschirm -> Rennstrecke.
    """

    def __init__(self, mode: str = "SUPER_LEAGUE_HARD"):
        self.mode = mode

    def step_action(self, env, buttons: List[int], frames: int, streamer=None, tracker=None, label: str = "MENU_NAV"):
        last_obs = None
        last_info = {}
        for _ in range(frames):
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
                    action_name=label,
                    episode=0,
                    reward=0.0
                )
        return last_obs, last_info

    def pulse_button(self, env, btn_name: str, press_f: int = 12, rel_f: int = 18, streamer=None, tracker=None, label: str = "PULSE"):
        btn = BTNS.get(btn_name, BTNS["NOOP"])
        self.step_action(env, btn, press_f, streamer, tracker, label)
        return self.step_action(env, BTNS["NOOP"], rel_f, streamer, tracker, "WAIT")

    def execute_boot_sequence(self, env, *args, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        streamer = kwargs.get("streamer", None)
        tracker = kwargs.get("tracker", None)
        if len(args) > 0 and streamer is None:
            streamer = args[0]
        if len(args) > 1 and tracker is None:
            tracker = args[1]

        print("\n" + "=" * 55)
        print("🏎️ [BootManager] STARTE VOLLAUTOMATISCHE MENÜ-NAVIGATION...")
        print("=" * 55)

        # -------------------------------------------------------------
        # SCHRITT 1: Intro-Splash Screens überspringen & 1 PLAYER betreten
        # (Dauert ca. 7-8 Sekunden = ~450 Frames)
        # -------------------------------------------------------------
        print("[BootManager] ⏭️ Überspringe Sega & Codemasters Intros & bestätige '1 PLAYER'...")
        for i in range(16):
            # Abwechselnd START und B drücken, um alle Intros sicher zu skippen
            self.pulse_button(env, "START", press_f=10, rel_f=15, streamer=streamer, tracker=tracker, label="SKIP_INTRO_START")
            self.pulse_button(env, "B", press_f=10, rel_f=15, streamer=streamer, tracker=tracker, label="SELECT_1_PLAYER_B")

        # -------------------------------------------------------------
        # SCHRITT 2: Modus wählen ('SUPER LEAGUE' Division 1)
        # -------------------------------------------------------------
        print("[BootManager] 🏆 Wähle 'SUPER LEAGUE' (Division 1)...")
        self.step_action(env, BTNS["NOOP"], 30, streamer, tracker, "WAIT_MENU")
        self.pulse_button(env, "DOWN", press_f=12, rel_f=18, streamer=streamer, tracker=tracker, label="DOWN_TO_SUPER_LEAGUE")
        
        for _ in range(6):
            self.pulse_button(env, "B", press_f=12, rel_f=18, streamer=streamer, tracker=tracker, label="CONFIRM_SUPER_LEAGUE")

        # Division 1 bestätigen
        print("[BootManager] 🥇 Bestätige Division 1...")
        self.step_action(env, BTNS["NOOP"], 30, streamer, tracker, "WAIT_DIV")
        for _ in range(5):
            self.pulse_button(env, "B", press_f=12, rel_f=18, streamer=streamer, tracker=tracker, label="CONFIRM_DIVISION_1")

        # -------------------------------------------------------------
        # SCHRITT 3: Fahrer 'Spider' auswählen (Schnellstes Formel-1 Auto)
        # -------------------------------------------------------------
        print("[BootManager] 🕷️ Wähle Fahrer 'Spider'...")
        self.step_action(env, BTNS["NOOP"], 30, streamer, tracker, "WAIT_CHAR")
        self.pulse_button(env, "RIGHT", press_f=12, rel_f=18, streamer=streamer, tracker=tracker, label="SELECT_SPIDER_RIGHT")
        
        for _ in range(6):
            self.pulse_button(env, "B", press_f=12, rel_f=18, streamer=streamer, tracker=tracker, label="CONFIRM_SPIDER_B")

        # -------------------------------------------------------------
        # SCHRITT 4: Strecken-Ladebildschirm & Ampel-Countdown (3, 2, 1, GO!)
        # -------------------------------------------------------------
        print("[BootManager] 🚦 Warte auf Rennstrecke & Startampel (3, 2, 1, GO!)...")
        last_obs, last_info = self.step_action(env, BTNS["NOOP"], 260, streamer, tracker, label="START_COUNTDOWN")

        print("=" * 55)
        print("🟢 [BootManager] START-AMPEL GRÜN! STEUERUNG AN JAX TRANSFORMER ÜBERGEBEN!")
        print("=" * 55 + "\n")
        return last_obs, last_info
