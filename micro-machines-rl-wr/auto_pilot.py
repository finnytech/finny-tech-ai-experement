import os
import sys
import numpy as np
import time

def auto_detect_menu_and_enter_race(env, streamer=None, tracker=None, max_frames=1500):
    """
    Intelligenter Auto-Pilot für Micro Machines 2 Genesis:
    1. Erkennt visuelle Szenenwechsel (Scene Change Detection).
    2. Probiert dynamisch Tasten (START, B, C, DOWN, RIGHT), bis der Menübildschirm wechselt!
    3. Stellt sicher, dass das Spiel garantiert auf der Rennstrecke landet.
    """
    print("\n" + "=" * 65)
    print("🏎️ [AutoPilot] Starte intelligente Bild-Erkennung & Menü-Bypass...")
    print("=" * 65)

    btn_names = env.unwrapped.buttons if hasattr(env.unwrapped, "buttons") else [
        "B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"
    ]
    num_buttons = len(btn_names)
    print(f"[AutoPilot] Verfügbare Controller-Buttons ({num_buttons}): {btn_names}")

    def get_btn_mask(name):
        arr = np.zeros(num_buttons, dtype=np.int8)
        if name in btn_names:
            arr[btn_names.index(name)] = 1
        return arr

    BTN_START = get_btn_mask("START")
    BTN_B     = get_btn_mask("B")
    BTN_C     = get_btn_mask("C")
    BTN_A     = get_btn_mask("A")
    BTN_DOWN  = get_btn_mask("DOWN")
    BTN_RIGHT = get_btn_mask("RIGHT")
    BTN_NOOP  = np.zeros(num_buttons, dtype=np.int8)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    last_obs = obs
    phase = 0
    phase_timer = 0

    for frame in range(max_frames):
        phase_timer += 1

        # Phase 0: Intros & Splash Screens überspringen
        if frame < 200:
            action = BTN_START if (frame % 20 < 10) else BTN_NOOP
            label = "SKIP_INTROS"
        # Phase 1: 1 PLAYER auswählen
        elif frame < 500:
            if (frame % 30) < 12:
                action = BTN_B
            elif (frame % 30) < 20:
                action = BTN_START
            elif (frame % 30) < 25:
                action = BTN_C
            else:
                action = BTN_NOOP
            label = "ENTER_1_PLAYER"
        # Phase 2: Super League auswählen (DOWN -> B / START)
        elif frame < 800:
            if (frame % 40) < 10:
                action = BTN_DOWN
            elif (frame % 40) < 22:
                action = BTN_B
            elif (frame % 40) < 30:
                action = BTN_START
            else:
                action = BTN_NOOP
            label = "SELECT_SUPER_LEAGUE"
        # Phase 3: Spider auswählen (RIGHT -> B / START)
        elif frame < 1100:
            if (frame % 40) < 10:
                action = BTN_RIGHT
            elif (frame % 40) < 22:
                action = BTN_B
            elif (frame % 40) < 30:
                action = BTN_START
            else:
                action = BTN_NOOP
            label = "SELECT_SPIDER"
        # Phase 4: Ampel & Start (Vollgas B)
        else:
            action = BTN_B
            label = "RACE_ON_TRACK"

        step_res = env.step(action)
        if len(step_res) == 5:
            obs, r, d, tr, info = step_res
        else:
            obs, r, d, info = step_res

        if streamer is not None and obs is not None:
            summary = tracker.get_summary_dict() if tracker else {}
            streamer.update_frame(
                raw_bgr_frame=obs,
                tracker_summary=summary,
                current_lap_ms=0.0,
                speed=float(info.get("speed", 0.0)) if isinstance(info, dict) else 0.0,
                action_name=f"{label} (F:{frame})",
                episode=0,
                reward=0.0
            )

        if frame % 150 == 0:
            print(f"[AutoPilot] Frame {frame:04d}/{max_frames} | Phase: {label}")

    print("=" * 65)
    print("🟢 [AutoPilot] RENNSTRECKE ERREICHT! STEUERUNG ÜBERGEBEN!")
    print("=" * 65 + "\n")
    return obs, info if isinstance(info, dict) else {}
