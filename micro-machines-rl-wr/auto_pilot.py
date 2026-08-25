import os
import sys
import numpy as np
import time

def auto_detect_menu_and_enter_race(env, streamer=None, tracker=None, max_frames=1200):
    """
    Universeller Menü-Bypass für Micro Machines 2 Genesis:
    Drückt systematisch die echten Codemasters Menü-Tasten (Taste A, C, START, B)
    und navigiert präzise von 1 PLAYER -> SUPER LEAGUE -> Spider -> Rennstrecke!
    """
    print("\n" + "=" * 65)
    print("🏎️ [AutoPilot] Starte Codemasters Menü-Bypass (A/C/START/B Tasten-Folge)...")
    print("=" * 65)

    btn_names = env.unwrapped.buttons if hasattr(env.unwrapped, "buttons") else [
        "B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"
    ]
    num_buttons = len(btn_names)

    def get_btn(name):
        arr = np.zeros(num_buttons, dtype=np.int8)
        if name in btn_names:
            arr[btn_names.index(name)] = 1
        return arr

    BTN_START = get_btn("START")
    BTN_A     = get_btn("A")
    BTN_B     = get_btn("B")
    BTN_C     = get_btn("C")
    BTN_DOWN  = get_btn("DOWN")
    BTN_RIGHT = get_btn("RIGHT")
    BTN_NOOP  = np.zeros(num_buttons, dtype=np.int8)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for frame in range(max_frames):
        # -------------------------------------------------------------
        # Phase 0 (Frames 0 - 200): Sega & Codemasters Splash Screens
        # -------------------------------------------------------------
        if frame < 200:
            action = BTN_START if (frame % 20 < 10) else BTN_NOOP
            label = "SKIP_INTROS"

        # -------------------------------------------------------------
        # Phase 1 (Frames 200 - 450): '1 PLAYER' mit Taste A, C, START, B bestätigen
        # -------------------------------------------------------------
        elif frame < 450:
            cycle = frame % 40
            if cycle < 10:
                action = BTN_A
            elif cycle < 20:
                action = BTN_C
            elif cycle < 30:
                action = BTN_START
            else:
                action = BTN_B
            label = "CONFIRM_1_PLAYER"

        # -------------------------------------------------------------
        # Phase 2 (Frames 450 - 700): 1x DOWN auf 'SUPER LEAGUE' & mit A/C/B bestätigen
        # -------------------------------------------------------------
        elif frame < 700:
            if frame < 480:
                action = BTN_DOWN if (frame % 20 < 10) else BTN_NOOP
            else:
                cycle = frame % 40
                if cycle < 10:
                    action = BTN_A
                elif cycle < 20:
                    action = BTN_C
                elif cycle < 30:
                    action = BTN_B
                else:
                    action = BTN_START
            label = "SELECT_SUPER_LEAGUE"

        # -------------------------------------------------------------
        # Phase 3 (Frames 700 - 950): Division 1 & Fahrer 'Spider' (1x RIGHT)
        # -------------------------------------------------------------
        elif frame < 950:
            if frame < 730:
                action = BTN_A if (frame % 20 < 10) else BTN_NOOP
            elif frame < 770:
                action = BTN_RIGHT if (frame % 20 < 10) else BTN_NOOP
            else:
                cycle = frame % 40
                if cycle < 10:
                    action = BTN_A
                elif cycle < 20:
                    action = BTN_C
                elif cycle < 30:
                    action = BTN_B
                else:
                    action = BTN_NOOP
            label = "SELECT_SPIDER"

        # -------------------------------------------------------------
        # Phase 4 (Frames 950 - 1200): Startampel & Rennstrecke (Gas geben)
        # -------------------------------------------------------------
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
    print("🟢 [AutoPilot] RENNSTRECKE ERREICHT! STEUERUNG AN TPU TRANSFORMER ÜBERGEBEN!")
    print("=" * 65 + "\n")
    return obs, info if isinstance(info, dict) else {}
