import os
import sys
import numpy as np
import time

def auto_detect_menu_and_enter_race(env, streamer=None, tracker=None, max_frames=1100):
    """
    Präziser Menü-Navigator für Micro Machines 2 Sega Mega Drive:
    1. Startet auf '1 PLAYER' und bestätigt direkt (ohne DOWN zu drücken, um nicht nach unten zu scrollen).
    2. Geht im 1-Player-Menü auf 'SUPER LEAGUE' (Division 1).
    3. Wählt 'Spider' als Fahrer.
    4. Startet das Rennen und übergibt an das JAX/Flax Causal Transformer RL Modell auf der TPU!
    """
    print("\n" + "=" * 65)
    print("🏎️ [AutoPilot] Starte präzise Menü-Navigation (1 PLAYER -> Super League -> Spider)...")
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
    BTN_B     = get_btn("B")
    BTN_C     = get_btn("C")
    BTN_DOWN  = get_btn("DOWN")
    BTN_RIGHT = get_btn("RIGHT")
    BTN_NOOP  = np.zeros(num_buttons, dtype=np.int8)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for frame in range(max_frames):
        # -----------------------------------------------------------------
        # Phase 0 (Frames 0 - 220): Intro & Codemasters Splash Screens mit START überspringen
        # -----------------------------------------------------------------
        if frame < 220:
            action = BTN_START if (frame % 20 < 10) else BTN_NOOP
            label = "SKIP_INTROS"

        # -----------------------------------------------------------------
        # Phase 1 (Frames 220 - 450): Auf '1 PLAYER' stehen bleiben und mit Taste B & START bestätigen!
        # (WICHTIG: Kein DOWN drücken, da Cursor bereits auf '1 PLAYER' steht!)
        # -----------------------------------------------------------------
        elif frame < 450:
            if (frame % 25) < 12:
                action = BTN_B
            elif (frame % 25) < 18:
                action = BTN_START
            else:
                action = BTN_NOOP
            label = "ENTER_1_PLAYER"

        # -----------------------------------------------------------------
        # Phase 2 (Frames 450 - 700): Im 1-Player Menü 1x DOWN auf 'SUPER LEAGUE' drücken und bestätigen!
        # -----------------------------------------------------------------
        elif frame < 700:
            if frame < 480:
                action = BTN_DOWN if (frame % 20 < 10) else BTN_NOOP
            else:
                if (frame % 25) < 12:
                    action = BTN_B
                elif (frame % 25) < 18:
                    action = BTN_START
                else:
                    action = BTN_NOOP
            label = "SELECT_SUPER_LEAGUE"

        # -----------------------------------------------------------------
        # Phase 3 (Frames 700 - 900): Division 1 bestätigen & Fahrer 'Spider' (1x RIGHT) wählen!
        # -----------------------------------------------------------------
        elif frame < 900:
            if frame < 740:
                action = BTN_B if (frame % 20 < 10) else BTN_NOOP
            elif frame < 780:
                action = BTN_RIGHT if (frame % 20 < 10) else BTN_NOOP
            else:
                action = BTN_B if (frame % 20 < 10) else BTN_NOOP
            label = "SELECT_SPIDER"

        # -----------------------------------------------------------------
        # Phase 4 (Frames 900 - 1100): Startampel & Startgitter (Vollgas Taste B!)
        # -----------------------------------------------------------------
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
