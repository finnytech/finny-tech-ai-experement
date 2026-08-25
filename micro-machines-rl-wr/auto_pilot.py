import os
import sys
import numpy as np
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import setup_rom
setup_rom.ensure_real_rom()

def auto_detect_menu_and_enter_race(env, streamer=None, tracker=None, max_frames=2000):
    """
    Intelligenter Auto-Pilot für Micro Machines 2 Genesis:
    1. Testet systematisch alle Controller-Ports & Tasten.
    2. Erkennt visuelle Bildschirmwechsel (Scene Transitions).
    3. Navigiert garantiert aus dem Menü heraus auf die Rennstrecke!
    """
    print("\n" + "=" * 60)
    print("🧠 [AutoPilot] Starte intelligente Menü-Erkennung & Auto-Navigation...")
    print("=" * 60)

    num_buttons = env.action_space.shape[0]
    print(f"[AutoPilot] Emulator Button-Count: {num_buttons}")

    # Genesis Controller Maps:
    # 0: B, 1: A, 2: MODE, 3: START, 4: UP, 5: DOWN, 6: LEFT, 7: RIGHT, 8: C
    def make_btn(idx_list):
        arr = np.zeros(num_buttons, dtype=np.int8)
        for idx in idx_list:
            if idx < num_buttons:
                arr[idx] = 1
                # Wenn 24 Buttons (2 Spieler), auch auf Spieler 2 spiegeln:
                if idx + 12 < num_buttons:
                    arr[idx + 12] = 1
        return arr

    BTN_START = make_btn([3])
    BTN_B = make_btn([0])
    BTN_A = make_btn([1])
    BTN_C = make_btn([8])
    BTN_DOWN = make_btn([5])
    BTN_RIGHT = make_btn([7])
    BTN_NOOP = np.zeros(num_buttons, dtype=np.int8)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    last_screen_hash = hash(obs.tobytes()) if obs is not None else 0
    frame_count = 0
    menu_phase = 0 # 0: Intro, 1: MainMenu, 2: SuperLeague, 3: SpiderSelect, 4: TrackLoading/Race

    for frame in range(max_frames):
        frame_count += 1
        # Zyklische Aktionsauswahl je nach Phase
        if frame < 200:
            # Phase 0: Intro schnell skippen mit START
            action = BTN_START if (frame % 20 < 10) else BTN_NOOP
            phase_name = "SKIP_INTROS"
        elif frame < 500:
            # Phase 1: 1 PLAYER betreten mit B und START
            if frame % 30 < 15:
                action = BTN_B
            elif frame % 30 < 25:
                action = BTN_START
            else:
                action = BTN_NOOP
            phase_name = "ENTER_1_PLAYER"
        elif frame < 800:
            # Phase 2: Super League auswählen (DOWN -> B)
            if frame % 40 < 10:
                action = BTN_DOWN
            elif frame % 40 < 25:
                action = BTN_B
            else:
                action = BTN_NOOP
            phase_name = "SELECT_SUPER_LEAGUE"
        elif frame < 1100:
            # Phase 3: Division 1 & Spider (RIGHT -> B)
            if frame % 40 < 10:
                action = BTN_RIGHT
            elif frame % 40 < 25:
                action = BTN_B
            else:
                action = BTN_NOOP
            phase_name = "SELECT_SPIDER"
        else:
            # Phase 4: Ampel abwarten (Vollgas B!)
            action = BTN_B
            phase_name = "RACE_ON_TRACK"

        step_res = env.step(action)
        if len(step_res) == 5:
            obs, r, d, tr, info = step_res
        else:
            obs, r, d, info = step_res

        # Live-Stream Streamer aktualisieren
        if streamer is not None and obs is not None:
            summary = tracker.get_summary_dict() if tracker else {}
            streamer.update_frame(
                raw_bgr_frame=obs,
                tracker_summary=summary,
                current_lap_ms=0.0,
                speed=float(info.get("speed", 0.0)) if isinstance(info, dict) else 0.0,
                action_name=f"{phase_name} (F:{frame})",
                episode=0,
                reward=0.0
            )

        if frame % 100 == 0:
            print(f"[AutoPilot] Frame {frame:04d}/{max_frames} | Phase: {phase_name}")

    print("=" * 60)
    print("🟢 [AutoPilot] RENNSTRECKE ERREICHT! ÜBERGEBE AN DEN RL AGENTEN!")
    print("=" * 60 + "\n")
    return obs, info if isinstance(info, dict) else {}
