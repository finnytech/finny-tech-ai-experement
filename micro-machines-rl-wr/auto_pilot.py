import os
import sys
import numpy as np
import time

def auto_detect_menu_and_enter_race(env, streamer=None, tracker=None, max_frames=1300):
    """
    Präziser Menü-Bypass für Micro Machines 2 Genesis:
    1. 1 PLAYER betreten
    2. Im 1-Player Menü 2x DOWN drücken auf 'SUPER LEAGUE' (CHALLENGE -> HEAD TO HEAD -> SUPER LEAGUE)
    3. Division 1 (Härtester Modus) bestätigen
    4. Fahrer 'Spider' auswählen
    5. Startampel abwarten -> Übergabe an den JAX Transformer auf der TPU!
    """
    print("\n" + "=" * 65)
    print("🏎️ [AutoPilot] Navigiere von 1-Player-Menü -> SUPER LEAGUE -> Spider...")
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

    def send_step(action_mask, count, label_name):
        nonlocal obs
        for _ in range(count):
            step_res = env.step(action_mask)
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
                    action_name=label_name,
                    episode=0,
                    reward=0.0
                )
        return obs, info if isinstance(info, dict) else {}

    def press_and_release(button_mask, press_frames=12, release_frames=15, label="PRESS"):
        send_step(button_mask, press_frames, label)
        return send_step(BTN_NOOP, release_frames, "RELEASE_NOOP")

    # -----------------------------------------------------------------
    # Schritt 1: Intros & Splash Screens überspringen
    # -----------------------------------------------------------------
    print("[AutoPilot] ⏭️ Überspringe Splash Screens...")
    send_step(BTN_NOOP, 60, "WAIT_INIT")
    for _ in range(8):
        press_and_release(BTN_START, press_frames=10, release_frames=15, label="SKIP_INTRO_START")

    # -----------------------------------------------------------------
    # Schritt 2: '1 PLAYER' betreten
    # -----------------------------------------------------------------
    print("[AutoPilot] 🎮 Betrete '1 PLAYER'...")
    send_step(BTN_NOOP, 30, "WAIT_MENU")
    for _ in range(2):
        press_and_release(BTN_B, press_frames=15, release_frames=15, label="ENTER_1_PLAYER_B")
        press_and_release(BTN_C, press_frames=15, release_frames=15, label="ENTER_1_PLAYER_C")

    # -----------------------------------------------------------------
    # Schritt 3: Im 1-Player Untermenü 2x DOWN drücken auf 'SUPER LEAGUE'
    # (Reihenfolge: CHALLENGE -> HEAD TO HEAD -> SUPER LEAGUE)
    # -----------------------------------------------------------------
    print("[AutoPilot] 🏆 Navigiere zu 'SUPER LEAGUE' (2x DOWN)...")
    send_step(BTN_NOOP, 40, "WAIT_SUBMENU")
    press_and_release(BTN_DOWN, press_frames=12, release_frames=18, label="DOWN_1_HEAD_TO_HEAD")
    press_and_release(BTN_DOWN, press_frames=12, release_frames=18, label="DOWN_2_SUPER_LEAGUE")
    
    # Bestätige SUPER LEAGUE
    for _ in range(2):
        press_and_release(BTN_B, press_frames=15, release_frames=15, label="CONFIRM_SUPER_LEAGUE_B")
        press_and_release(BTN_C, press_frames=15, release_frames=15, label="CONFIRM_SUPER_LEAGUE_C")

    # -----------------------------------------------------------------
    # Schritt 4: Division 1 (Härtester Modus) bestätigen
    # -----------------------------------------------------------------
    print("[AutoPilot] 🥇 Bestätige Division 1...")
    send_step(BTN_NOOP, 40, "WAIT_DIV")
    for _ in range(2):
        press_and_release(BTN_B, press_frames=15, release_frames=15, label="CONFIRM_DIV1_B")
        press_and_release(BTN_C, press_frames=15, release_frames=15, label="CONFIRM_DIV1_C")

    # -----------------------------------------------------------------
    # Schritt 5: Fahrer 'Spider' (Formel 1, Top-Speed) auswählen
    # -----------------------------------------------------------------
    print("[AutoPilot] 🕷️ Wähle Fahrer 'Spider' (1x RIGHT)...")
    send_step(BTN_NOOP, 40, "WAIT_CHAR")
    press_and_release(BTN_RIGHT, press_frames=12, release_frames=18, label="SELECT_SPIDER_RIGHT")
    
    for _ in range(2):
        press_and_release(BTN_B, press_frames=15, release_frames=15, label="CONFIRM_SPIDER_B")
        press_and_release(BTN_C, press_frames=15, release_frames=15, label="CONFIRM_SPIDER_C")

    # -----------------------------------------------------------------
    # Schritt 6: Strecken-Ladebildschirm & Startampel (3, 2, 1, GO!)
    # -----------------------------------------------------------------
    print("[AutoPilot] 🚦 Warte auf Rennstrecke & Startampel...")
    last_obs, last_info = send_step(BTN_B, 240, "START_RACE_ACCEL")

    print("=" * 65)
    print("🟢 [AutoPilot] RENNSTRECKE ERREICHT! STEUERUNG AN TPU TRANSFORMER ÜBERGEBEN!")
    print("=" * 65 + "\n")
    return last_obs, last_info
