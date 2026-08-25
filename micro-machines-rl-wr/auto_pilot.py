import os
import sys
import numpy as np
import time

def auto_detect_menu_and_enter_race(env, streamer=None, tracker=None, max_frames=1500):
    """
    Hardware-präziser Menü-Bypass für Micro Machines 2 Genesis:
    Sendet saubere 'Press & Release' Zyklen mit echtem NOOP-Zustand (Null-Spannung),
    damit der Genesis 68000 Prozessor die Flanken-Interrupts (Tastendrücke) sauber erkennt!
    """
    print("\n" + "=" * 65)
    print("🏎️ [AutoPilot] Starte Hardware-präzisen 'Press & Release' Menü-Bypass...")
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
    print("[AutoPilot] ⏭️ Überspringe Sega & Codemasters Splash Screens...")
    send_step(BTN_NOOP, 60, "WAIT_INIT")
    for _ in range(8):
        press_and_release(BTN_START, press_frames=10, release_frames=15, label="SKIP_INTRO_START")

    # -----------------------------------------------------------------
    # Schritt 2: Auf '1 PLAYER' stehen bleiben und mit B / START / C / A bestätigen
    # -----------------------------------------------------------------
    print("[AutoPilot] 🎮 Bestätige '1 PLAYER'...")
    send_step(BTN_NOOP, 30, "WAIT_MENU")
    for _ in range(3):
        press_and_release(BTN_B, press_frames=15, release_frames=15, label="CONFIRM_1_PLAYER_B")
        press_and_release(BTN_START, press_frames=15, release_frames=15, label="CONFIRM_1_PLAYER_START")
        press_and_release(BTN_C, press_frames=15, release_frames=15, label="CONFIRM_1_PLAYER_C")
        press_and_release(BTN_A, press_frames=15, release_frames=15, label="CONFIRM_1_PLAYER_A")

    # -----------------------------------------------------------------
    # Schritt 3: Im 1-Player-Menü 1x DOWN auf 'SUPER LEAGUE' navigieren & bestätigen
    # -----------------------------------------------------------------
    print("[AutoPilot] 🏆 Wähle 'SUPER LEAGUE' (Division 1)...")
    send_step(BTN_NOOP, 30, "WAIT_SUBMENU")
    press_and_release(BTN_DOWN, press_frames=12, release_frames=18, label="DOWN_TO_SUPER_LEAGUE")
    
    for _ in range(3):
        press_and_release(BTN_B, press_frames=15, release_frames=15, label="CONFIRM_SUPER_LEAGUE_B")
        press_and_release(BTN_C, press_frames=15, release_frames=15, label="CONFIRM_SUPER_LEAGUE_C")
        press_and_release(BTN_START, press_frames=15, release_frames=15, label="CONFIRM_SUPER_LEAGUE_START")

    # -----------------------------------------------------------------
    # Schritt 4: Division 1 bestätigen & Fahrer 'Spider' auswählen
    # -----------------------------------------------------------------
    print("[AutoPilot] 🕷️ Wähle Fahrer 'Spider'...")
    send_step(BTN_NOOP, 30, "WAIT_DIV")
    press_and_release(BTN_B, press_frames=15, release_frames=15, label="CONFIRM_DIV1_B")
    
    send_step(BTN_NOOP, 30, "WAIT_CHAR")
    press_and_release(BTN_RIGHT, press_frames=12, release_frames=18, label="SELECT_SPIDER_RIGHT")
    
    for _ in range(3):
        press_and_release(BTN_B, press_frames=15, release_frames=15, label="CONFIRM_SPIDER_B")
        press_and_release(BTN_C, press_frames=15, release_frames=15, label="CONFIRM_SPIDER_C")
        press_and_release(BTN_START, press_frames=15, release_frames=15, label="CONFIRM_SPIDER_START")

    # -----------------------------------------------------------------
    # Schritt 5: Startampel & Rennstrecke
    # -----------------------------------------------------------------
    print("[AutoPilot] 🚦 Warte auf Rennstrecke & Startampel...")
    last_obs, last_info = send_step(BTN_B, 220, "START_RACE_ACCEL")

    print("=" * 65)
    print("🟢 [AutoPilot] RENNSTRECKE ERREICHT! STEUERUNG AN TPU TRANSFORMER ÜBERGEBEN!")
    print("=" * 65 + "\n")
    return last_obs, last_info
