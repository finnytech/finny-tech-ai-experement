import os
import sys
import numpy as np
import time

def auto_detect_menu_and_enter_race(env, streamer=None, tracker=None, max_frames=1200):
    """
    Universeller 4-Spieler Multi-Port Navigator für Micro Machines 2 Genesis:
    Sendet jeden Tastendruck parallel an Spieler 1, Spieler 2, Spieler 3 und Spieler 4 (J-Cart Ports).
    """
    print("\n" + "=" * 65)
    print("🏎️ [AutoPilot] Starte 4-Spieler Multi-Port J-Cart Menü-Bypass...")
    print("=" * 65)

    num_buttons = env.action_space.shape[0]
    print(f"[AutoPilot] Emulator Action-Space Dimension: {num_buttons} Buttons (Ports: {num_buttons // 12})")

    # Genesis Controller Masken (12 Buttons pro Port):
    # 0: B, 1: A, 2: MODE, 3: START, 4: UP, 5: DOWN, 6: LEFT, 7: RIGHT, 8: C
    def make_universal_action(base_12_indices):
        action = np.zeros(num_buttons, dtype=np.int8)
        # Über alle aktiven Spieler-Ports (1 bis 4) spiegeln
        for port_offset in range(0, max(1, num_buttons), 12):
            for idx in base_12_indices:
                if port_offset + idx < num_buttons:
                    action[port_offset + idx] = 1
        return action

    BTN_START = make_universal_action([3])
    BTN_B = make_universal_action([0])
    BTN_C = make_universal_action([8])
    BTN_DOWN = make_universal_action([5])
    BTN_RIGHT = make_universal_action([7])
    BTN_NOOP = np.zeros(num_buttons, dtype=np.int8)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for frame in range(max_frames):
        # Dynamische Multi-Phase
        if frame < 150:
            # Phase 0: Logos überspringen mit START
            action = BTN_START if (frame % 20 < 10) else BTN_NOOP
            label = "SKIP_INTROS_START"
        elif frame < 400:
            # Phase 1: 1 PLAYER betreten mit Taste B und Taste C (Pulse)
            if frame % 25 < 12:
                action = BTN_B
            elif frame % 25 < 18:
                action = BTN_C
            else:
                action = BTN_NOOP
            label = "CONFIRM_1_PLAYER"
        elif frame < 650:
            # Phase 2: Super League auswählen (DOWN -> B)
            if frame % 40 < 12:
                action = BTN_DOWN
            elif frame % 40 < 25:
                action = BTN_B
            else:
                action = BTN_NOOP
            label = "SELECT_SUPER_LEAGUE"
        elif frame < 900:
            # Phase 3: Spider auswählen (RIGHT -> B)
            if frame % 40 < 12:
                action = BTN_RIGHT
            elif frame % 40 < 25:
                action = BTN_B
            else:
                action = BTN_NOOP
            label = "SELECT_SPIDER"
        else:
            # Phase 4: Ampel & Start (Vollgas B)
            action = BTN_B
            label = "START_LIGHTS_COUNTDOWN"

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
