import os
import sys
import gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import time
import jax
import jax.numpy as jnp
import numpy as np

# Setup the real Sega Mega Drive ROM
import setup_rom
setup_rom.ensure_real_rom()

from model import MicroMachinesTransformerRL
from ppo_jax import PPOAgent
from wr_tracker import WorldRecordTracker
from streamer import StreamBroadcaster
from env_wrapper import MicroMachinesEnvWrapper, ACTION_NAMES

_GLOBAL_RETRO_ENV = None

def close_existing_retro_env():
    """Closes any previously active Retro emulator instance to prevent singleton collisions."""
    global _GLOBAL_RETRO_ENV
    if _GLOBAL_RETRO_ENV is not None:
        try:
            _GLOBAL_RETRO_ENV.close()
        except Exception:
            pass
        _GLOBAL_RETRO_ENV = None
    gc.collect()

def make_real_sega_env():
    """Lädt das echte 16-Bit Sega Mega Drive Spiel Micro Machines 2 in stable-retro."""
    global _GLOBAL_RETRO_ENV
    close_existing_retro_env()

    import retro
    import retro.data

    # Integration Pfad registrieren
    custom_dir = setup_rom.get_retro_custom_dir()
    parent_integrations = os.path.dirname(custom_dir)
    try:
        retro.data.Integrations.add_custom_path(parent_integrations)
    except Exception:
        pass

    print("[Env] 🎮 Initialisiere echten Genesis Plus GX Emulator für Micro Machines 2...")
    
    # Clean load with ALL integrations
    try:
        env = retro.make(
            game="MicroMachines2-Genesis",
            inttype=retro.data.Integrations.ALL
        )
        _GLOBAL_RETRO_ENV = env
        print("[Env] ✅ ECHTES SEGA MEGA DRIVE SPIEL ERFOLGREICH GELADEN! (Echte 16-Bit Grafik & Physik)")
        return env
    except Exception as e:
        print(f"[Env] Versuche direkten Game-Load ({e})...")
        close_existing_retro_env()
        env = retro.make(game="MicroMachines2-Genesis")
        _GLOBAL_RETRO_ENV = env
        print("[Env] ✅ ECHTES SEGA MEGA DRIVE SPIEL GELADEN!")
        return env

def main():
    print("=" * 60)
    print("🚀 FINNY TECH DEEP LABS - MICRO MACHINES 2 WR AGENT (JAX/FLAX)")
    print(f"JAX Devices detected: {jax.devices()}")
    print("=" * 60)

    # 1. Setup World Record Tracker & Target (42.50s Benchmark)
    TARGET_WR_MS = 42500.0
    tracker = WorldRecordTracker(
        track_name="Breakfast Bends",
        target_wr_ms=TARGET_WR_MS,
        log_file=os.path.join(SCRIPT_DIR, "wr_tracker_log.json"),
        best_run_file=os.path.join(SCRIPT_DIR, "best_world_record.json")
    )

    # 2. Setup Live Streamer & Server
    streamer = StreamBroadcaster(port=8080)
    streamer.start_http_server()
    streamer.start_cloudflare_tunnel()

    # 3. Setup ECHTES SEGA MEGA DRIVE GAME
    base_env = make_real_sega_env()
    env = MicroMachinesEnvWrapper(base_env, seq_len=16, frame_skip=4, target_wr_ms=TARGET_WR_MS, game_mode="SUPER_LEAGUE_HARD")

    # 4. Initialize JAX PPO Agent
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    agent = PPOAgent(
        num_actions=len(ACTION_NAMES),
        seq_len=16,
        ram_state_dim=16,
        learning_rate=3e-4
    )
    train_state = agent.init_state(init_rng)
    print("[Agent] JAX/Flax Transformer Policy Initialized on TPU/GPU/CPU!")

    # 5. Training Loop
    total_episodes = 10000
    print("\n🏁 Starting REAL Sega Mega Drive World Record Speedrun Training...\n")

    for ep in range(1, total_episodes + 1):
        frames_seq, rams_seq, raw_obs, telemetry = env.reset()
        ep_reward = 0.0
        done = False

        buf_frames = []
        buf_rams = []
        buf_actions = []
        buf_log_probs = []
        buf_rewards = []
        buf_values = []
        buf_dones = []

        while not done:
            rng, act_rng = jax.random.split(rng)
            
            action_arr, log_prob_arr, value_arr = agent.select_action(
                train_state,
                jnp.array(frames_seq),
                jnp.array(rams_seq),
                act_rng
            )
            action_idx = int(action_arr[0])
            log_prob = float(log_prob_arr[0])
            value = float(value_arr[0])

            next_frames, next_rams, reward, done, raw_obs, telemetry = env.step(action_idx)
            ep_reward += reward

            buf_frames.append(frames_seq[0])
            buf_rams.append(rams_seq[0])
            buf_actions.append(action_idx)
            buf_log_probs.append(log_prob)
            buf_rewards.append(reward)
            buf_values.append(value)
            buf_dones.append(1.0 if done else 0.0)

            # Echte 16-Bit Spielgrafik ins Live-Stream Overlay rendern
            streamer.update_frame(
                raw_bgr_frame=raw_obs,
                tracker_summary=tracker.get_summary_dict(),
                current_lap_ms=telemetry["lap_time_ms"],
                speed=telemetry["speed"],
                action_name=ACTION_NAMES[action_idx],
                episode=ep,
                reward=ep_reward
            )

            frames_seq = next_frames
            rams_seq = next_rams

        lap_time = telemetry.get("lap_time_ms", 0.0)
        points = telemetry.get("points", 0)
        won = telemetry.get("won", False)
        
        metrics = tracker.register_run(
            episode=ep,
            lap_time_ms=lap_time,
            points=points,
            won=won,
            total_reward=ep_reward,
            avg_speed=telemetry.get("speed", 0.0),
            checkpoint_progress_pct=telemetry.get("progress", 0.0) * 100.0
        )

        if metrics.is_world_record:
            print(f"\n🔥🔥🔥 [NEW WORLD RECORD BROKEN!] Lap: {metrics.lap_time_str} (Target: {tracker.ms_to_time_str(TARGET_WR_MS)}) Delta: {metrics.delta_to_wr_ms:.2f}ms | Points: {points} | Ep: {ep} 🔥🔥🔥\n")
        else:
            print(f"[Ep {ep:04d}] Lap: {metrics.lap_time_str} | Best: {tracker.ms_to_time_str(tracker.current_best_lap_ms)} | Pts: {points} | Wins: {tracker.total_wins} | Rew: {ep_reward:.1f}")

        if len(buf_frames) > 16:
            _, _, last_val_arr = agent.select_action(train_state, jnp.array(frames_seq), jnp.array(rams_seq), rng)
            advantages, returns = agent.compute_gae(
                rewards=jnp.array(buf_rewards),
                values=jnp.array(buf_values),
                dones=jnp.array(buf_dones),
                last_value=float(last_val_arr[0]),
                gamma=agent.gamma,
                lam=agent.gae_lambda
            )

            train_state, train_metrics = agent.train_step(
                state=train_state,
                frames=jnp.array(buf_frames),
                ram_states=jnp.array(buf_rams),
                actions=jnp.array(buf_actions),
                old_log_probs=jnp.array(buf_log_probs),
                returns=returns,
                advantages=advantages
            )

if __name__ == "__main__":
    main()
