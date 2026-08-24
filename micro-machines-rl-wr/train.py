import time
import jax
import jax.numpy as jnp
import numpy as np
import os

from model import MicroMachinesTransformerRL
from ppo_jax import PPOAgent
from wr_tracker import WorldRecordTracker
from streamer import StreamBroadcaster
from env_wrapper import MicroMachinesEnvWrapper, ACTION_NAMES

def make_mock_or_retro_env():
    """Tries to create stable-retro/gym-retro env, otherwise creates an emulation mock for testing."""
    try:
        import retro
        print("[Env] Found retro/stable-retro! Loading Micro Machines 2 Genesis...")
        return retro.make(game="MicroMachines2-Genesis", state="BreakfastBends.TimeAttack")
    except Exception as e:
        print(f"[Env] Note: running in mock simulation mode ({e}). For real MD execution install stable-retro.")
        
        class MockRetroEnv:
            def __init__(self):
                self.step_idx = 0
                self.progress = 0.0
                self.speed = 40.0
            def reset(self):
                self.step_idx = 0
                self.progress = 0.0
                self.speed = 40.0
                obs = np.random.randint(0, 255, (224, 320, 3), dtype=np.uint8)
                info = {"speed": self.speed, "lap": 1, "checkpoint": 0, "max_checkpoints": 16, "off_track": 0, "points": 0, "won": False}
                return obs, info
            def step(self, buttons):
                self.step_idx += 1
                # If accel (button 0 or 1)
                if buttons[0] == 1 or buttons[1] == 1:
                    self.speed = min(120.0, self.speed + 1.2)
                else:
                    self.speed = max(0.0, self.speed - 0.8)
                self.progress = min(1.0, self.progress + (self.speed / 5000.0))
                done = (self.progress >= 1.0) or (self.step_idx > 800)
                lap_completed = (self.progress >= 1.0)
                lap_time_ms = self.step_idx * 16.666
                obs = np.zeros((224, 320, 3), dtype=np.uint8)
                # Draw mock car track
                obs[100:150, 50:270, 1] = 180
                obs[120:130, int(60 + self.progress * 180):int(70 + self.progress * 180), 0] = 255
                info = {
                    "speed": self.speed,
                    "lap": 1,
                    "checkpoint": int(self.progress * 16),
                    "max_checkpoints": 16,
                    "off_track": 0 if 100 < self.speed < 115 else 1,
                    "points": int(self.progress * 50),
                    "won": lap_completed and (lap_time_ms < 48250.0),
                    "lap_completed": lap_completed,
                    "lap_time_ms": lap_time_ms
                }
                return obs, 1.0, done, False, info
        return MockRetroEnv()

def main():
    print("=" * 60)
    print("🚀 FINNY TECH DEEP LABS - MICRO MACHINES 2 WR AGENT (JAX/FLAX)")
    print(f"Devices detected: {jax.devices()}")
    print("=" * 60)

    # 1. Setup World Record Tracker & Target
    TARGET_WR_MS = 48250.0 # 48.25s World Record Target
    tracker = WorldRecordTracker(
        track_name="Breakfast Bends",
        target_wr_ms=TARGET_WR_MS,
        log_file="wr_tracker_log.json",
        best_run_file="best_world_record.json"
    )

    # 2. Setup Live Streamer & Server
    streamer = StreamBroadcaster(port=8080)
    streamer.start_http_server()
    streamer.start_cloudflare_tunnel()

    # 3. Setup Environment
    base_env = make_mock_or_retro_env()
    env = MicroMachinesEnvWrapper(base_env, seq_len=16, target_wr_ms=TARGET_WR_MS)

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
    rollout_steps = 128

    print("\n🏁 Starting World Record Speedrun Training Loop...\n")

    for ep in range(1, total_episodes + 1):
        frames_seq, rams_seq, raw_obs, telemetry = env.reset()
        ep_reward = 0.0
        done = False

        # Trajectory Buffers for PPO Update
        buf_frames = []
        buf_rams = []
        buf_actions = []
        buf_log_probs = []
        buf_rewards = []
        buf_values = []
        buf_dones = []

        while not done:
            rng, act_rng = jax.random.split(rng)
            
            # Agent selects action
            action_arr, log_prob_arr, value_arr = agent.select_action(
                train_state,
                jnp.array(frames_seq),
                jnp.array(rams_seq),
                act_rng
            )
            action_idx = int(action_arr[0])
            log_prob = float(log_prob_arr[0])
            value = float(value_arr[0])

            # Step Environment
            next_frames, next_rams, reward, done, raw_obs, telemetry = env.step(action_idx)
            ep_reward += reward

            # Record to Buffers
            buf_frames.append(frames_seq[0])
            buf_rams.append(rams_seq[0])
            buf_actions.append(action_idx)
            buf_log_probs.append(log_prob)
            buf_rewards.append(reward)
            buf_values.append(value)
            buf_dones.append(1.0 if done else 0.0)

            # Update Live Streamer HUD
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

        # Episode Ended -> Register with World Record Tracker
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

        # Print Live Terminal Status & WR Alerts
        if metrics.is_world_record:
            print(f"\n🔥🔥🔥 [NEW WORLD RECORD BROKEN!] Lap: {metrics.lap_time_str} (Target: {tracker.ms_to_time_str(TARGET_WR_MS)}) Delta: {metrics.delta_to_wr_ms:.2f}ms | Points: {points} | Ep: {ep} 🔥🔥🔥\n")
        else:
            print(f"[Ep {ep:04d}] Lap: {metrics.lap_time_str} | Best: {tracker.ms_to_time_str(tracker.current_best_lap_ms)} | Pts: {points} | Wins: {tracker.total_wins} | Rew: {ep_reward:.1f}")

        # Train PPO on TPU with collected Trajectories
        if len(buf_frames) > 16:
            _, _, last_val_arr = agent.select_action(train_state, jnp.array(frames_seq), jnp.array(rams_seq), rng)
            advantages, returns = agent.compute_gae(
                rewards=jnp.array(buf_rewards),
                values=jnp.array(buf_values),
                dones=jnp.array(buf_dones),
                last_value=last_val_arr[0],
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
