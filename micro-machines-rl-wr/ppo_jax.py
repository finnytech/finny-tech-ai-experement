import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple
from flax.training.train_state import TrainState
from model import MicroMachinesTransformerRL

class PPOAgent:
    def __init__(
        self,
        num_actions: int = 9,
        seq_len: int = 16,
        ram_state_dim: int = 16,
        learning_rate: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.num_actions = num_actions
        self.seq_len = seq_len
        self.ram_state_dim = ram_state_dim
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.model = MicroMachinesTransformerRL(
            num_actions=num_actions,
            embed_dim=256,
            num_layers=4,
            num_heads=4,
            ram_state_dim=ram_state_dim
        )
        self.tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
        )

    def init_state(self, rng: jax.Array) -> TrainState:
        dummy_frames = jnp.zeros((1, self.seq_len, 84, 84, 1), dtype=jnp.float32)
        dummy_ram = jnp.zeros((1, self.seq_len, self.ram_state_dim), dtype=jnp.float32)
        params = self.model.init(rng, dummy_frames, dummy_ram, deterministic=True)
        return TrainState.create(apply_fn=self.model.apply, params=params, tx=self.tx)

    @staticmethod
    @jax.jit
    def select_action(
        state: TrainState,
        frames: jnp.ndarray,
        ram_states: jnp.ndarray,
        rng: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # frames: [B, T, 84, 84, 1], ram_states: [B, T, D]
        logits, value = state.apply_fn(state.params, frames, ram_states, deterministic=True)
        action = jax.random.categorical(rng, logits, axis=-1)
        log_prob = jax.nn.log_softmax(logits, axis=-1)
        action_log_prob = jnp.take_along_axis(log_prob, jnp.expand_dims(action, -1), axis=-1).squeeze(-1)
        return action, action_log_prob, value

    @staticmethod
    def compute_gae(
        rewards: jnp.ndarray,    # [T_steps, B]
        values: jnp.ndarray,     # [T_steps, B]
        dones: jnp.ndarray,      # [T_steps, B]
        last_value: jnp.ndarray, # [B]
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        t_steps = rewards.shape[0]
        advantages = []
        last_gae = 0.0

        for t in reversed(range(t_steps)):
            if t == t_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages.insert(0, last_gae)

        advantages = jnp.array(advantages)
        returns = advantages + values
        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        return advantages, returns

    @staticmethod
    def ppo_loss_fn(
        params,
        apply_fn,
        frames,
        ram_states,
        actions,
        old_log_probs,
        returns,
        advantages,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01
    ):
        logits, values = apply_fn(params, frames, ram_states, deterministic=False)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        cur_log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)

        # Policy Ratio
        ratio = jnp.exp(cur_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Value Loss
        value_loss = 0.5 * jnp.mean(jnp.square(values - returns))

        # Entropy Loss
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy
        }

    @staticmethod
    @jax.jit
    def train_step(
        state: TrainState,
        frames: jnp.ndarray,
        ram_states: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        returns: jnp.ndarray,
        advantages: jnp.ndarray,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01
    ) -> Tuple[TrainState, Dict[str, Any]]:
        grad_fn = jax.value_and_grad(PPOAgent.ppo_loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(
            state.params,
            state.apply_fn,
            frames,
            ram_states,
            actions,
            old_log_probs,
            returns,
            advantages,
            clip_eps,
            vf_coef,
            ent_coef
        )
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics
