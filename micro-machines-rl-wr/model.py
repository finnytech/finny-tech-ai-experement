import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any

class OptimizedCausalSelfAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x: [B, T, D]
        b, t, d = x.shape
        total_dim = self.num_heads * self.head_dim

        # Fused QKV Projection
        qkv = nn.Dense(3 * total_dim, use_bias=False, dtype=self.dtype, name="qkv_proj")(x)
        qkv = qkv.reshape((b, t, 3, self.num_heads, self.head_dim))
        
        # Split into Q, K, V -> [B, num_heads, T, head_dim]
        q = qkv[:, :, 0, :, :].swapaxes(1, 2)
        k = qkv[:, :, 1, :, :].swapaxes(1, 2)
        v = qkv[:, :, 2, :, :].swapaxes(1, 2)

        # Scaled Dot-Product Attention
        scale = 1.0 / jnp.sqrt(self.head_dim).astype(self.dtype)
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale # [B, heads, T, T]

        # Causal Lower-Triangular Mask
        causal_mask = jnp.tril(jnp.ones((t, t), dtype=bool))
        causal_mask = jnp.expand_dims(causal_mask, (0, 1)) # [1, 1, T, T]
        scores = jnp.where(causal_mask, scores, -1e9)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        if not deterministic and self.dropout_rate > 0.0 and self.has_rng("dropout"):
            attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=deterministic)

        out = jnp.matmul(attn_weights, v) # [B, heads, T, head_dim]
        out = out.swapaxes(1, 2).reshape((b, t, total_dim))
        out = nn.Dense(d, dtype=self.dtype, name="out_proj")(out)
        return out

class OptimizedTransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Pre-LN Self-Attention
        norm1 = nn.LayerNorm(dtype=self.dtype)(x)
        attn_out = OptimizedCausalSelfAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype
        )(norm1, deterministic=deterministic)
        x = x + attn_out

        # Pre-LN GELU MLP
        norm2 = nn.LayerNorm(dtype=self.dtype)(x)
        d = x.shape[-1]
        mlp_out = nn.Dense(d * self.mlp_ratio, dtype=self.dtype)(norm2)
        mlp_out = jax.nn.gelu(mlp_out)
        mlp_out = nn.Dense(d, dtype=self.dtype)(mlp_out)
        x = x + mlp_out
        return x

class FastPatchCNN(nn.Module):
    embed_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, frame: jnp.ndarray) -> jnp.ndarray:
        # frame: [B, T, 84, 84, 1]
        b, t, h, w, c = frame.shape
        flat = frame.reshape((b * t, h, w, c)).astype(self.dtype)

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), dtype=self.dtype)(flat)
        x = jax.nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), dtype=self.dtype)(x)
        x = jax.nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), dtype=self.dtype)(x)
        x = jax.nn.relu(x)

        x = x.reshape((b * t, -1))
        x = nn.Dense(self.embed_dim, dtype=self.dtype)(x)
        return x.reshape((b, t, self.embed_dim))

class MicroMachinesTransformerRL(nn.Module):
    num_actions: int = 9
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ram_state_dim: int = 16
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        frames: jnp.ndarray,
        ram_states: jnp.ndarray,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b, t, _, _, _ = frames.shape

        # 1. Visual Feature Extraction
        frame_tokens = FastPatchCNN(embed_dim=self.embed_dim, dtype=self.dtype)(frames)

        # 2. RAM Telemetry Projection
        ram_tokens = nn.Dense(self.embed_dim, dtype=self.dtype)(ram_states.astype(self.dtype))
        ram_tokens = jax.nn.gelu(ram_tokens)

        # 3. Multimodal Fusion + Learned Positional Encoding
        pos_emb = self.param("pos_emb", nn.initializers.normal(stddev=0.02), (1, t, self.embed_dim))
        x = frame_tokens + ram_tokens + pos_emb.astype(self.dtype)

        # 4. Transformer Attention Backbone
        head_dim = self.embed_dim // self.num_heads
        for i in range(self.num_layers):
            x = OptimizedTransformerBlock(
                num_heads=self.num_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                name=f"t_block_{i}"
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(dtype=self.dtype, name="final_ln")(x)
        latest_token = x[:, -1, :].astype(jnp.float32)

        # 5. Actor Head
        action_logits = nn.Dense(self.num_actions, name="actor_head")(latest_token)

        # 6. Critic Head
        value = nn.Dense(1, name="critic_head")(latest_token)
        value = jnp.squeeze(value, axis=-1)

        return action_logits, value
