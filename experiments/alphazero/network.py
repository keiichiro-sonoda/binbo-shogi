# Portions (AZNet, BlockV1, BlockV2) are derived from the Pgx AlphaZero example
# (https://github.com/sotetsuk/pgx, examples/alphazero/network.py),
# Copyright 2023 The Pgx Authors, licensed under the Apache License, Version 2.0
# (see LICENSES/Apache-2.0.txt).
# Modified for binbo-shogi (2026): TransformerNet, EncoderLayer and _PGX_ORDER were added.
#
# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class BlockV1(hk.Module):
    def __init__(self, num_channels, name="BlockV1"):
        super(BlockV1, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        return jax.nn.relu(x + i)


class BlockV2(hk.Module):
    def __init__(self, num_channels, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class AZNet(hk.Module):
    """AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        name="az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.resnet_cls = BlockV2 if resnet_v2 else BlockV1

    def __call__(self, x, is_training, test_local_stats):
        x = x.astype(jnp.float32)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        for i in range(self.num_blocks):
            x = self.resnet_cls(self.num_channels, name=f"block_{i}")(
                x, is_training, test_local_stats
            )

        if self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        # policy head
        logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        logits = hk.BatchNorm(True, True, 0.9)(logits, is_training, test_local_stats)
        logits = jax.nn.relu(logits)
        logits = hk.Flatten()(logits)
        logits = hk.Linear(self.num_actions)(logits)

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v


# --- Task 087 T1: encoder-only Transformer trunk (ChessFormer-style square tokens) ---
#
# Design (arXiv 2605.19091 imported to shogi): the 81 board squares are tokens.
# We reuse the SAME (9,9,119) observation as AZNet (pure trunk swap = one-variable
# arch-only comparison); each square's 119-dim feature vector is the token embedding
# input. Policy head is per-token: token = DESTINATION square `to`, emitting 27
# direction logits, placed at Pgx action id = direction*81 + to (see m2_moves).
#
# The obs spatial layout is jnp.rot90(board, k=3) of the Pgx square index (x*9+y),
# so we reorder the 81 flattened tokens back into Pgx-square order first — makes the
# ChessFormer "square token predicts its own moves" prior geometrically exact. The
# permutation is correct-by-construction via argsort of the same rot90 (no manual
# index math): _PGX_ORDER[k] = position in obs-flat holding Pgx square k.
_PGX_ORDER = np.argsort(np.rot90(np.arange(81).reshape(9, 9), k=3).reshape(81))


class EncoderLayer(hk.Module):
    """Pre-LN transformer encoder block: LN->MHA->res, LN->FFN(gelu)->res."""

    def __init__(self, d_model, n_heads, ff_mult=4, name="encoder"):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_mult = ff_mult

    def __call__(self, x):
        # attention sublayer (pre-LN, full self-attention over 81 tokens, no mask)
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        mha = hk.MultiHeadAttention(
            num_heads=self.n_heads,
            key_size=self.d_model // self.n_heads,
            model_size=self.d_model,
            w_init=hk.initializers.VarianceScaling(1.0),
        )
        x = x + mha(h, h, h)
        # feed-forward sublayer
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        h = hk.Linear(self.ff_mult * self.d_model)(h)
        h = jax.nn.gelu(h)
        h = hk.Linear(self.d_model)(h)
        return x + h


class TransformerNet(hk.Module):
    """Encoder-only Transformer with AZNet-compatible (logits, v) output.

    Signature matches AZNet so make_forward wraps it identically; is_training /
    test_local_stats are ignored (LayerNorm is stateless -> empty transform state).
    """

    def __init__(self, num_actions, d_model=256, num_blocks=4, n_heads=8,
                 ff_mult=4, name="transformer_net"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.n_heads = n_heads
        self.ff_mult = ff_mult

    def __call__(self, x, is_training, test_local_stats):
        del is_training, test_local_stats
        b = x.shape[0]
        # (B,9,9,119) -> 81 tokens in Pgx-square order (token index == square `to`)
        x = x.astype(jnp.float32).reshape(b, 81, 119)[:, _PGX_ORDER, :]
        x = hk.Linear(self.d_model)(x)                       # token embedding
        pos = hk.get_parameter("pos_emb", [81, self.d_model], init=hk.initializers.TruncatedNormal(0.02))
        x = x + pos
        for i in range(self.num_blocks):
            x = EncoderLayer(self.d_model, self.n_heads, self.ff_mult, name=f"enc_{i}")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        # policy head: per-token 27 direction logits -> action id = dir*81 + to
        pol = hk.Linear(27)(x)                               # (B,81,27), token=to
        logits = jnp.transpose(pol, (0, 2, 1)).reshape(b, self.num_actions)

        # value head: mean-pool tokens -> MLP -> tanh scalar
        v = jnp.mean(x, axis=1)
        v = jax.nn.gelu(hk.Linear(self.d_model)(v))
        v = jnp.tanh(hk.Linear(1)(v))
        v = v.reshape((-1,))
        return logits, v
