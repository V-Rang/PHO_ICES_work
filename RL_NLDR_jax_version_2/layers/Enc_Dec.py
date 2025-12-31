
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any

class EncDecLayer(nn.Module):
    in_length: int
    d_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=self.d_model, name='net1')(x)
        # x = nn.relu(x)
        x = nn.Dense(features=self.d_model, name='net2')(x)
        # x = nn.relu(x)
        return x

class Encoder_Decoder(nn.Module):
    in_length: int
    d_model: int
    e_layers: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.e_layers):
            x = EncDecLayer(self.in_length, self.d_model, name=f'layer_{i}')(x)
        x = nn.Dense(features=6, name='final_layer')(x)
        return x