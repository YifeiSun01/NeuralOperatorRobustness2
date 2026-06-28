import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any


def gelu(x):
    # Use exact GELU, matching PyTorch default GELU.
    return jax.nn.gelu(x, approximate=False)


def init_linear_params(key, in_features, out_features, dtype=jnp.float32):
    # Initialize a Linear layer similar to a simple uniform initializer.
    k1, k2 = jax.random.split(key)

    limit = 1.0 / jnp.sqrt(in_features)

    weight = jax.random.uniform(
        k1,
        shape=(out_features, in_features),
        minval=-limit,
        maxval=limit,
        dtype=dtype,
    )

    bias = jax.random.uniform(
        k2,
        shape=(out_features,),
        minval=-limit,
        maxval=limit,
        dtype=dtype,
    )

    return {
        "weight": weight,
        "bias": bias,
    }


def linear_apply(x, params):
    # x shape: [batch, size_x, in_features]
    # weight shape: [out_features, in_features]
    return jnp.einsum("bni,oi->bno", x, params["weight"]) + params["bias"]


def init_conv1d_kernel1_params(key, in_channels, out_channels, dtype=jnp.float32):
    # Initialize a Conv1d layer with kernel_size=1.
    k1, k2 = jax.random.split(key)

    limit = 1.0 / jnp.sqrt(in_channels)

    weight = jax.random.uniform(
        k1,
        shape=(out_channels, in_channels, 1),
        minval=-limit,
        maxval=limit,
        dtype=dtype,
    )

    bias = jax.random.uniform(
        k2,
        shape=(out_channels,),
        minval=-limit,
        maxval=limit,
        dtype=dtype,
    )

    return {
        "weight": weight,
        "bias": bias,
    }


def conv1d_kernel1_apply(x, params):
    # x shape: [batch, in_channels, size_x]
    # weight shape: [out_channels, in_channels, 1]
    return jnp.einsum("bix,oix->box", x, params["weight"]) + params["bias"][None, :, None]


def init_spectral_conv1d_params(
    key,
    in_channels,
    out_channels,
    modes1,
    dtype=jnp.float32,
):
    # Initialize the complex Fourier weights.
    scale = 1.0 / (in_channels * out_channels)

    if dtype == jnp.float64:
        real_dtype = jnp.float64
        complex_dtype = jnp.complex128
    else:
        real_dtype = jnp.float32
        complex_dtype = jnp.complex64

    k1, k2 = jax.random.split(key)

    real_part = jax.random.uniform(
        k1,
        shape=(in_channels, out_channels, modes1),
        dtype=real_dtype,
    )

    imag_part = jax.random.uniform(
        k2,
        shape=(in_channels, out_channels, modes1),
        dtype=real_dtype,
    )

    weights1 = scale * (real_part + 1j * imag_part)
    weights1 = weights1.astype(complex_dtype)

    return {
        "weights1": weights1,
    }


def spectral_conv1d_apply(x, params, out_channels, modes1):
    # x shape: [batch, in_channels, size_x]
    batchsize = x.shape[0]
    size_x = x.shape[-1]

    x_ft = jnp.fft.rfft(x, axis=-1)

    out_ft = jnp.zeros(
        shape=(batchsize, out_channels, size_x // 2 + 1),
        dtype=x_ft.dtype,
    )

    low_modes = jnp.einsum(
        "bix,iox->box",
        x_ft[:, :, :modes1],
        params["weights1"].astype(x_ft.dtype),
    )

    out_ft = out_ft.at[:, :, :modes1].set(low_modes)

    x = jnp.fft.irfft(out_ft, n=size_x, axis=-1)

    return x


def init_mlp_params(
    key,
    in_channels,
    out_channels,
    mid_channels,
    dtype=jnp.float32,
):
    # Equivalent to:
    # Conv1d(in_channels, mid_channels, 1)
    # GELU
    # Conv1d(mid_channels, out_channels, 1)
    k1, k2 = jax.random.split(key)

    return {
        "conv0": init_conv1d_kernel1_params(
            k1,
            in_channels=in_channels,
            out_channels=mid_channels,
            dtype=dtype,
        ),
        "conv2": init_conv1d_kernel1_params(
            k2,
            in_channels=mid_channels,
            out_channels=out_channels,
            dtype=dtype,
        ),
    }


def mlp_apply(x, params):
    # x shape: [batch, channels, size_x]
    x = conv1d_kernel1_apply(x, params["conv0"])
    x = gelu(x)
    x = conv1d_kernel1_apply(x, params["conv2"])
    return x


def get_grid(batchsize, size_x, dtype):
    # Create grid with shape [batch, size_x, 1].
    gridx = jnp.linspace(0, 1, size_x, dtype=dtype)
    gridx = gridx.reshape(1, size_x, 1)
    gridx = jnp.repeat(gridx, batchsize, axis=0)
    return gridx


def init_fno1d_params(
    key,
    modes,
    width,
    num_layers=4,
    dtype=jnp.float32,
):
    # Initialize all parameters for FNO1d.
    keys = jax.random.split(key, 1 + num_layers * 3 + 1)

    key_index = 0

    params = {
        "p": init_linear_params(
            keys[key_index],
            in_features=2,
            out_features=width,
            dtype=dtype,
        ),
        "conv_layers": [],
        "mlp_layers": [],
        "w_layers": [],
        "q": None,
    }

    key_index += 1

    for _ in range(num_layers):
        params["conv_layers"].append(
            init_spectral_conv1d_params(
                keys[key_index],
                in_channels=width,
                out_channels=width,
                modes1=modes,
                dtype=dtype,
            )
        )
        key_index += 1

        params["mlp_layers"].append(
            init_mlp_params(
                keys[key_index],
                in_channels=width,
                out_channels=width,
                mid_channels=width,
                dtype=dtype,
            )
        )
        key_index += 1

        params["w_layers"].append(
            init_conv1d_kernel1_params(
                keys[key_index],
                in_channels=width,
                out_channels=width,
                dtype=dtype,
            )
        )
        key_index += 1

    params["q"] = init_mlp_params(
        keys[key_index],
        in_channels=width,
        out_channels=1,
        mid_channels=width * 2,
        dtype=dtype,
    )

    return params


def fno1d_apply(
    params,
    x,
    modes,
    width,
    num_layers=4,
    dtype=jnp.float32,
):
    # Equivalent to PyTorch FNO1d.forward.
    x = x.astype(dtype)

    batchsize = x.shape[0]
    size_x = x.shape[1]

    grid = get_grid(batchsize, size_x, dtype)

    x = jnp.concatenate((x, grid), axis=-1)

    x = linear_apply(x, params["p"])

    x = jnp.transpose(x, axes=(0, 2, 1))

    for i in range(num_layers):
        x1 = spectral_conv1d_apply(
            x,
            params["conv_layers"][i],
            out_channels=width,
            modes1=modes,
        )

        x1 = mlp_apply(x1, params["mlp_layers"][i])

        x2 = conv1d_kernel1_apply(x, params["w_layers"][i])

        x = x1 + x2

        if i < num_layers - 1:
            x = gelu(x)

    x = mlp_apply(x, params["q"])

    x = jnp.transpose(x, axes=(0, 2, 1))

    return x


def cast_fno1d_params(params, dtype=jnp.float32):
    # Convert all floating and complex parameters to the target dtype.
    if dtype == jnp.float64:
        complex_dtype = jnp.complex128
    else:
        complex_dtype = jnp.complex64

    def cast_one(x):
        if jnp.iscomplexobj(x):
            return x.astype(complex_dtype)
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(cast_one, params)


class FNO1dJAX:
    def __init__(
        self,
        modes,
        width,
        num_layers=4,
        dtype=jnp.float32,
        seed=0,
    ):
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.dtype = dtype

        key = jax.random.PRNGKey(seed)

        self.params = init_fno1d_params(
            key=key,
            modes=modes,
            width=width,
            num_layers=num_layers,
            dtype=dtype,
        )

    def __call__(self, x):
        return fno1d_apply(
            params=self.params,
            x=x,
            modes=self.modes,
            width=self.width,
            num_layers=self.num_layers,
            dtype=self.dtype,
        )

    def apply(self, params, x):
        return fno1d_apply(
            params=params,
            x=x,
            modes=self.modes,
            width=self.width,
            num_layers=self.num_layers,
            dtype=self.dtype,
        )

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.params = cast_fno1d_params(self.params, dtype=dtype)
        return self


if __name__ == "__main__":
    np.random.seed(0)

    modes = 8
    width = 16
    num_layers = 4
    dtype = jnp.float32

    model = FNO1dJAX(
        modes=modes,
        width=width,
        num_layers=num_layers,
        dtype=dtype,
        seed=0,
    )

    x = np.random.randn(2, 32, 1).astype(np.float32)
    x = jnp.asarray(x)

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output dtype:", y.dtype)
    print("Output sample:")
    print(np.asarray(y[0, :5, 0]))