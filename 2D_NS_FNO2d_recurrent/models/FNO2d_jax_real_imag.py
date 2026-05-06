import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def gelu(x):
    # Use exact GELU, matching PyTorch default GELU.
    return jax.nn.gelu(x, approximate=False)


def init_linear_params(key, in_features, out_features, dtype=jnp.float32):
    # Initialize a Linear layer with PyTorch-like uniform initialization.
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
    # x shape: [batch, size_x, size_y, in_features]
    # weight shape: [out_features, in_features]
    return jnp.einsum("bxyi,oi->bxyo", x, params["weight"]) + params["bias"]


def init_conv2d_kernel1_params(key, in_channels, out_channels, dtype=jnp.float32):
    # Initialize a Conv2d layer with kernel_size=1.
    k1, k2 = jax.random.split(key)

    limit = 1.0 / jnp.sqrt(in_channels)

    weight = jax.random.uniform(
        k1,
        shape=(out_channels, in_channels, 1, 1),
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


def conv2d_kernel1_apply(x, params):
    # x shape: [batch, in_channels, size_x, size_y]
    # weight shape: [out_channels, in_channels, 1, 1]
    return (
        jnp.einsum("bixy,oiuv->boxy", x, params["weight"])
        + params["bias"][None, :, None, None]
    )


def init_spectral_conv2d_params(
    key,
    in_channels,
    out_channels,
    modes1,
    modes2,
    dtype=jnp.float32,
):
    # Initialize Fourier weights as real and imaginary tensors separately.
    scale = 1.0 / (in_channels * out_channels)

    if dtype == jnp.float64:
        real_dtype = jnp.float64
    else:
        real_dtype = jnp.float32

    k1, k2, k3, k4 = jax.random.split(key, 4)

    weights1_real = scale * jax.random.uniform(
        k1,
        shape=(in_channels, out_channels, modes1, modes2),
        dtype=real_dtype,
    )

    weights1_imag = scale * jax.random.uniform(
        k2,
        shape=(in_channels, out_channels, modes1, modes2),
        dtype=real_dtype,
    )

    weights2_real = scale * jax.random.uniform(
        k3,
        shape=(in_channels, out_channels, modes1, modes2),
        dtype=real_dtype,
    )

    weights2_imag = scale * jax.random.uniform(
        k4,
        shape=(in_channels, out_channels, modes1, modes2),
        dtype=real_dtype,
    )

    return {
        "weights1_real": weights1_real,
        "weights1_imag": weights1_imag,
        "weights2_real": weights2_real,
        "weights2_imag": weights2_imag,
    }


def complex_einsum2d_from_real_parts(x_ft_low, weights_real, weights_imag):
    # Manual complex multiplication and channel contraction.
    # x_ft_low shape: [batch, in_channels, modes1, modes2]
    # weights_* shape: [in_channels, out_channels, modes1, modes2]
    x_real = jnp.real(x_ft_low)
    x_imag = jnp.imag(x_ft_low)

    weights_real = weights_real.astype(x_real.dtype)
    weights_imag = weights_imag.astype(x_real.dtype)

    # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    out_real = (
        jnp.einsum("bixy,ioxy->boxy", x_real, weights_real)
        - jnp.einsum("bixy,ioxy->boxy", x_imag, weights_imag)
    )
    out_imag = (
        jnp.einsum("bixy,ioxy->boxy", x_real, weights_imag)
        + jnp.einsum("bixy,ioxy->boxy", x_imag, weights_real)
    )

    return out_real, out_imag


def spectral_conv2d_apply(x, params, out_channels, modes1, modes2):
    # x shape: [batch, in_channels, size_x, size_y]
    # Fourier weights are stored as real and imaginary tensors separately.
    batchsize = x.shape[0]
    size_x = x.shape[-2]
    size_y = x.shape[-1]

    x_ft = jnp.fft.rfft2(x, axes=(-2, -1))

    top_modes_real, top_modes_imag = complex_einsum2d_from_real_parts(
        x_ft[:, :, :modes1, :modes2],
        params["weights1_real"],
        params["weights1_imag"],
    )

    bottom_modes_real, bottom_modes_imag = complex_einsum2d_from_real_parts(
        x_ft[:, :, -modes1:, :modes2],
        params["weights2_real"],
        params["weights2_imag"],
    )

    out_ft_real = jnp.zeros(
        shape=(batchsize, out_channels, size_x, size_y // 2 + 1),
        dtype=top_modes_real.dtype,
    )
    out_ft_imag = jnp.zeros(
        shape=(batchsize, out_channels, size_x, size_y // 2 + 1),
        dtype=top_modes_real.dtype,
    )

    out_ft_real = out_ft_real.at[:, :, :modes1, :modes2].set(top_modes_real)
    out_ft_imag = out_ft_imag.at[:, :, :modes1, :modes2].set(top_modes_imag)

    out_ft_real = out_ft_real.at[:, :, -modes1:, :modes2].set(bottom_modes_real)
    out_ft_imag = out_ft_imag.at[:, :, -modes1:, :modes2].set(bottom_modes_imag)

    out_ft = out_ft_real + 1j * out_ft_imag
    x = jnp.fft.irfft2(out_ft, s=(size_x, size_y), axes=(-2, -1))

    return x


def init_mlp_params(
    key,
    in_channels,
    out_channels,
    mid_channels,
    dtype=jnp.float32,
):
    # Equivalent to:
    # Conv2d(in_channels, mid_channels, 1)
    # GELU
    # Conv2d(mid_channels, out_channels, 1)
    k1, k2 = jax.random.split(key)

    return {
        "conv0": init_conv2d_kernel1_params(
            k1,
            in_channels=in_channels,
            out_channels=mid_channels,
            dtype=dtype,
        ),
        "conv2": init_conv2d_kernel1_params(
            k2,
            in_channels=mid_channels,
            out_channels=out_channels,
            dtype=dtype,
        ),
    }


def mlp_apply(x, params):
    # x shape: [batch, channels, size_x, size_y]
    x = conv2d_kernel1_apply(x, params["conv0"])
    x = gelu(x)
    x = conv2d_kernel1_apply(x, params["conv2"])
    return x


def get_grid(batchsize, size_x, size_y, dtype):
    # Create grid with shape [batch, size_x, size_y, 2].
    gridx = jnp.linspace(0, 1, size_x, dtype=dtype)
    gridy = jnp.linspace(0, 1, size_y, dtype=dtype)

    gridx = gridx.reshape(1, size_x, 1, 1)
    gridx = jnp.repeat(gridx, batchsize, axis=0)
    gridx = jnp.repeat(gridx, size_y, axis=2)

    gridy = gridy.reshape(1, 1, size_y, 1)
    gridy = jnp.repeat(gridy, batchsize, axis=0)
    gridy = jnp.repeat(gridy, size_x, axis=1)

    return jnp.concatenate((gridx, gridy), axis=-1)


def init_fno2d_params(
    key,
    modes1,
    modes2,
    width,
    num_layers=4,
    in_channels=10,
    dtype=jnp.float32,
):
    # Initialize all parameters for FNO2d.
    keys = jax.random.split(key, 1 + num_layers * 3 + 1)

    key_index = 0

    params = {
        "p": init_linear_params(
            keys[key_index],
            in_features=in_channels + 2,
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
            init_spectral_conv2d_params(
                keys[key_index],
                in_channels=width,
                out_channels=width,
                modes1=modes1,
                modes2=modes2,
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
            init_conv2d_kernel1_params(
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
        mid_channels=width * 4,
        dtype=dtype,
    )

    return params


def fno2d_apply(
    params,
    x,
    modes1,
    modes2,
    width,
    num_layers=4,
    dtype=jnp.float32,
):
    # Equivalent to PyTorch FNO2d.forward.
    x = x.astype(dtype)

    batchsize = x.shape[0]
    size_x = x.shape[1]
    size_y = x.shape[2]

    grid = get_grid(
        batchsize=batchsize,
        size_x=size_x,
        size_y=size_y,
        dtype=dtype,
    )

    x = jnp.concatenate((x, grid), axis=-1)

    x = linear_apply(x, params["p"])

    x = jnp.transpose(x, axes=(0, 3, 1, 2))

    for i in range(num_layers):
        x1 = spectral_conv2d_apply(
            x,
            params["conv_layers"][i],
            out_channels=width,
            modes1=modes1,
            modes2=modes2,
        )

        x1 = mlp_apply(x1, params["mlp_layers"][i])

        x2 = conv2d_kernel1_apply(x, params["w_layers"][i])

        x = x1 + x2

        if i < num_layers - 1:
            x = gelu(x)

    x = mlp_apply(x, params["q"])

    x = jnp.transpose(x, axes=(0, 2, 3, 1))

    return x


def recurrent_predictor_apply(
    params,
    x_init,
    modes1,
    modes2,
    width,
    num_layers=4,
    T_out=10,
    step=1,
    dtype=jnp.float32,
):
    # Equivalent to RecurrentPredictor.forward.
    # x_init shape: [batch, size_x, size_y, T_in]
    outputs = []

    x = x_init.astype(dtype)

    for _ in range(0, T_out, step):
        y_pred = fno2d_apply(
            params=params,
            x=x,
            modes1=modes1,
            modes2=modes2,
            width=width,
            num_layers=num_layers,
            dtype=dtype,
        )

        outputs.append(y_pred)

        x = jnp.concatenate(
            [x[..., step:], y_pred],
            axis=-1,
        )

    return jnp.concatenate(outputs, axis=-1)


def cast_fno2d_params(params, dtype=jnp.float32):
    # Convert all floating parameters to the target dtype.
    def cast_one(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(cast_one, params)


class FNO2dJAX:
    def __init__(
        self,
        modes1,
        modes2,
        width,
        num_layers=4,
        in_channels=10,
        dtype=jnp.float32,
        seed=0,
    ):
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.dtype = dtype

        key = jax.random.PRNGKey(seed)

        self.params = init_fno2d_params(
            key=key,
            modes1=modes1,
            modes2=modes2,
            width=width,
            num_layers=num_layers,
            in_channels=in_channels,
            dtype=dtype,
        )

    def __call__(self, x):
        return fno2d_apply(
            params=self.params,
            x=x,
            modes1=self.modes1,
            modes2=self.modes2,
            width=self.width,
            num_layers=self.num_layers,
            dtype=self.dtype,
        )

    def apply(self, params, x):
        return fno2d_apply(
            params=params,
            x=x,
            modes1=self.modes1,
            modes2=self.modes2,
            width=self.width,
            num_layers=self.num_layers,
            dtype=self.dtype,
        )

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.params = cast_fno2d_params(self.params, dtype=dtype)
        return self


class RecurrentPredictorJAX:
    def __init__(
        self,
        model,
        T_out=10,
        step=1,
    ):
        self.model = model
        self.T_out = T_out
        self.step = step

    def __call__(self, x_init):
        return recurrent_predictor_apply(
            params=self.model.params,
            x_init=x_init,
            modes1=self.model.modes1,
            modes2=self.model.modes2,
            width=self.model.width,
            num_layers=self.model.num_layers,
            T_out=self.T_out,
            step=self.step,
            dtype=self.model.dtype,
        )


if __name__ == "__main__":
    np.random.seed(0)

    modes1 = 12
    modes2 = 12
    width = 20
    num_layers = 4
    in_channels = 10
    dtype = jnp.float32

    model = FNO2dJAX(
        modes1=modes1,
        modes2=modes2,
        width=width,
        num_layers=num_layers,
        in_channels=in_channels,
        dtype=dtype,
        seed=0,
    )

    recurrent_model = RecurrentPredictorJAX(
        model=model,
        T_out=10,
        step=1,
    )

    x = np.random.randn(2, 32, 32, 10).astype(np.float32)
    x = jnp.asarray(x)

    y_one_step = model(x)
    y_recurrent = recurrent_model(x)

    print("Input shape:", x.shape)
    print("One-step output shape:", y_one_step.shape)
    print("Recurrent output shape:", y_recurrent.shape)
    print("One-step output dtype:", y_one_step.dtype)
    print("Recurrent output dtype:", y_recurrent.dtype)
    print("One-step output sample:")
    print(np.asarray(y_one_step[0, :2, :2, 0]))
    print("Recurrent output sample:")
    print(np.asarray(y_recurrent[0, :2, :2, :3]))
