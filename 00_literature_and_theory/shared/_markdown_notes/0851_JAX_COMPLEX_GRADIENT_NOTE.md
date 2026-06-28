# JAX Complex Gradient Note for FNO Training

This note records the issue observed when training Fourier Neural Operator
models with complex Fourier weights in JAX, and the convention this repository
should use going forward.

## Short Conclusion

For FNO spectral weights, do not train JAX models by storing Fourier weights as
native complex arrays and directly applying the raw JAX gradient to the
parameters.

Use the real/imaginary split implementation instead:

- `1D_Burgers/models/FNO1d_jax_real_imag.py`
- `2D_NS_FNO2d_recurrent/models/FNO2d_jax_real_imag.py`

In training commands, use:

```bash
FRAMEWORKS=pytorch,jax_real_imag
```

The older direct complex JAX implementations are useful for debugging, but they
should not be the default model for training or attack experiments:

- `1D_Burgers/models/FNO1d_jax.py`
- `2D_NS_FNO2d_recurrent/models/FNO2d_jax.py`

## What Went Wrong

The FNO spectral convolution layers use complex Fourier weights. The training
loss is real-valued, but some trainable parameters are complex-valued.

For a complex parameter

```text
z = x + i y
```

a real-valued loss is really a function of two real coordinates:

```text
L(z) = L(x, y)
```

The parameter update that matches ordinary real-coordinate gradient descent is:

```text
x <- x - lr * dL/dx
y <- y - lr * dL/dy
```

In complex notation, this corresponds to using the conjugate-Wirtinger style
descent direction. PyTorch and JAX do not expose complex gradients with the same
convention for this use case. In our local FNO experiments, the raw JAX gradient
for complex Fourier weights was effectively conjugated relative to the PyTorch
descent direction. If that raw JAX complex gradient is sent directly into the
optimizer, the imaginary part is updated in the wrong direction relative to the
real-coordinate update we want.

That is the important point:

```text
raw JAX complex gradient != the update direction we want for matching PyTorch
```

If we update complex parameters directly using the raw JAX gradient, the model
is not following the same descent direction as the PyTorch model. The result can
be slower convergence, worse final error, unstable loss spikes, and predictions
that visibly drift away from the PyTorch baseline.

## Observed Symptoms

In the 1D Burgers FNO experiments, the direct complex JAX model showed clear
problems:

- the loss decreased more slowly than PyTorch;
- the final relative error was worse;
- longer runs could show sudden large relative-loss spikes;
- some final predictions became nearly flat and did not match the ground truth;
- gradient comparison plots showed that PyTorch and direct complex JAX gradients
  already differ in early training steps, and the difference accumulates.

The real/imaginary split JAX model behaved much closer to the PyTorch baseline.

## Why Conjugating the Gradient Helps but Is Not the Best Default

One possible fix is to conjugate complex gradients before the optimizer update:

```python
def conjugate_complex_grads(grads):
    return jax.tree_util.tree_map(
        lambda g: jnp.conj(g) if jnp.iscomplexobj(g) else g,
        grads,
    )
```

This is the idea behind:

```text
1D_Burgers/models/FNO1d_jax_conjugate.py
```

This can correct the immediate gradient direction issue for complex leaves.
However, it is still less clean than splitting real and imaginary parts,
especially when using adaptive optimizers such as Adam. With native complex
parameter leaves, the optimizer state and update arithmetic are still operating
on complex arrays. That can fail to exactly match the behavior of treating the
real and imaginary parts as two independent real trainable tensors.

For debugging, the conjugated-gradient version is useful. For production
training and attack experiments, it is not the preferred default.

## Recommended Implementation

The preferred implementation stores each complex Fourier weight as two real
tensors:

```text
weights_real
weights_imag
```

The forward pass manually reconstructs the complex multiplication:

```text
(a + i b) * (c + i d) = (a c - b d) + i (a d + b c)
```

This has three practical advantages:

1. JAX autodiff sees all trainable parameters as real arrays.
2. The returned gradients are ordinary real gradients for ordinary real
   parameters.
3. Adam and other optimizers update the real and imaginary parts as separate
   real tensors, which is the behavior we want.

This is why the repository should use the real/imaginary split versions:

```text
1D_Burgers/models/FNO1d_jax_real_imag.py
2D_NS_FNO2d_recurrent/models/FNO2d_jax_real_imag.py
```

## Practical Rule

For future FNO experiments:

- PyTorch baseline: use the normal PyTorch model.
- JAX baseline: use the `jax_real_imag` model.
- Do not use direct complex JAX as the main result.
- Only use direct complex JAX or conjugated-gradient JAX for ablation/debugging.

For 1D Burgers attacks, use the saved attack-ready models under:

```text
1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/
```

For 2D Navier-Stokes training, use:

```bash
PROBLEMS=ns \
FRAMEWORKS=pytorch,jax_real_imag \
MODES_2D=12 \
WIDTH_2D=20 \
bash tools/run_full_fno_training_suite.sh
```

The key reason is not cosmetic. It changes the optimization direction for
complex Fourier parameters. If the gradient convention is wrong, every update
can move the imaginary component in the wrong direction, and the error can
compound across training.
