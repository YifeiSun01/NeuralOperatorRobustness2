# Modified Exponax Solver Code Walkthrough

This note documents the installed `exponax` package currently used by this
workspace, with a focus on the two solvers used here:

- 1D periodic Burgers
- 2D periodic Navier-Stokes in vorticity form, using the modified Zongyi forcing

The installed package is:

```text
package: exponax 0.1.0
source: https://github.com/YifeiSun01/modified_exponax.git
commit: febe20b2103192654b256243eb464742616e95df
installed path:
  adv_robust/lib/python3.12/site-packages/exponax/
```

## High-Level Design

Exponax solves periodic PDEs with Fourier pseudo-spectral spatial derivatives
and Exponential Time Differencing Runge-Kutta time integration.

The common PDE form is:

```text
u_t = L u + N(u)
```

where:

- `L` is a linear differential operator represented diagonally in Fourier space.
- `N(u)` is a nonlinear function evaluated pseudo-spectrally.
- The state convention is channel-first and unbatched:
  - 1D scalar field: `(1, N)`
  - 2D scalar vorticity field: `(1, N, N)`
  - batching is done outside with `jax.vmap` or `jax.lax.map`.

The shared call chain for a single step is:

```text
concrete stepper class
  -> BaseStepper.__init__
      -> build_derivative_operator(...)
      -> concrete_stepper._build_linear_operator(...)
      -> concrete_stepper._build_nonlinear_fun(...)
      -> ETDRK{order}(...)
  -> stepper(u)
      -> BaseStepper.__call__
      -> BaseStepper.step
      -> exponax.fft(u)
      -> ETDRK.step_fourier(u_hat)
      -> nonlinear_fun(u_hat) as needed
      -> exponax.ifft(u_next_hat)
      -> u_next
```

The key shared files are:

| Layer | File | Main role |
|---|---|---|
| package API | `exponax/__init__.py` | exposes `stepper`, `fft`, `ifft`, `repeat`, `rollout`, etc. |
| PDE registry | `exponax/stepper/__init__.py` | exports public PDE steppers like `Burgers` and `NavierStokesVorticity` |
| generic stepper base | `exponax/_base_stepper.py` | constructs `L`, `N`, and the ETDRK integrator |
| spectral utilities | `exponax/_spectral.py` | wavenumbers, derivative operator, Laplacian, FFT/IFFT, dealiasing masks, modified forcing |
| nonlinear base | `exponax/nonlin_fun/_base.py` | shared FFT/IFFT/dealias helper for nonlinear terms |
| time integration | `exponax/etdrk/_etdrk_0.py` through `_etdrk_4.py` | ETDRK time stepping |
| repetition helpers | `exponax/_utils.py` | `repeat(...)` and `rollout(...)` use `jax.lax.scan` |

## Periodic Boundary Convention

The package assumes periodic domains. The grid has `N` degrees of freedom and
excludes the right boundary point:

```text
[0, L)
```

This is why `BaseStepper` sets:

```python
self.dx = domain_extent / num_points
```

and why `_spectral.py` uses real FFTs:

```python
jnp.fft.rfftn(...)
jnp.fft.irfftn(...)
```

## Shared BaseStepper Flow

File:

```text
adv_robust/lib/python3.12/site-packages/exponax/_base_stepper.py
```

The important constructor section is lines 101-156:

```python
derivative_operator = build_derivative_operator(
    num_spatial_dims, domain_extent, num_points
)

linear_operator = self._build_linear_operator(derivative_operator)
nonlinear_fun = self._build_nonlinear_fun(derivative_operator)

if order == 4:
    self._integrator = ETDRK4(
        dt,
        linear_operator,
        nonlinear_fun,
        num_circle_points=num_circle_points,
        circle_radius=circle_radius,
    )
```

The important step section is lines 200-238:

```python
u_hat = fft(u, num_spatial_dims=self.num_spatial_dims)
u_next_hat = self.step_fourier(u_hat)
u_next = ifft(
    u_next_hat,
    num_spatial_dims=self.num_spatial_dims,
    num_points=self.num_points,
)
return u_next
```

So concrete PDE classes do not implement the full time stepping loop. They only
tell `BaseStepper` what `L` and `N` are.

## Spectral Operators

File:

```text
adv_robust/lib/python3.12/site-packages/exponax/_spectral.py
```

Important functions:

| Function | Lines | Role |
|---|---:|---|
| `build_scaled_wavenumbers` | 52 | builds Fourier wavenumbers scaled by `2*pi/L` |
| `build_derivative_operator` | 86 | returns `1j * scaled_wavenumbers` |
| `build_laplace_operator` | 118 | returns `sum(derivative_operator ** order)` |
| `low_pass_filter_mask` | 271 | builds the dealiasing mask |
| `fft` | 713 | wraps `jnp.fft.rfftn` |
| `ifft` | 758 | wraps `jnp.fft.irfftn` |

For a periodic PDE, derivatives become multiplication in Fourier space:

```text
partial_x  ->  i k_x
Laplacian  ->  - |k|^2
```

That is why Burgers diffusion and NS viscosity can be represented as:

```python
diffusivity * build_laplace_operator(derivative_operator)
```

### Modified Forcing Utilities

The modified package adds forcing helpers in `_spectral.py`.

`build_forcing(...)`, lines 522-541:

```python
t = jnp.linspace(0, 1, num_points+1)[:-1]
coords = jnp.meshgrid(*[t]*num_spatial_dims, indexing=indexing)
X, Y = coords[0], coords[1]
f_physical = 0.1 * (jnp.sin(2*jnp.pi*(X + Y)) + jnp.cos(2*jnp.pi*(X + Y)))
f_hat = jnp.fft.rfftn(f_physical)
return injection_scale * f_hat
```

`build_forcing_patterns(...)`, lines 604-627, supports named patterns:

```text
ringsCos, sBands, isoCircles, petals, ringsL1, ringsLinf
```

The current dataset-generation code uses `NavierStokesVorticityZongyi`, which
uses `build_forcing(...)`, not the named pattern version.

## ETDRK4 Time Integrator

Your current Burgers and 2D NS generation scripts instantiate the steppers with
`order=4`, so the active time integrator is:

```text
adv_robust/lib/python3.12/site-packages/exponax/etdrk/_etdrk_4.py
```

`ETDRK4.__init__`, lines 79-109:

- stores the nonlinear function
- builds `exp(dt * L)` and `exp(0.5 * dt * L)`
- computes contour-integral coefficients using roots of unity

The core step is lines 111-137:

```python
u_nonlin_hat = self._nonlinear_fun(u_hat)
u_stage_1_hat = self._half_exp_term * u_hat + self._coef_1 * u_nonlin_hat

u_stage_1_nonlin_hat = self._nonlinear_fun(u_stage_1_hat)
u_stage_2_hat = (
    self._half_exp_term * u_hat + self._coef_2 * u_stage_1_nonlin_hat
)

u_stage_2_nonlin_hat = self._nonlinear_fun(u_stage_2_hat)
u_stage_3_hat = self._half_exp_term * u_stage_1_hat + self._coef_3 * (
    2 * u_stage_2_nonlin_hat - u_nonlin_hat
)

u_stage_3_nonlin_hat = self._nonlinear_fun(u_stage_3_hat)

u_next_hat = (
    self._exp_term * u_hat
    + self._coef_4 * u_nonlin_hat
    + self._coef_5 * 2 * (u_stage_1_nonlin_hat + u_stage_2_nonlin_hat)
    + self._coef_6 * u_stage_3_nonlin_hat
)
```

So each physical step is:

```text
FFT state -> ETDRK4 stages in Fourier space -> IFFT next state
```

## 1D Periodic Burgers Solver

### Project Entry Point

File:

```text
1D_Burgers/data_generation/generate_burgers_exponax_batched.py
```

The important function is `make_burgers_final_fn(...)`, lines 86-113:

```python
steps = int(round(t_final / dt))
stepper = ex.stepper.Burgers(
    1,
    domain_extent,
    nx,
    dt,
    diffusivity=nu,
    convection_scale=1.0,
    order=4,
    conservative=conservative,
)
repeat_fn = ex.repeat(stepper, steps)

def one(u0):
    full_ic = jnp.expand_dims(u0.squeeze(), axis=0).astype(jnp.float32)
    return repeat_fn(full_ic)[0]

return jax.jit(jax.vmap(one, in_axes=0, out_axes=0))
```

The project script converts each initial condition from shape:

```text
(N,)
```

to Exponax shape:

```text
(1, N)
```

Then it calls `repeat_fn(full_ic)` to advance `steps = t_final / dt` times.
Finally, it removes the single channel with `[0]`, so saved data has shape:

```text
x: (1500, 1024)
y: (1500, 1024)
```

### Exponax Burgers Class

File:

```text
adv_robust/lib/python3.12/site-packages/exponax/stepper/_burgers.py
```

Class:

```python
class Burgers(BaseStepper):
```

Important constructor lines:

- lines 15-30 define arguments.
- lines 122-126 choose `num_channels`.
- lines 128-137 call `BaseStepper.__init__`.

For your 1D case:

```python
ex.stepper.Burgers(
    num_spatial_dims=1,
    domain_extent=2.0,
    num_points=1024,
    dt=0.001,
    diffusivity=nu,
    convection_scale=1.0,
    order=4,
    conservative=False,
)
```

Because `single_channel=False` and `num_spatial_dims=1`, `num_channels=1`.

### Burgers Linear Operator

Lines 139-144:

```python
def _build_linear_operator(self, derivative_operator):
    return self.diffusivity * build_laplace_operator(derivative_operator)
```

This represents:

```text
nu * u_xx
```

in Fourier space:

```text
nu * (-k^2) * u_hat
```

### Burgers Nonlinear Operator

Lines 146-158:

```python
return ConvectionNonlinearFun(
    self.num_spatial_dims,
    self.num_points,
    derivative_operator=derivative_operator,
    dealiasing_fraction=self.dealiasing_fraction,
    scale=self.convection_scale,
    single_channel=self.single_channel,
    conservative=self.conservative,
)
```

File:

```text
adv_robust/lib/python3.12/site-packages/exponax/nonlin_fun/_convection.py
```

For the default non-conservative 1D case, `ConvectionNonlinearFun.__call__`
uses `_multi_channel_nonconservative_eval(...)` because:

```text
single_channel = False
conservative = False
num_channels = num_spatial_dims = 1
```

The relevant implementation is lines 141-175:

```python
u_hat_dealiased = self.dealias(u_hat)
u = self.ifft(u_hat_dealiased)
nabla_u = self.ifft(
    self.derivative_operator[None, :] * u_hat_dealiased[:, None]
)
conv_u = jnp.sum(
    u[None, :] * nabla_u,
    axis=1,
)
return -self.scale * self.fft(conv_u)
```

For 1D this is:

```text
N(u) = - b * u * u_x
```

With `conservative=True`, it instead uses the conservative form:

```text
N(u) = - 1/2 * b * (u^2)_x
```

implemented in `_multi_channel_conservative_eval(...)`, lines 104-139.

### Burgers Full Call Chain

```text
generate_burgers_exponax_batched.py
  -> make_burgers_final_fn(...)
  -> ex.stepper.Burgers(...)
      -> stepper/_burgers.py:Burgers.__init__
      -> _base_stepper.py:BaseStepper.__init__
          -> _spectral.py:build_derivative_operator
          -> Burgers._build_linear_operator
              -> _spectral.py:build_laplace_operator
          -> Burgers._build_nonlinear_fun
              -> nonlin_fun/_convection.py:ConvectionNonlinearFun
          -> etdrk/_etdrk_4.py:ETDRK4
  -> ex.repeat(stepper, steps)
      -> _utils.py:repeat
      -> jax.lax.scan(stepper, length=steps)
  -> stepper(u)
      -> BaseStepper.__call__
      -> BaseStepper.step
      -> _spectral.py:fft
      -> ETDRK4.step_fourier
          -> ConvectionNonlinearFun(u_hat)
          -> ETDRK4 stage updates
      -> _spectral.py:ifft
  -> y final state
```

## 2D Periodic Navier-Stokes Solver

### Project Entry Point

File:

```text
2D_NS_FNO2d_recurrent/data_generation/generate_ns_real_initial_batched.py
```

The important function is `make_ns_rollout_fn(...)`, lines 76-123.

It constructs the modified Exponax stepper:

```python
stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
    2,
    domain_extent,
    nx,
    fixed_step,
    diffusivity=nu,
    order=4,
    num_circle_points=16,
    dealiasing_fraction=2 / 3,
)
```

Important note: this class is not exported as `ex.stepper.NavierStokesVorticityZongyi`
in `stepper/__init__.py`. The project calls the private module path:

```text
ex.stepper._navier_stokes.NavierStokesVorticityZongyi
```

The project's current wrapper also rotates/flips the initial field before
passing it to Exponax:

```python
u = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))
```

Then it performs:

```text
fixed_step = 0.005
steps_per_second = 1 / fixed_step = 200
t_final = 20
```

So each sample runs:

```text
20 seconds * 200 micro-steps per second = 4000 Exponax steps
```

The code only saves integer-second frames:

```python
def micro(carry, _):
    return stepper(carry[None, ...])[0], None

def one_second(carry, _):
    u_next, _ = jax.lax.scan(micro, carry, None, length=steps_per_second)
    return u_next, u_next

_, seconds = jax.lax.scan(one_second, u, None, length=t_final)
seq = jnp.concatenate([u[None, ...], seconds], axis=0)
return jnp.swapaxes(seq, -1, -2)
```

For exact sequential reproducibility, the current script default is:

```python
solver_mode = "lax-map"
```

which does:

```python
seq = jax.lax.map(one, u0_batch)
```

instead of `jax.vmap(one, ...)`.

### Exponax Zongyi Navier-Stokes Class

File:

```text
adv_robust/lib/python3.12/site-packages/exponax/stepper/_navier_stokes.py
```

Class:

```python
class NavierStokesVorticityZongyi(BaseStepper):
```

Important lines:

- lines 313-333 define the modified class and constructor arguments.
- lines 334-350 enforce 2D, store parameters, and call `BaseStepper`.
- lines 352-358 define the linear operator.
- lines 360-370 define the nonlinear function.

### 2D NS Linear Operator

Lines 352-358:

```python
return self.diffusivity * build_laplace_operator(
    derivative_operator, order=2
) + self.drag * build_laplace_operator(derivative_operator, order=0)
```

This represents:

```text
nu * Delta omega + drag * omega
```

In your current generation command:

```text
nu = 1e-5
drag = 0.0
```

so the active linear part is essentially:

```text
1e-5 * Delta omega
```

### 2D NS Nonlinear Operator

Lines 360-370:

```python
return VorticityConvection2dZongyi(
    self.num_spatial_dims,
    self.num_points,
    convection_scale=self.vorticity_convection_scale,
    derivative_operator=derivative_operator,
    dealiasing_fraction=self.dealiasing_fraction,
)
```

File:

```text
adv_robust/lib/python3.12/site-packages/exponax/nonlin_fun/_vorticity_convection.py
```

Base class:

```python
class VorticityConvection2d(BaseNonlinearFun):
```

Core vorticity convection lines 83-104:

```python
vorticity_hat = u_hat
stream_function_hat = self.inv_laplacian * vorticity_hat

u_hat = +self.derivative_operator[1:2] * stream_function_hat
v_hat = -self.derivative_operator[0:1] * stream_function_hat
del_vorticity_del_x_hat = self.derivative_operator[0:1] * vorticity_hat
del_vorticity_del_y_hat = self.derivative_operator[1:2] * vorticity_hat

u = self.ifft(self.dealias(u_hat))
v = self.ifft(self.dealias(v_hat))
del_vorticity_del_x = self.ifft(self.dealias(del_vorticity_del_x_hat))
del_vorticity_del_y = self.ifft(self.dealias(del_vorticity_del_y_hat))

convection = u * del_vorticity_del_x + v * del_vorticity_del_y
convection_hat = self.fft(convection)
return -self.convection_scale * convection_hat
```

Mathematically this is:

```text
psi = Delta^{-1} omega
velocity = (partial_y psi, -partial_x psi)
convection = velocity dot grad(omega)
N_base(omega) = - convection
```

### Zongyi Modification: Added Forcing

`VorticityConvection2dZongyi` is lines 190-215:

```python
class VorticityConvection2dZongyi(VorticityConvection2d):
    def __init__(..., injection_scale: float = 1.0, **kwargs):
        super().__init__(...)
        self.injection = build_forcing(...)

    def __call__(self, u_hat):
        neg_convection_hat = super().__call__(u_hat)
        return neg_convection_hat + self.injection
```

The forcing is built by `_spectral.py:build_forcing(...)`, lines 522-541:

```python
f_physical = 0.1 * (jnp.sin(2*jnp.pi*(X + Y)) + jnp.cos(2*jnp.pi*(X + Y)))
f_hat = jnp.fft.rfftn(f_physical)
return injection_scale * f_hat
```

Therefore the modified Zongyi NS nonlinear term is:

```text
N_Zongyi(omega) = - velocity dot grad(omega) + forcing_hat
```

and the full PDE integrated by the stepper is:

```text
omega_t = nu * Delta omega + drag * omega
          - velocity dot grad(omega)
          + forcing
```

### 2D NS Full Call Chain

```text
generate_ns_real_initial_batched.py
  -> load_source_x(...)
      -> reads dataset["x"] from NS_data_zongyi_{split}_all_frame_real_initial.pt
  -> spectral_upsample(...)
      -> torch FFT zero-padding from 64x64 to 256x256
  -> make_ns_rollout_fn(...)
      -> ex.stepper._navier_stokes.NavierStokesVorticityZongyi(...)
          -> stepper/_navier_stokes.py:NavierStokesVorticityZongyi.__init__
          -> _base_stepper.py:BaseStepper.__init__
              -> _spectral.py:build_derivative_operator
              -> NavierStokesVorticityZongyi._build_linear_operator
                  -> _spectral.py:build_laplace_operator
              -> NavierStokesVorticityZongyi._build_nonlinear_fun
                  -> nonlin_fun/_vorticity_convection.py:VorticityConvection2dZongyi
                      -> VorticityConvection2d.__init__
                          -> _spectral.py:build_laplace_operator
                          -> inverse Laplacian for Poisson solve
                      -> _spectral.py:build_forcing
              -> etdrk/_etdrk_4.py:ETDRK4
  -> one(u0)
      -> rotate/flip initial condition
      -> 200 micro steps per second via jax.lax.scan
      -> save integer-second frames 0..20
  -> batch(u0_batch)
      -> jax.lax.map(one, u0_batch) if solver_mode="lax-map"
      -> jax.vmap(one, ...) if solver_mode="vmap"
  -> output shape:
      x: (num_samples, 256, 256)
      y: (num_samples, 256, 256, 21)
```

## Why Burgers vmap Was Exact But 2D NS vmap Was Not

The package itself says batched operation should use `jax.vmap` around a single
state stepper. For 1D Burgers in the tested setup:

```text
vmap batch output == batch_size=1 output
```

For 2D NS, the `vmap` path changed the XLA/Fourier floating-point execution
path enough that the nonlinear 4000-step rollout amplified small differences.
That is why the project script now defaults to:

```text
solver_mode = "lax-map"
```

This keeps the per-sample execution path aligned with sequential `batch_size=1`
in the A100 tests, at the cost of much smaller speedup.

## Public vs Private API Notes

`Burgers` is public:

```python
ex.stepper.Burgers
```

because `stepper/__init__.py` imports it.

`NavierStokesVorticityZongyi` is currently private/not exported:

```python
ex.stepper._navier_stokes.NavierStokesVorticityZongyi
```

The class exists in `stepper/_navier_stokes.py`, but `stepper/__init__.py` only
exports:

```python
KolmogorovFlowVorticity
NavierStokesVorticity
```

Similarly, `VorticityConvection2dZongyi` exists in
`nonlin_fun/_vorticity_convection.py`, but `nonlin_fun/__init__.py` does not
export it.

This is not a runtime problem for the current project because the data
generation script intentionally calls the private module path.

## Installed Package File Scope

The installed package contains about 14,196 lines of Python source under:

```text
adv_robust/lib/python3.12/site-packages/exponax/
```

The solver-related source files read for this note include:

```text
exponax/__init__.py
exponax/stepper/__init__.py
exponax/stepper/_burgers.py
exponax/stepper/_navier_stokes.py
exponax/_base_stepper.py
exponax/_spectral.py
exponax/_utils.py
exponax/_repeated_stepper.py
exponax/nonlin_fun/__init__.py
exponax/nonlin_fun/_base.py
exponax/nonlin_fun/_convection.py
exponax/nonlin_fun/_vorticity_convection.py
exponax/etdrk/__init__.py
exponax/etdrk/_base_etdrk.py
exponax/etdrk/_etdrk_0.py
exponax/etdrk/_etdrk_1.py
exponax/etdrk/_etdrk_2.py
exponax/etdrk/_etdrk_3.py
exponax/etdrk/_etdrk_4.py
exponax/etdrk/_utils.py
```

Other package areas exist but are not directly on the Burgers/Zongyi-NS solver
path:

```text
exponax/ic/
exponax/metrics/
exponax/viz/
exponax/stepper/generic/
exponax/stepper/reaction/
exponax/_interpolation.py
exponax/_poisson.py
exponax/_forced_stepper.py
```

### Complete Python File Inventory

This is the installed Python source inventory under `exponax/` with line counts.
The Burgers/Zongyi-NS solver path is covered in detail above; the remaining
files are supporting initial-condition generators, metrics, visualization,
generic steppers, and reaction PDE steppers.

```text
    46 exponax/__init__.py
   269 exponax/_base_stepper.py
   108 exponax/_forced_stepper.py
   288 exponax/_interpolation.py
   127 exponax/_poisson.py
   138 exponax/_repeated_stepper.py
  1201 exponax/_spectral.py
   347 exponax/_utils.py
    22 exponax/etdrk/__init__.py
    80 exponax/etdrk/_base_etdrk.py
    34 exponax/etdrk/_etdrk_0.py
    74 exponax/etdrk/_etdrk_1.py
    92 exponax/etdrk/_etdrk_2.py
   131 exponax/etdrk/_etdrk_3.py
   137 exponax/etdrk/_etdrk_4.py
    23 exponax/etdrk/_utils.py
    39 exponax/ic/__init__.py
    72 exponax/ic/_base_ic.py
    37 exponax/ic/_clamping.py
    84 exponax/ic/_diffused_noise.py
   166 exponax/ic/_discontinuities.py
   164 exponax/ic/_gaussian_blob.py
   101 exponax/ic/_gaussian_random_field.py
    93 exponax/ic/_multi_channel.py
    41 exponax/ic/_scaled.py
   177 exponax/ic/_sine_waves_1d.py
   152 exponax/ic/_truncated_fourier_series.py
    56 exponax/metrics/__init__.py
    60 exponax/metrics/_correlation.py
   368 exponax/metrics/_derivative.py
   588 exponax/metrics/_fourier.py
   629 exponax/metrics/_spatial.py
    15 exponax/metrics/_utils.py
    47 exponax/nonlin_fun/__init__.py
   137 exponax/nonlin_fun/_base.py
   251 exponax/nonlin_fun/_convection.py
   119 exponax/nonlin_fun/_general_nonlinear.py
   101 exponax/nonlin_fun/_gradient_norm.py
    76 exponax/nonlin_fun/_polynomial.py
   252 exponax/nonlin_fun/_vorticity_convection.py
    38 exponax/nonlin_fun/_zero.py
    94 exponax/stepper/__init__.py
   104 exponax/stepper/_advection.py
   141 exponax/stepper/_advection_diffusion.py
   158 exponax/stepper/_burgers.py
   126 exponax/stepper/_diffusion.py
   126 exponax/stepper/_dispersion.py
   119 exponax/stepper/_hyper_diffusion.py
   219 exponax/stepper/_korteweg_de_vries.py
   316 exponax/stepper/_kuramoto_sivashinsky.py
   438 exponax/stepper/_navier_stokes.py
    77 exponax/stepper/generic/__init__.py
   375 exponax/stepper/generic/_convection.py
   348 exponax/stepper/generic/_gradient_norm.py
   327 exponax/stepper/generic/_linear.py
   403 exponax/stepper/generic/_nonlinear.py
   338 exponax/stepper/generic/_polynomial.py
   553 exponax/stepper/generic/_utils.py
   142 exponax/stepper/generic/_vorticity_convection.py
    28 exponax/stepper/reaction/__init__.py
   128 exponax/stepper/reaction/_allen_cahn.py
   102 exponax/stepper/reaction/_belousov_zhabotinsky.py
   158 exponax/stepper/reaction/_cahn_hilliard.py
   129 exponax/stepper/reaction/_fisher_kpp.py
   179 exponax/stepper/reaction/_gray_scott.py
   128 exponax/stepper/reaction/_swift_hohenberg.py
    74 exponax/viz/__init__.py
   359 exponax/viz/_animate.py
   450 exponax/viz/_animate_facet.py
   438 exponax/viz/_plot.py
   493 exponax/viz/_plot_facet.py
   146 exponax/viz/_volume.py
 14196 total
```

## Quick Summary

1D Burgers:

```text
Burgers
  L = nu * Laplacian
  N = ConvectionNonlinearFun
  time integrator = ETDRK4
  project repeats 1000 dt=0.001 steps and saves final state
```

2D Zongyi Navier-Stokes:

```text
NavierStokesVorticityZongyi
  L = nu * Laplacian + drag * Identity
  N = VorticityConvection2dZongyi
    = - velocity dot grad(vorticity) + build_forcing(...)
  time integrator = ETDRK4
  project performs 4000 dt=0.005 micro-steps and saves frames 0..20
```
