import numpy as np
from scipy.ndimage import gaussian_filter
import jax
import jax.numpy as jnp
import exponax as ex
import torch
import os
from tqdm import tqdm
import time

# Set default device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRFGenerator:
    @staticmethod
    def generate_grf(shape, kernel='gaussian', kernel_params=None, bc='periodic', seed=None, zero_mean=False):
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(0)
        
        # Generate random field on GPU using JAX
        with jax.default_device(jax.devices('gpu')[0] if jax.devices('gpu') else jax.default_device(jax.devices('cpu')[0])):
            white_noise = jax.random.normal(key, shape=shape)
            
            if kernel == 'gaussian':
                cl = kernel_params.get('correlation_length', 1.0)*shape[0]
                if bc == 'periodic':
                    mode = 'wrap'
                else:
                    mode = 'nearest'
                grf = gaussian_filter(white_noise, sigma=cl, mode=mode)
            elif kernel == 'matern':
                if bc != 'periodic':
                    raise NotImplementedError("Matern kernel currently only supports periodic BCs")
                cl = kernel_params.get('correlation_length', 1.0)*shape[0]
                nu = kernel_params.get('nu', 1.5)
                if len(shape) == 1:
                    grf = GRFGenerator._matern_1d(shape, cl, nu, seed)
                elif len(shape) == 2:
                    grf = GRFGenerator._matern_2d(shape, cl, nu, seed)
                else:
                    raise ValueError("Only 1D or 2D GRFs supported")
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            grf = (grf - grf.min()) / (grf.max() - grf.min())
            if zero_mean:
                grf -= grf.mean()
            
            return grf
    
    @staticmethod
    def _matern_1d(shape, cl, nu, seed):
        key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)
        with jax.default_device(jax.devices('gpu')[0] if jax.devices('gpu') else jax.default_device(jax.devices('cpu')[0])):
            k = jnp.fft.fftfreq(shape[0])
            scaling = (2 * jnp.pi) ** 2 / (cl ** 2)
            k_sq = (2 * jnp.pi * k) ** 2
            sk = (scaling + k_sq) ** (-nu - 0.5)
            noise = (jax.random.normal(key, shape=shape) + 1j * jax.random.normal(key+1, shape=shape)) * jnp.sqrt(sk)
            grf = jnp.fft.ifft(noise).real
            return grf
    
    @staticmethod
    def _matern_2d(shape, cl, nu, seed):
        key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)
        with jax.default_device(jax.devices('gpu')[0] if jax.devices('gpu') else jax.default_device(jax.devices('cpu')[0])):
            nx, ny = shape
            kx = jnp.fft.fftfreq(nx)
            ky = jnp.fft.fftfreq(ny)
            kxx, kyy = jnp.meshgrid(kx, ky, indexing='ij')
            k_sq = (2 * jnp.pi * kxx) ** 2 + (2 * jnp.pi * kyy) ** 2
            scaling = (2 * jnp.pi) ** 2 / (cl ** 2)
            sk = (scaling + k_sq) ** (-nu - 1)
            noise = (jax.random.normal(key, shape=shape) + 1j * jax.random.normal(key+1, shape=shape)) * jnp.sqrt(sk)
            grf = jnp.fft.ifft2(noise).real
            return grf

class ExponaxVTSolver2D:
    def __init__(self, nx, ny, Lx, Ly, nu, forcing=None, bc='periodic'):
        self.nx = nx
        self.Lx = Lx
        self.x = torch.linspace(-Lx/2, Lx/2, nx, device=device)
        self.dx = 2/nx
        self.nu = nu
        self.bc = bc

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))
            closest_solutions[t] = (closest_key, sol_dict[closest_key])
        
        return closest_solutions

    def solve(self, u0, t_final, t_eval, step):
        # Move initial condition to GPU
        full_ic = jnp.expand_dims(jnp.array(u0, dtype=jnp.float32), axis=0)
        
        # Initialize solver on GPU
        slower_NS_stepper = ex.stepper.NavierStokesVorticity(
            2, self.Lx, self.nx, step, 
            diffusivity=self.nu
        )
        
        longer_rollout_NS_stepper = ex.rollout(
            slower_NS_stepper, int(t_final/step), include_init=True
        )

        # Run simulation on GPU
        longer_trajectory = longer_rollout_NS_stepper(full_ic)
        
        # Convert results back to numpy for compatibility
        sol_dict = {i*step: np.array(longer_trajectory[i, 0, ...].T.block_until_ready())
            for i in range(longer_trajectory.shape[0])}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        
        return sol_dict

def generate_vorticity_transport_data(time_list, grf_params, grf, nu):
    # Initialize solver
    exponaxsolver = ExponaxVTSolver2D(
        nx=grf_params["nx"], ny=grf_params["nx"], 
        Lx=128, Ly=128, nu=nu
    )
    
    # Convert input to numpy (handling both JAX and numpy arrays)
    if hasattr(grf, '__jax_array__') or isinstance(grf, jnp.ndarray):  # JAX array
        grf_numpy = np.array(grf)  # Convert JAX to numpy
    elif isinstance(grf, np.ndarray):  # Already numpy
        grf_numpy = grf
    elif isinstance(grf, torch.Tensor):  # PyTorch tensor
        grf_numpy = grf.cpu().numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(grf)}")
    
    # Get solutions
    exponax_solutions = exponaxsolver.solve(
        grf_numpy,  # Use converted numpy array
        t_final=time_list[-1],
        t_eval=[0] + time_list,
        step=0.01
    )
    
    # Convert results to PyTorch tensors on GPU
    x = torch.from_numpy(exponax_solutions[0][1]).float().to(device)
    y = torch.from_numpy(exponax_solutions[time_list[-1]][1]).float().to(device)
    
    return x, y

class DatasetGenerator:
    def __init__(self, base_seed=0):
        self.global_seed_counter = base_seed
        self.dataset_counter = 0
    
    def _generate_filename(self, params):
        return (f"dim{params['dim']}d_nx{params['nx']}_N{params['nsamples']}_"
                f"solver=exponax_kernel={params['kernel']}_"
                f"correlation_length{params['cl']:.2f}_"
                f"bc{params['bc']}_nu{params['nu_ns']:.3f}_"
                f"t{params['tfinal']:.1f}_"
                f"seed{params['seed']}.pt")
    
    def generate_dataset(self, cl, nu_grf, nu_ns, tfinal, nsamples, save_dir=None):
        dataset_x = []
        dataset_y = []
        start_seed = self.global_seed_counter
        
        # Add tqdm progress bar for sample generation
        for _ in tqdm(range(nsamples), desc="Generating samples", unit="sample"):
            grf_params = {
                'dim': 2,
                'nx': 256,
                'kernel': 'matern',
                'kernel_params': {'correlation_length': cl, "nu": nu_grf},
                'bc': 'periodic',
                'seed': self.global_seed_counter,
                'zero_mean': False
            }
            
            # Generate on GPU
            grf = GRFGenerator.generate_grf(
                shape=(grf_params["nx"], grf_params["nx"]),
                kernel=grf_params['kernel'],
                kernel_params=grf_params['kernel_params'],
                bc=grf_params['bc'],
                seed=int(grf_params['seed']),
                zero_mean=grf_params['zero_mean']
            )
            
            time_list = np.linspace(0, tfinal, 5001)
            x, y = generate_vorticity_transport_data(
                time_list,
                grf_params,
                np.array(grf),
                nu_ns,
            )
            
            dataset_x.append(x)
            dataset_y.append(y)
            self.global_seed_counter += 1
        
        # Stack on GPU with progress indication
        with tqdm(total=2, desc="Stacking tensors") as pbar:
            tensor_x = torch.stack(dataset_x).to(device)
            pbar.update(1)
            tensor_y = torch.stack(dataset_y).to(device)
            pbar.update(1)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_params = {
                'dim': 2,
                'nx': 256,
                'nsamples': nsamples,
                'kernel': 'matern',
                'cl': cl,
                'bc': 'periodic',
                'nu_ns': nu_ns,
                'tfinal': tfinal,
                'seed': start_seed
            }
            
            filename = self._generate_filename(file_params)
            save_path = os.path.join(save_dir, filename)
            
            # Add progress bar for saving
            with tqdm(total=1, desc="Saving dataset") as pbar:
                torch.save({
                    'x': tensor_x.cpu(),
                    'y': tensor_y.cpu(),
                    'metadata': {
                        'cl': cl,
                        'nu_grf': nu_grf,
                        'nu_ns': nu_ns,
                        'tfinal': tfinal,
                        'nsamples': nsamples,
                        'seed_range': (start_seed, self.global_seed_counter-1)
                    }
                }, save_path)
                pbar.update(1)
            
            print(f"\nDataset saved to {save_path}")
            return save_path
        else:
            return tensor_x, tensor_y

if __name__ == "__main__":
    # Usage Example
    generator = DatasetGenerator(base_seed=61000)

    # First generation with progress bars
    print("Generating first dataset:")
    path1 = generator.generate_dataset(
        cl=0.5,
        nu_grf=1.5,
        nu_ns=0.01,
        tfinal=40,
        nsamples=200,
        save_dir="./datasets"
    )



