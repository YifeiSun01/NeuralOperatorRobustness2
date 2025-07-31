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

def generate_vorticity_transport_data(time_list, grf, nu, ntimepoints):
    # Initialize solver
    exponaxsolver = ExponaxVTSolver2D(
        nx=grf.shape[0], ny=grf.shape[1], 
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

    selected_indices = np.linspace(0, len(time_list)-1, ntimepoints, dtype=int)
    selected_times = time_list[selected_indices]
    
    # Convert results to PyTorch tensors on GPU
    x = torch.from_numpy(exponax_solutions[0][1]).float().to(device)
    y_numpy = np.array([exponax_solutions[selected_time][1] for selected_time in selected_times])
    y = torch.from_numpy(y_numpy).float().to(device)
    
    return x, y, selected_times

class DatasetGenerator:
    def __init__(self, base_seed=0):
        self.dataset_counter = 0
    
    def _generate_filename(self, params):
        return (f"dim{params['dim']}d_nx{params['nx']}_N{params['nsamples']}_"
                f"solver=exponax_"
                f"nu{params['nu_ns']:.3f}_"
                f"t{params['tfinal']:.1f}_"
                f"all_frames.pt")
    
    def generate_dataset(self, nu_ns, tfinal, nsamples, save_dir=None):
        dataset_x = []
        dataset_y = []
        zongyi_dataset = torch.load('../datasets/2D/NS/NS_data_zongyi_test_all_frame.pt', weights_only=False)

        # Add tqdm progress bar for sample generation
        for _ in tqdm(range(nsamples), desc="Generating samples", unit="sample"):
            
            # Generate on GPU
            grf = 
            
            time_list = np.linspace(0, tfinal, 5001)
            ntimepoints = tfinal + 1
            x, y, selected_times = generate_vorticity_transport_data(
                time_list,
                np.array(grf),
                nu_ns,
                ntimepoints
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
                'nx': grf.shape[0],
                'nsamples': nsamples,
                'nu_ns': nu_ns,
                'tfinal': tfinal,
            }
            
            filename = self._generate_filename(file_params)
            save_path = os.path.join(save_dir, filename)
            
            # Add progress bar for saving
            with tqdm(total=1, desc="Saving dataset") as pbar:
                torch.save({
                    'u0': tensor_x.cpu(),
                    'u': tensor_y.cpu(),
                    'metadata': {
                        'nu_ns': nu_ns,
                        'tfinal': tfinal,
                        'nsamples': nsamples,
                        'seed_range': (start_seed, self.global_seed_counter-1),
                        "times": selected_times
                    }
                }, save_path)
                pbar.update(1)
            
            print(f"\nDataset saved to {save_path}")
            return save_path
        else:
            return tensor_x, tensor_y

if __name__ == "__main__":
    # Usage Example
    generator = DatasetGenerator(base_seed=0)

    print("Generating first dataset:")
    path1 = generator.generate_dataset(
        nu_ns=0.01,
        tfinal=20,
        nsamples=5,
        save_dir="./datasets"
    )




