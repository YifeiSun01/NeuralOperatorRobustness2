# import numpy as np
from jax import numpy as jnp
import exponax as ex
import phi.flow as phiflow
import phiml.math as math
from scipy.integrate import solve_ivp
import os
import jax
import numpy as np
import torch

class SciPyBurgersSolver1D:
    def __init__(self, nx, nu, bc='periodic', num_int_method="RK45",conservative=False, xlim=(-1,1)):
        self.nx = nx
        self.x = np.linspace(xlim[0], xlim[1], nx)
        self.L = xlim[1]-xlim[0]
        self.dx = self.L/nx
        self.nu = nu
        self.bc = bc
        self.num_int_method = num_int_method
        self.conservative = conservative
    
    def compute_derivatives(self, u):
        d2u_dx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / self.dx**2
        if self.conservative == True:
            u_squared = u**2 / 2
            convective_term = (np.roll(u_squared, -1) - np.roll(u_squared, 1)) / (2 * self.dx)
            return convective_term, d2u_dx2
        else:
            du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * self.dx)
            return du_dx, d2u_dx2
    
    def dudt(self, t, u):
        if self.conservative == True:
            convective_term, d2u_dx2 = self.compute_derivatives(u)
            return -convective_term + self.nu * d2u_dx2
        else:
            du_dx, d2u_dx2 = self.compute_derivatives(u)
            return -u * du_dx + self.nu * d2u_dx2

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions
    
    # @jax.jit 
    def solve(self, u0, t_final, t_eval, max_step):
        current_u = u0.copy()
        sol = solve_ivp(
            self.dudt, 
            [0, t_final],  # Integrate over one time step
            current_u,
            t_eval=t_eval,
            method=self.num_int_method, 
            max_step=max_step,
            rtol=1e-6, 
            atol=1e-8
        )
        # print(t_eval)
        # print("SciPy shape: ", sol.y.shape)
        # print(sol.y)
        sol_dict = {t:sol.y[...,i] for i,t in enumerate(t_eval)}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        return sol_dict

class SciPySpectralBurgersSolver1D:
    def __init__(self, nx, nu, bc='periodic', num_int_method="RK45", xlim=(-1,1)):
        """
        Initialize the spectral Burgers equation solver.
        
        Parameters:
            nx (int): Number of spatial grid points
            nu (float): Viscosity coefficient
            L (float): Domain length (default: 2π)
            bc (str): Boundary condition type (only 'periodic' supported for spectral)
            num_int_method (str): Numerical integration method (e.g., "RK45")
        """
        self.x = np.linspace(xlim[0], xlim[1], nx)
        self.L = xlim[1]-xlim[0]
        self.dx = self.L/nx
        self.nu = nu
        self.bc = bc
        self.num_int_method = num_int_method
        
        # Wave numbers for Fourier transform
        self.k = 2 * np.pi * np.fft.fftfreq(nx, d=self.L/nx)
        
    def compute_rhs(self, t, u_hat):
        """
        Compute the right-hand side of the Burgers equation in spectral space.
        """
        # Transform to physical space for nonlinear term
        u = np.fft.ifft(u_hat).real
        ux = np.fft.ifft(1j * self.k * u_hat).real
        
        # Compute nonlinear term in spectral space
        nonlinear = -0.5 * np.fft.fft(u * ux)
        
        # Viscous term remains in spectral space
        viscous = -self.nu * (self.k**2) * u_hat
        
        return nonlinear + viscous
    
    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions
    
    # @jax.jit 
    def solve(self, u0, t_final, t_eval=None, max_step=None):
        """
        Solve the Burgers equation with given initial condition.
        
        Parameters:
            u0 (array): Initial condition in physical space
            t_final (float): Final time
            t_eval (array): Times at which to store the solution
            max_step (float): Maximum time step size
            
        Returns:
            dict: Dictionary of solutions at requested times
        """
        # Transform initial condition to spectral space
        u0_hat = np.fft.fft(u0)
        
        # Solve the ODE system in spectral space
        sol = solve_ivp(
            self.compute_rhs,
            [0, t_final],
            u0_hat,
            t_eval=t_eval,
            method=self.num_int_method,
            max_step=max_step,
            rtol=1e-6,
            atol=1e-8
        )
        
        # Transform solutions back to physical space
        sol_dict = {}
        for i, t in enumerate(sol.t):
            sol_dict[t] = np.fft.ifft(sol.y[:, i]).real
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        return sol_dict

"""
class ExponaxBurgersSolver1D:
    def __init__(self, nx, nu, bc='periodic',conservative=False, xlim=(-1,1)):
        self.nx = nx
        # self.x = np.linspace(xlim[0], xlim[1], nx)
        self.x = jnp.linspace(xlim[0], xlim[1], nx)
        self.L = xlim[1]-xlim[0]
        self.dx = self.L/nx
        self.nu = nu
        self.bc = bc
        self.conservative = conservative

    def get_closest_solutions(self, sol_dict, t_eval):
       closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions
    
    # @jax.jit 
    def solve(self, u0, t_final, t_eval, step):
        steps = int(t_final/step)
        # full_ic = np.expand_dims(u0, axis=1).T
        full_ic = jnp.expand_dims(u0.squeeze(),axis=0).astype(jnp.float64)
        # print("full_ic.shape: ",full_ic.shape)
        slower_burgers_stepper = ex.stepper.Burgers(
            1, self.L, self.nx, step, 
            diffusivity=self.nu,
            convection_scale=1.0,
            order=4,
            conservative=self.conservative
        )
        longer_rollout_advection_stepper = ex.rollout(
            slower_burgers_stepper, steps, include_init=True
        )
        longer_trajectory = longer_rollout_advection_stepper(full_ic)
        # print("Exponax shape: ", longer_trajectory.shape)
        sol_dict = {i*step:longer_trajectory[i, 0, :].T for i in range(longer_trajectory.shape[0])}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        # print("Exponax time keys: ", sol_dict.keys())
        return sol_dict
""" 

import jax
import jax.numpy as jnp
from jax import device_put

class ExponaxBurgersSolver1D:
    def __init__(self, nx, nu, bc='periodic', conservative=False, xlim=(-1,1)):
        # 打印当前JAX使用的后端设备
        print(f"JAX正在使用的后端: {jax.default_backend()}")
        print(f"可用的JAX设备: {jax.devices()}")
        
        self.nx = nx
        self.x = device_put(jnp.linspace(xlim[0], xlim[1], nx))  # 将数组放到设备上
        self.L = xlim[1]-xlim[0]
        self.dx = self.L/nx
        self.nu = nu
        self.bc = bc
        self.conservative = conservative
        
        # 初始化stepper时也确保在GPU上
        self.slower_burgers_stepper = ex.stepper.Burgers(
            1, self.L, self.nx, 0.001,  # step会在solve方法中被覆盖
            diffusivity=self.nu,
            convection_scale=1.0,
            order=4,
            conservative=self.conservative
        )

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))
            closest_solutions[t] = (closest_key, sol_dict[closest_key])

        return closest_solutions

    def solve(self, u0, t_final, t_eval, step):
        # 确保输入数据在GPU上
        if isinstance(u0, np.ndarray):
            u0 = device_put(jnp.array(u0))
        elif isinstance(u0, torch.Tensor):
            u0 = device_put(jnp.array(u0.cpu().numpy()))
        
        steps = int(t_final/step)
        full_ic = jnp.expand_dims(u0.squeeze(), axis=0).astype(jnp.float64)
        
        # 使用预初始化的stepper但更新步长
        slower_burgers_stepper = ex.stepper.Burgers(
            1, self.L, self.nx, step, 
            diffusivity=self.nu,
            convection_scale=1.0,
            order=4,
            conservative=self.conservative
        )
        
        # 使用JIT编译加速
        @jax.jit
        def rollout_fn(ic):
            return ex.rollout(slower_burgers_stepper, steps, include_init=True)(ic)
        
        longer_trajectory = rollout_fn(full_ic)
        
        sol_dict = {i*step: longer_trajectory[i, 0, :].T for i in range(longer_trajectory.shape[0])}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        
        # 将结果转换为numpy数组以便与其他代码兼容
        return {k: (t, jnp.array(v)) for k, (t, v) in sol_dict.items()}

class PhiFlowBurgersSolver1D:
    def __init__(self, nx, nu, bc='periodic', xlim=(-1,1)):
        self.nx = nx
        self.x = np.linspace(xlim[0], xlim[1], nx)
        self.L = xlim[1]-xlim[0]
        self.dx = self.L/nx
        self.nu = nu
        self.bc = bc
        self.xlim = xlim

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions
    
    # @jax.jit 
    def solve(self, u0, t_final, t_eval, step):
        u0 = math.tensor(u0, phiflow.spatial('x'))
        if self.bc == "periodic":
            u0 = phiflow.CenteredGrid(u0, phiflow.extrapolation.PERIODIC, x=self.nx, bounds=phiflow.Box(x=self.xlim))
        else:
            u0 = phiflow.CenteredGrid(u0, x=self.nx, bounds=phiflow.Box(x=self.xlim))
        steps = int(t_final/step)
        solutions = [u0] 
        for _ in range(steps):
            v1 = phiflow.diffuse.implicit(solutions[-1], self.nu, step)
            # v2 = phiflow.advect.semi_lagrangian(v1, v1, step)
            v2 = phiflow.advect.rk4(v1, v1, step)
            solutions.append(v2)
        # print("PhiFlow time length: ", len(solutions))
        solutions = [v.values.numpy('x,vector') for v in solutions]
        sol_dict = {i*step:solution for i,solution in enumerate(solutions)}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        # print("PhiFlow time keys: ", sol_dict.keys())
        return sol_dict

