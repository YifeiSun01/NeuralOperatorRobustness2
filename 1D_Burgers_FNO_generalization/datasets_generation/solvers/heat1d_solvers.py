import numpy as np
import exponax as ex
import phi.flow as phiflow
import phiml.math as math
from scipy.integrate import solve_ivp
import os

class SciPyHeatSolver1D:
    """使用SciPy求解一维热传导方程"""
    def __init__(self, nx, alpha, bc='periodic', num_int_method="RK45"):
        self.nx = nx
        self.x = np.linspace(-1, 1, nx)
        self.dx = 2/nx
        self.alpha = alpha  # 热扩散系数
        self.bc = bc
        self.num_int_method = num_int_method
    
    def laplacian(self, u):
        """计算二阶导数"""
        return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / self.dx**2
    
    def dudt(self, t, u):
        """热传导方程的右边项"""
        return self.alpha * self.laplacian(u)

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions

    def solve(self, u0, t_final, t_eval, max_step):
        """求解方程"""
        sol = solve_ivp(
            self.dudt,
            [0, t_final],
            u0,
            t_eval=t_eval,
            method=self.num_int_method,
            max_step=max_step,
            rtol=1e-6,
            atol=1e-8
        )
        sol_dict = {t: sol.y[:,i] for i,t in enumerate(sol.t)}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        return sol_dict

class SciPySpectralHeatSolver1D:
    """使用谱方法求解热传导方程"""
    def __init__(self, nx, alpha, L=2, bc='periodic', num_int_method="RK45"):
        self.nx = nx
        self.L = L
        self.alpha = alpha
        self.k = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    
    def compute_rhs(self, t, u_hat):
        """谱空间中的方程右边项"""
        return -self.alpha * (self.k**2) * u_hat

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions
    
    def solve(self, u0, t_final, t_eval, max_step):
        u0_hat = np.fft.fft(u0)
        sol = solve_ivp(
            self.compute_rhs,
            [0, t_final],
            u0_hat,
            t_eval=t_eval,
            method='RK45',
            max_step=max_step
        )
        sol_dict = {t: np.fft.ifft(sol.y[:,i]).real for i,t in enumerate(sol.t)}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        return sol_dict

class ExponaxHeatSolver1D:
    """使用Exponax库求解热传导方程"""
    def __init__(self, nx, alpha, bc='periodic'):
        self.nx = nx
        self.alpha = alpha

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions
    
    def solve(self, u0, t_final, t_eval, step):
        steps = int(t_final/step)
        full_ic = np.expand_dims(u0, axis=1).T
        diffusion_stepper = ex.stepper.Diffusion(
            1, 2, self.nx, step, 
            diffusivity=self.alpha
        )
        longer_rollout_advection_stepper = ex.rollout(
            diffusion_stepper, steps, include_init=True
        )
        longer_trajectory = longer_rollout_advection_stepper(full_ic)
        # print("Exponax shape: ", longer_trajectory.shape)
        sol_dict = {i*step:longer_trajectory[i, 0, :].T for i in range(longer_trajectory.shape[0])}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        # print("Exponax time keys: ", sol_dict.keys())
        return sol_dict

class PhiFlowHeatSolver1D:
    """使用PhiFlow求解热传导方程"""
    def __init__(self, nx, alpha, bc='periodic'):
        self.nx = nx
        self.alpha = alpha
        self.bc = bc

    def get_closest_solutions(self, sol_dict, t_eval):
        closest_solutions = {}
        sorted_keys = sorted(sol_dict.keys())  # Ensure keys are sorted

        for t in t_eval:
            closest_key = min(sorted_keys, key=lambda k: abs(k - t))  # Find closest key
            closest_solutions[t] = (closest_key,sol_dict[closest_key])  # Store closest solution

        return closest_solutions
    
    def solve(self, u0, t_final, t_eval, step):
        u0 = math.tensor(u0, phiflow.spatial('x'))
        if self.bc == "periodic":
            u0 = phiflow.CenteredGrid(u0, phiflow.extrapolation.PERIODIC, 
                                    x=self.nx, bounds=phiflow.Box(x=(-1,1)))
        else:
            u0 = phiflow.CenteredGrid(u0, x=self.nx, bounds=phiflow.Box(x=(-1,1)))
        
        steps = int(t_final/step)
        solutions = [u0]
        for _ in range(steps):
            next_sol = phiflow.diffuse.implicit(solutions[-1], self.alpha, step)
            solutions.append(next_sol)
        solutions = [v.values.numpy('x,vector') for v in solutions]
        sol_dict = {i*step:solution for i,solution in enumerate(solutions)}
        sol_dict = self.get_closest_solutions(sol_dict, t_eval)
        # print("PhiFlow time keys: ", sol_dict.keys())
        return sol_dict