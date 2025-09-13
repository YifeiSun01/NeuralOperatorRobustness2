# ====== 修复的高性能PDE Solver实现 ======

from datetime import datetime
import os
import sys
import time
import gc
import warnings
import subprocess
from pathlib import Path
from pprint import pprint

import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
import jaxlib

from jax.extend import backend as jax_backend

# ====================== 优化的JAX<->PyTorch桥（稳定缓存） ======================
class OptimizedJaxPDEWrapper(torch.autograd.Function):
    # 使用模块级缓存，确保跨实例共享
    _vjp_cache = {}
    _compilation_stats = {'hits': 0, 'misses': 0}

    @staticmethod
    def forward(ctx, a_torch: torch.Tensor, g, cache_key):
        if not a_torch.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        
        # 确保输入连续性
        if not a_torch.is_contiguous():
            a_torch = a_torch.contiguous()
            
        # PyTorch → JAX (DLPack 零拷贝)
        a_dlpack = torch.utils.dlpack.to_dlpack(a_torch)
        a_jax = jax.dlpack.from_dlpack(a_dlpack)
        g_output_jax = g(a_jax)
        out_dlpack = jax.dlpack.to_dlpack(g_output_jax)
        g_output = torch.utils.dlpack.from_dlpack(out_dlpack)
        
        ctx.save_for_backward(a_torch)
        ctx.g = g
        ctx.cache_key = cache_key
        return g_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (a_torch,) = ctx.saved_tensors
        g = ctx.g
        cache_key = ctx.cache_key

        # 使用稳定的缓存键
        vjp_key = f"vjp_{cache_key}"
        if vjp_key not in OptimizedJaxPDEWrapper._vjp_cache:
            def jax_vjp(a_jax, grad_jax):
                _, vjp_fn = jax.vjp(g, a_jax)
                return vjp_fn(grad_jax)
            OptimizedJaxPDEWrapper._vjp_cache[vjp_key] = jax.jit(jax_vjp)
            OptimizedJaxPDEWrapper._compilation_stats['misses'] += 1
        else:
            OptimizedJaxPDEWrapper._compilation_stats['hits'] += 1
            
        jitted_vjp = OptimizedJaxPDEWrapper._vjp_cache[vjp_key]

        # 确保梯度连续性
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if not a_torch.is_contiguous():
            a_torch = a_torch.contiguous()

        # PyTorch → JAX 转换
        grad_dlpack = torch.utils.dlpack.to_dlpack(grad_output)
        grad_jax = jax.dlpack.from_dlpack(grad_dlpack)
        a_dlpack = torch.utils.dlpack.to_dlpack(a_torch)
        a_jax = jax.dlpack.from_dlpack(a_dlpack)

        grad_input_jax, = jitted_vjp(a_jax, grad_jax)
        grad_input_dlpack = jax.dlpack.to_dlpack(grad_input_jax)
        return torch.utils.dlpack.from_dlpack(grad_input_dlpack), None, None

    @classmethod
    def print_stats(cls):
        total = cls._compilation_stats['hits'] + cls._compilation_stats['misses']
        if total > 0:
            hit_rate = cls._compilation_stats['hits'] / total * 100
            print(f"JAX编译缓存统计: 命中率 {hit_rate:.1f}% ({cls._compilation_stats['hits']}/{total})")

# ====================== PDE 求解（保持数值精度） ======================
def generate_sequence_every_second(u0: jnp.ndarray, nu: float, T_seconds: int, fixed_step: float = 0.005):
    """单样本版本：完全不变，保持数值一致性"""
    u0_proc = jnp.rot90(jnp.flip(u0, axis=-2), 3, axes=(-2, -1))
    H, W = u0_proc.shape
    assert H == W, "目前只支持正方域网格"

    import exponax as ex
    stepper = ex.stepper._navier_stokes.NavierStokesVorticityZongyi(
        2, 1, H, fixed_step, diffusivity=nu, order=4
    )

    steps_per_sec = int(round(1.0 / fixed_step))

    def step_once(u):
        return stepper(u[None, ...])[0]

    step_once = jax.checkpoint(step_once)

    def micro(carry, _):
        u = step_once(carry)
        return u, None

    def run_one_second(carry, _):
        u, _ = jax.lax.scan(micro, carry, None, length=steps_per_sec)
        return u, u

    _, seconds = jax.lax.scan(run_one_second, u0_proc, None, length=T_seconds)
    seq = jnp.concatenate([u0_proc[None, ...], seconds], axis=0)
    seq = jnp.swapaxes(seq, -1, -2)
    return seq

def _batched_generate_sequence(u0_batched: jnp.ndarray, nu: float, T_seconds: int, fixed_step: float):
    """批处理版本"""
    fn = lambda a: generate_sequence_every_second(a, nu, T_seconds, fixed_step)
    return jax.vmap(fn, in_axes=0, out_axes=1)(u0_batched)

# ====================== 高性能PDE求解器 ======================
class HighPerformancePDESolver:
    """高性能PDE求解器：解决缓存失效问题"""
    
    def __init__(self, nu: float, device="cuda", *, fixed_step: float = 0.005):
        self.nu = nu
        self.device = device
        self.fixed_step = fixed_step
        
        # 预编译的函数存储：使用确定性键
        self._compiled_functions = {}
        self._warmup_complete = False
        
        # 禁用内置性能测量
        self.measure_performance = False
        
        print(f"初始化PDE求解器: nu={nu}, device={device}")
        
    def _get_cache_key(self, shape: tuple, T_seconds: int) -> str:
        """生成稳定的缓存键"""
        return f"pde_{len(shape)}d_{'x'.join(map(str, shape))}_{T_seconds}_{self.nu}_{self.fixed_step}"
        
    def _get_or_compile_function(self, input_shape: tuple, T_seconds: int):
        """获取或编译JAX函数"""
        cache_key = self._get_cache_key(input_shape, T_seconds)
        
        if cache_key in self._compiled_functions:
            return self._compiled_functions[cache_key], cache_key
            
        # 编译新函数
        ndim = len(input_shape)
        
        if ndim == 2:  # 单样本 (H, W)
            def pde_func(a):
                return generate_sequence_every_second(a, self.nu, T_seconds, self.fixed_step)
        elif ndim == 3:  # 批处理 (B, H, W)
            def pde_func(A):
                return _batched_generate_sequence(A, self.nu, T_seconds, self.fixed_step)
        else:
            raise ValueError(f"不支持的维度: {ndim}")
        
        # JIT编译
        compiled_func = jax.jit(pde_func)
        self._compiled_functions[cache_key] = compiled_func
        
        if not self._warmup_complete:
            print(f"编译新函数: {cache_key}")
        
        return compiled_func, cache_key
    
    def warmup(self, common_shapes=None, common_times=None):
        """预热编译缓存"""
        if common_shapes is None:
            common_shapes = [(256, 256), (1, 256, 256), (2, 256, 256), (4, 256, 256)]
        if common_times is None:
            common_times = [9, 19, 29]  # 常用的时间长度
            
        print("开始预热PDE编译缓存...")
        device = self.device
        
        total_compilations = len(common_shapes) * len(common_times)
        compiled = 0
        
        for shape in common_shapes:
            dummy = torch.zeros(shape, device=device, dtype=torch.float32)
            for T in common_times:
                # 触发编译
                func, cache_key = self._get_or_compile_function(shape, T)
                
                # 执行一次确保编译完成
                with torch.no_grad():
                    _ = self.rollout_seconds(dummy, T)
                    
                compiled += 1
                print(f"预热进度: {compiled}/{total_compilations} - {cache_key}")
        
        self._warmup_complete = True
        print(f"预热完成! 已编译 {len(self._compiled_functions)} 个函数")
        OptimizedJaxPDEWrapper.print_stats()
    
    def rollout_seconds(self, x0: torch.Tensor, T_seconds: int):
        """主要接口：高性能rollout"""
        assert isinstance(x0, torch.Tensor) and x0.is_cuda
        
        # 确保输入连续性
        if not x0.is_contiguous():
            x0 = x0.contiguous()
        
        # 获取编译函数
        func, cache_key = self._get_or_compile_function(x0.shape, T_seconds)
        
        # 执行计算
        start_time = time.time() if not self._warmup_complete else None
        
        result = OptimizedJaxPDEWrapper.apply(x0, func, cache_key)
        
        if start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > 2.0:
                print(f"⚠️  rollout_seconds took {elapsed:.2f}s for {cache_key}")
        
        return result
    
    def get_stats(self):
        """获取性能统计"""
        return {
            'compiled_functions': len(self._compiled_functions),
            'function_keys': list(self._compiled_functions.keys()),
            'warmup_complete': self._warmup_complete
        }

# ====================== 兼容性接口 ======================
# 为了不破坏现有代码，提供兼容接口
class DifferentiablePDESolver(HighPerformancePDESolver):
    """兼容性包装器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 保持原有的接口
        self.perf_records = {}

# ====================== 使用示例 ======================
def test_performance():
    """性能测试"""
    print("=" * 50)
    print("PDE求解器性能测试")
    print("=" * 50)
    
    device = torch.device('cuda')
    solver = HighPerformancePDESolver(nu=1e-5, device=device)
    
    # 预热
    solver.warmup()
    
    # 性能测试
    batch_sizes = [1, 2, 4]
    times = [9, 19]
    
    print("\n性能测试结果:")
    for bs in batch_sizes:
        for T in times:
            test_input = torch.randn(bs, 256, 256, device=device)
            
            # 预热一次
            _ = solver.rollout_seconds(test_input, T)
            
            # 测试多次取平均
            torch.cuda.synchronize()
            start = time.time()
            
            n_runs = 5
            for _ in range(n_runs):
                _ = solver.rollout_seconds(test_input, T)
                
            torch.cuda.synchronize()
            avg_time = (time.time() - start) / n_runs
            
            print(f"  batch_size={bs}, T={T}: {avg_time:.3f}s")
    
    OptimizedJaxPDEWrapper.print_stats()
    print(f"\n编译函数统计: {solver.get_stats()}")

if __name__ == "__main__":
    # 禁用所有不必要的测量
    os.environ["MEASURE_PDE"] = "0"
    test_performance()