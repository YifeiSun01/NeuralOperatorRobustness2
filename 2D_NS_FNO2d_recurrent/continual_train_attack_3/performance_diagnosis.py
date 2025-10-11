# ====== 性能问题诊断与解决方案 ======

"""
问题根源分析：
1. JAX函数重复编译：每次调用rollout_seconds都可能触发重新编译
2. 缓存键不稳定：函数对象ID变化导致缓存失效
3. 内存布局不一致：DLPack转换时的内存布局差异
4. 过度的CUDA同步：性能测量导致不必要的同步等待
"""

import os
import time
import torch
import jax
from solver import DifferentiablePDESolver
from PGD_attack_batch import PGDAdamConfig, attack_and_rollout_seq

def diagnose_performance_bottleneck(batch_size=2, size=256):
    """诊断性能瓶颈的具体位置"""
    
    print("=" * 60)
    print("性能诊断开始...")
    print("=" * 60)
    
    device = torch.device('cuda')
    pde_solver = DifferentiablePDESolver(nu=1e-5, device=device)
    
    # 测试数据
    dummy_batch = torch.randn(batch_size, size, size, device=device)
    dummy_seq = torch.randn(batch_size, size, size, 10, device=device)
    
    # 1. 测试单次PDE rollout时间
    print("1. 测试单次PDE rollout...")
    torch.cuda.synchronize()
    start = time.time()
    
    # 第一次调用（包含编译时间）
    result1 = pde_solver.rollout_seconds(dummy_batch, 19)
    torch.cuda.synchronize()
    first_call_time = time.time() - start
    
    # 第二次调用（应该很快，如果缓存有效）
    torch.cuda.synchronize()
    start = time.time()
    result2 = pde_solver.rollout_seconds(dummy_batch, 19)
    torch.cuda.synchronize()
    second_call_time = time.time() - start
    
    print(f"   第一次调用（含编译）: {first_call_time:.2f}s")
    print(f"   第二次调用（缓存命中）: {second_call_time:.2f}s")
    print(f"   缓存效果: {'✓' if second_call_time < 0.1 else '✗ 缓存失效!'}")
    
    # 2. 测试不同输入形状的影响
    print("\n2. 测试不同batch size的编译缓存...")
    for bs in [1, 2, 4]:
        test_batch = torch.randn(bs, size, size, device=device)
        torch.cuda.synchronize()
        start = time.time()
        _ = pde_solver.rollout_seconds(test_batch, 19)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"   batch_size={bs}: {elapsed:.2f}s")
    
    # 3. 测试内存连续性的影响
    print("\n3. 测试内存连续性...")
    non_contiguous = dummy_batch.transpose(1, 2).transpose(1, 2)  # 破坏连续性
    contiguous = non_contiguous.contiguous()
    
    torch.cuda.synchronize()
    start = time.time()
    _ = pde_solver.rollout_seconds(non_contiguous, 19)
    torch.cuda.synchronize()
    non_cont_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    _ = pde_solver.rollout_seconds(contiguous, 19)
    torch.cuda.synchronize()
    cont_time = time.time() - start
    
    print(f"   非连续内存: {non_cont_time:.2f}s")
    print(f"   连续内存: {cont_time:.2f}s")
    
    # 4. 查看JAX编译缓存状态
    print(f"\n4. JAX编译缓存状态:")
    print(f"   VJP缓存大小: {len(pde_solver.__class__._vjp_cache) if hasattr(pde_solver.__class__, '_vjp_cache') else 'N/A'}")
    print(f"   编译函数缓存: {len(pde_solver._compiled)}")
    
    return {
        'first_call': first_call_time,
        'second_call': second_call_time,
        'cache_effective': second_call_time < 0.1,
        'non_contiguous': non_cont_time,
        'contiguous': cont_time
    }

# ====== 优化的PDE Solver ======
class OptimizedDifferentiablePDESolver(DifferentiablePDESolver):
    """优化版本的PDE求解器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 强制预编译常用的函数
        self._warmup_cache()
        print("PDE Solver 预热完成")
    
    def _warmup_cache(self):
        """预热编译缓存"""
        device = self.device
        common_shapes = [(1, 256, 256), (2, 256, 256), (4, 256, 256)]
        common_times = [9, 19, 29]  # T_in + T - 1, 第19帧, etc.
        
        print("预热PDE编译缓存...")
        for shape in common_shapes:
            dummy = torch.zeros(shape, device=device, dtype=torch.float32)
            for T in common_times:
                # 触发编译
                _ = self._get_jitted(dummy.ndim, T)
                # 实际执行一次确保编译完成
                with torch.no_grad():
                    _ = self.rollout_seconds(dummy, T)
        
    def rollout_seconds(self, x0: torch.Tensor, T_seconds: int):
        """优化版本：确保输入连续性"""
        # 强制连续化输入
        if not x0.is_contiguous():
            x0 = x0.contiguous()
        
        return super().rollout_seconds(x0, T_seconds)

# ====== 优化的攻击配置 ======
def create_optimized_attack_config(size=256):
    """创建优化的攻击配置"""
    return PGDAdamConfig(
        epsilon=0.0006 * (size * size),
        alpha=2.0,  # 增大步长
        num_steps=20,  # 大幅减少步数
        norm=2,
        mode_spec="wwwwwwwwww",
        beta1=0.9, 
        beta2=0.999, 
        adam_eps=1e-8, 
        amsgrad=False,
        use_sign_for_linf=True
    )

# ====== 性能监控装饰器 ======
def monitor_performance(func_name):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            if elapsed > 1.0:  # 只记录超过1秒的调用
                print(f"⚠️  {func_name} took {elapsed:.2f}s")
            return result
        return wrapper
    return decorator

# ====== 快速诊断脚本 ======
def quick_diagnosis():
    """快速诊断性能问题"""
    print("🔍 快速诊断开始...")
    
    # 检查环境变量
    cache_dir = os.environ.get("jax_compilation_cache_dir")
    measure_pde = os.environ.get("MEASURE_PDE", "1")
    
    print(f"JAX缓存目录: {cache_dir}")
    print(f"PDE性能测量: {measure_pde}")
    
    if measure_pde == "1":
        print("⚠️  建议设置 MEASURE_PDE=0 禁用性能测量")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
        print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
    
    # 运行基础性能测试
    try:
        diag_results = diagnose_performance_bottleneck()
        
        if not diag_results['cache_effective']:
            print("\n❌ 问题识别: JAX编译缓存无效!")
            print("解决方案:")
            print("1. 确保solver._get_jitted()返回稳定的函数对象")
            print("2. 检查输入张量的连续性")
            print("3. 验证JAX缓存目录的写入权限")
            
        if diag_results['non_contiguous'] > diag_results['contiguous'] * 2:
            print("\n⚠️  内存连续性影响性能")
            print("解决方案: 调用.contiguous()确保张量连续性")
            
    except Exception as e:
        print(f"诊断过程中出错: {e}")

if __name__ == "__main__":
    # 禁用PDE性能测量
    os.environ["MEASURE_PDE"] = "0"
    
    quick_diagnosis()