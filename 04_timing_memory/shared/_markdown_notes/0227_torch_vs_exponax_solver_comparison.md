# PyTorch solvers.py vs Exponax/JAX Solver Comparison

Each solver was run in a separate process. `first_run_seconds` includes JAX JIT compilation for Exponax; `second_run_seconds` is the warm run after compilation.

## Settings

```json
{
  "seed": 2026,
  "burgers": {
    "batch": 4,
    "nx": 256,
    "t_final": 0.1,
    "dt": 0.001,
    "nu": 0.001,
    "domain_extent": 2.0
  },
  "ns": {
    "batch": 2,
    "nx": 64,
    "t_final": 1,
    "dt": 0.005,
    "nu": 1e-05,
    "domain_extent": 1.0
  }
}
```

## Summary

| case | comparison relative L2 | max abs | PyTorch warm sec | Exponax/JAX warm sec | PyTorch global delta MiB | Exponax/JAX global delta MiB | PyTorch process peak MiB | Exponax/JAX process peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| burgers_1d | 8.87537e-07 | 1.01328e-06 | 0.0707752 | 0.0300273 | 8 | 20 | 0 | 0 |
| ns_2d | 2.45074e-06 | 3.09944e-06 | 0.279536 | 0.306505 | 10 | 12 | 0 | 0 |

## Full Run Details

```json
{
  "burgers_1d": {
    "runs": {
      "pytorch_solvers_py": {
        "case": "burgers_1d",
        "framework": "pytorch_solvers_py",
        "backend": {
          "torch_cuda_available": true,
          "torch_device": "NVIDIA A100-SXM4-80GB",
          "device_used": "cuda"
        },
        "output_shape": [
          4,
          256
        ],
        "first_run_seconds": 0.19670322502497584,
        "second_run_seconds": 0.07077517302241176,
        "memory": {
          "sample_count": 3,
          "sampled_process_gpu_peak_mib": 0.0,
          "sampled_global_gpu_start_mib": 13527.0,
          "sampled_global_gpu_peak_mib": 13535.0,
          "sampled_global_gpu_peak_delta_mib": 8.0,
          "torch_peak_allocated_mib": 0.1279296875,
          "torch_peak_reserved_mib": 2.0
        }
      },
      "jax_exponax": {
        "case": "burgers_1d",
        "framework": "jax_exponax",
        "backend": {
          "jax_backend": "gpu",
          "jax_devices": [
            "cuda:0"
          ]
        },
        "output_shape": [
          4,
          256
        ],
        "first_run_seconds": 0.35368310404010117,
        "second_run_seconds": 0.0300273469183594,
        "memory": {
          "sample_count": 4,
          "sampled_process_gpu_peak_mib": 0.0,
          "sampled_global_gpu_start_mib": 13461.0,
          "sampled_global_gpu_peak_mib": 13481.0,
          "sampled_global_gpu_peak_delta_mib": 20.0
        }
      }
    },
    "comparison": {
      "max_abs": 1.0132789611816406e-06,
      "mae": 1.3415046851150692e-07,
      "rmse": 1.8219941466668388e-07,
      "relative_l2": 8.87536543814349e-07
    }
  },
  "ns_2d": {
    "runs": {
      "pytorch_solvers_py": {
        "case": "ns_2d",
        "framework": "pytorch_solvers_py",
        "backend": {
          "torch_cuda_available": true,
          "torch_device": "NVIDIA A100-SXM4-80GB",
          "device_used": "cuda"
        },
        "output_shape": [
          2,
          64,
          64,
          2
        ],
        "first_run_seconds": 0.4916207219939679,
        "second_run_seconds": 0.27953608403913677,
        "memory": {
          "sample_count": 7,
          "sampled_process_gpu_peak_mib": 0.0,
          "sampled_global_gpu_start_mib": 13527.0,
          "sampled_global_gpu_peak_mib": 13537.0,
          "sampled_global_gpu_peak_delta_mib": 10.0,
          "torch_peak_allocated_mib": 1.91064453125,
          "torch_peak_reserved_mib": 4.0
        }
      },
      "jax_exponax": {
        "case": "ns_2d",
        "framework": "jax_exponax",
        "backend": {
          "jax_backend": "gpu",
          "jax_devices": [
            "cuda:0"
          ]
        },
        "output_shape": [
          2,
          64,
          64,
          2
        ],
        "first_run_seconds": 1.1973947109654546,
        "second_run_seconds": 0.30650468403473496,
        "memory": {
          "sample_count": 12,
          "sampled_process_gpu_peak_mib": 0.0,
          "sampled_global_gpu_start_mib": 13501.0,
          "sampled_global_gpu_peak_mib": 13513.0,
          "sampled_global_gpu_peak_delta_mib": 12.0
        }
      }
    },
    "comparison": {
      "max_abs": 3.0994415283203125e-06,
      "mae": 2.755718071512092e-07,
      "rmse": 5.021263973503665e-07,
      "relative_l2": 2.450741703796666e-06,
      "per_time_index": [
        {
          "time_index": 0,
          "max_abs": 0.0,
          "mae": 0.0,
          "rmse": 0.0,
          "relative_l2": 0.0
        },
        {
          "time_index": 1,
          "max_abs": 3.0994415283203125e-06,
          "mae": 5.511436143024184e-07,
          "rmse": 7.10114022695052e-07,
          "relative_l2": 3.129692913717008e-06
        }
      ]
    }
  }
}
```
