# PyTorch solvers.py vs Exponax/JAX Solver Comparison

Each solver was run in a separate process. `first_run_seconds` includes JAX JIT compilation for Exponax; `second_run_seconds` is the warm run after compilation.

## Settings

```json
{
  "seed": 2026,
  "seeds": [
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2032,
    2033,
    2034,
    2035
  ],
  "num_conditions": 10,
  "plot_dir": "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10",
  "max_visualized_samples_per_seed": 1,
  "burgers": {
    "batch": 1,
    "nx": 1024,
    "t_final": 1.0,
    "dt": 0.001,
    "nu": 0.001,
    "domain_extent": 2.0
  },
  "ns": {
    "batch": 1,
    "nx": 256,
    "t_final": 20,
    "dt": 0.005,
    "nu": 1e-05,
    "domain_extent": 1.0
  }
}
```

## Error Summary Across Conditions

| case | conditions | relative L2 mean | relative L2 std | max abs mean | max abs max |
|---|---:|---:|---:|---:|---:|
| burgers_1d | 10 | 3.60008e-06 | 1.62089e-06 | 6.097e-06 | 1.29864e-05 |
| ns_2d | 10 | 0.000960721 | 0.000797421 | 0.0326048 | 0.101758 |

## Runtime And Memory Summary

`p_value` is from a paired t-test across the repeated initial conditions. The paired test is `PyTorch - Exponax/JAX`, so a negative mean difference means PyTorch was smaller/faster.

| case | metric | PyTorch mean | PyTorch std | Exponax/JAX mean | Exponax/JAX std | mean PyTorch-JAX | paired p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| burgers_1d | cold seconds | 0.749614 | 0.0149871 | 0.572556 | 0.0226659 | 0.177058 | 2.40313e-09 |
| burgers_1d | warm seconds | 0.639435 | 0.0173103 | 0.271111 | 0.00421149 | 0.368323 | 1.48306e-13 |
| burgers_1d | global GPU delta MiB | 8 | 0 | 19.2 | 1.68655 | -11.2 | 5.90199e-09 |
| burgers_1d | process GPU peak MiB | 0 | 0 | 0 | 0 | 0 | 1 |
| ns_2d | cold seconds | 5.84725 | 0.129746 | 4.53326 | 0.0556899 | 1.314 | 3.67346e-10 |
| ns_2d | warm seconds | 5.59492 | 0.153867 | 3.689 | 0.0172626 | 1.90592 | 3.62924e-11 |
| ns_2d | global GPU delta MiB | 64 | 0 | 45.6 | 1.26491 | 18.4 | 5.42637e-12 |
| ns_2d | process GPU peak MiB | 0 | 0 | 0 | 0 | 0 | 1 |

## Visualization Files

- Plot directory: `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10`
- `burgers_1d` plots: 10
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2026_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2027_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2028_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2029_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2030_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2031_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2032_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2033_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2034_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2035_sample0.png`
- `ns_2d` plots: 10
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2026_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2027_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2028_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2029_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2030_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2031_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2032_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2033_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2034_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2035_sample0.png`

## Full Run Details

```json
{
  "burgers_1d": {
    "conditions": [
      {
        "condition_index": 0,
        "seed": 2026,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7438625879585743,
            "second_run_seconds": 0.632341934950091,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.6119461479829624,
            "second_run_seconds": 0.2727402710588649,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 1.2986361980438232e-05,
          "mae": 8.741590136196464e-07,
          "rmse": 1.2990686855118838e-06,
          "relative_l2": 6.183686309668701e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2026_sample0.png"
        ]
      },
      {
        "condition_index": 1,
        "seed": 2027,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7573624739889055,
            "second_run_seconds": 0.6319539959076792,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5970266349613667,
            "second_run_seconds": 0.2702741449465975,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 3.7141144275665283e-06,
          "mae": 1.5343084669439122e-07,
          "rmse": 3.253390445934201e-07,
          "relative_l2": 2.8294348339841235e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2027_sample0.png"
        ]
      },
      {
        "condition_index": 2,
        "seed": 2028,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.741931441007182,
            "second_run_seconds": 0.6297187289455906,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5618213550187647,
            "second_run_seconds": 0.27368420001585037,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13465.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 16.0
            }
          }
        },
        "comparison": {
          "max_abs": 9.268522262573242e-06,
          "mae": 2.4973451218102127e-07,
          "rmse": 6.512792651847121e-07,
          "relative_l2": 3.979084340244299e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2028_sample0.png"
        ]
      },
      {
        "condition_index": 3,
        "seed": 2029,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7464503969531506,
            "second_run_seconds": 0.6202517300844193,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5866576069965959,
            "second_run_seconds": 0.27094919397495687,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13465.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 16.0
            }
          }
        },
        "comparison": {
          "max_abs": 2.600252628326416e-06,
          "mae": 3.8644884625682607e-07,
          "rmse": 5.549310913011141e-07,
          "relative_l2": 4.515388809522847e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2029_sample0.png"
        ]
      },
      {
        "condition_index": 4,
        "seed": 2030,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7261554700089619,
            "second_run_seconds": 0.6256290141027421,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5344213569769636,
            "second_run_seconds": 0.26454426092095673,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 8.724629878997803e-06,
          "mae": 2.982924343086779e-07,
          "rmse": 8.373993978239014e-07,
          "relative_l2": 5.502113253896823e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2030_sample0.png"
        ]
      },
      {
        "condition_index": 5,
        "seed": 2031,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.739980575046502,
            "second_run_seconds": 0.6369463520823047,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5655183719936758,
            "second_run_seconds": 0.2675095880404115,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 3.8743019104003906e-07,
          "mae": 1.2834516383009031e-07,
          "rmse": 1.5837926525819057e-07,
          "relative_l2": 1.6746889741625637e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2031_sample0.png"
        ]
      },
      {
        "condition_index": 6,
        "seed": 2032,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7694206520682201,
            "second_run_seconds": 0.6501809869660065,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5730201540281996,
            "second_run_seconds": 0.27563121903222054,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 2.980232238769531e-07,
          "mae": 5.76501406612806e-08,
          "rmse": 7.626281472994378e-08,
          "relative_l2": 8.647868980915518e-07
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2032_sample0.png"
        ]
      },
      {
        "condition_index": 7,
        "seed": 2033,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7571530250133947,
            "second_run_seconds": 0.6645328510785475,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5827721430687234,
            "second_run_seconds": 0.2652474510250613,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 1.2375414371490479e-05,
          "mae": 4.180146788712591e-07,
          "rmse": 9.547286481392803e-07,
          "relative_l2": 2.8960664621990873e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2033_sample0.png"
        ]
      },
      {
        "condition_index": 8,
        "seed": 2034,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7387346959440038,
            "second_run_seconds": 0.6301848309813067,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5589160000672564,
            "second_run_seconds": 0.2738222110783681,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 7.241964340209961e-06,
          "mae": 8.730949048185721e-07,
          "rmse": 1.1637259831331903e-06,
          "relative_l2": 3.8999605749268085e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2034_sample0.png"
        ]
      },
      {
        "condition_index": 9,
        "seed": 2035,
        "runs": {
          "pytorch_solvers_py": {
            "case": "burgers_1d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              1024
            ],
            "first_run_seconds": 0.7750849069561809,
            "second_run_seconds": 0.6726070019649342,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.451171875,
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
              1,
              1024
            ],
            "first_run_seconds": 0.5534567759605125,
            "second_run_seconds": 0.27671197801828384,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13461.0,
              "sampled_global_gpu_peak_mib": 13481.0,
              "sampled_global_gpu_peak_delta_mib": 20.0
            }
          }
        },
        "comparison": {
          "max_abs": 3.373250365257263e-06,
          "mae": 2.815413608914241e-07,
          "rmse": 3.893139250976674e-07,
          "relative_l2": 3.6555818496708525e-06
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/burgers_1d/burgers_seed2035_sample0.png"
        ]
      }
    ],
    "statistics": {
      "comparison": {
        "max_abs": {
          "count": 10,
          "mean": 6.096996366977692e-06,
          "std": 4.6750787047403044e-06,
          "min": 2.980232238769531e-07,
          "max": 1.2986361980438232e-05
        },
        "mae": {
          "count": 10,
          "mean": 3.720711902133189e-07,
          "std": 2.8676476149489607e-07,
          "min": 5.76501406612806e-08,
          "max": 8.741590136196464e-07
        },
        "rmse": {
          "count": 10,
          "mean": 6.410428120773304e-07,
          "std": 4.170860981631721e-07,
          "min": 7.626281472994378e-08,
          "max": 1.2990686855118838e-06
        },
        "relative_l2": {
          "count": 10,
          "mean": 3.600079230636766e-06,
          "std": 1.620894244701797e-06,
          "min": 8.647868980915518e-07,
          "max": 6.183686309668701e-06
        }
      },
      "runs": {
        "pytorch_solvers_py": {
          "first_run_seconds": {
            "count": 10,
            "mean": 0.7496136224945076,
            "std": 0.014987105246438924,
            "min": 0.7261554700089619,
            "max": 0.7750849069561809
          },
          "second_run_seconds": {
            "count": 10,
            "mean": 0.6394347427063621,
            "std": 0.017310257338578817,
            "min": 0.6202517300844193,
            "max": 0.6726070019649342
          },
          "sampled_process_gpu_peak_mib": {
            "count": 10,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 10,
            "mean": 8.0,
            "std": 0.0,
            "min": 8.0,
            "max": 8.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 10,
            "mean": 13535.0,
            "std": 0.0,
            "min": 13535.0,
            "max": 13535.0
          },
          "torch_peak_allocated_mib": {
            "count": 10,
            "mean": 0.451171875,
            "std": 0.0,
            "min": 0.451171875,
            "max": 0.451171875
          },
          "torch_peak_reserved_mib": {
            "count": 10,
            "mean": 2.0,
            "std": 0.0,
            "min": 2.0,
            "max": 2.0
          }
        },
        "jax_exponax": {
          "first_run_seconds": {
            "count": 10,
            "mean": 0.5725556547055021,
            "std": 0.02266587703719595,
            "min": 0.5344213569769636,
            "max": 0.6119461479829624
          },
          "second_run_seconds": {
            "count": 10,
            "mean": 0.27111145181115714,
            "std": 0.004211489908202967,
            "min": 0.26454426092095673,
            "max": 0.27671197801828384
          },
          "sampled_process_gpu_peak_mib": {
            "count": 10,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 10,
            "mean": 19.2,
            "std": 1.6865480854231356,
            "min": 16.0,
            "max": 20.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 10,
            "mean": 13481.0,
            "std": 0.0,
            "min": 13481.0,
            "max": 13481.0
          },
          "torch_peak_allocated_mib": {
            "count": 0,
            "mean": null,
            "std": null,
            "min": null,
            "max": null
          },
          "torch_peak_reserved_mib": {
            "count": 0,
            "mean": null,
            "std": null,
            "min": null,
            "max": null
          }
        }
      },
      "paired_tests_pytorch_minus_jax": {
        "first_run_seconds": {
          "count": 10,
          "mean_left_minus_right": 0.17705796778900548,
          "t_statistic": 23.243646566661003,
          "p_value": 2.403125918516613e-09
        },
        "second_run_seconds": {
          "count": 10,
          "mean_left_minus_right": 0.368323290895205,
          "t_statistic": 68.69503154790506,
          "p_value": 1.4830563648536093e-13
        },
        "sampled_process_gpu_peak_mib": {
          "count": 10,
          "mean_left_minus_right": 0.0,
          "t_statistic": 0.0,
          "p_value": 1.0
        },
        "sampled_global_gpu_peak_delta_mib": {
          "count": 10,
          "mean_left_minus_right": -11.2,
          "t_statistic": -21.0,
          "p_value": 5.901994059532861e-09
        },
        "sampled_global_gpu_peak_mib": {
          "count": 10,
          "mean_left_minus_right": 54.0,
          "t_statistic": null,
          "p_value": 0.0
        }
      }
    }
  },
  "ns_2d": {
    "conditions": [
      {
        "condition_index": 0,
        "seed": 2026,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.759284605970606,
            "second_run_seconds": 5.5040509949903935,
            "memory": {
              "sample_count": 20,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.603050834964961,
            "second_run_seconds": 3.6874663409544155,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.04043865203857422,
          "mae": 0.00029602518770843744,
          "rmse": 0.001583258737809956,
          "relative_l2": 0.0013180350651964545,
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
              "max_abs": 1.7285346984863281e-06,
              "mae": 2.9084077368679573e-07,
              "rmse": 3.692781831432512e-07,
              "relative_l2": 1.8026415773420013e-06
            },
            {
              "time_index": 2,
              "max_abs": 3.2782554626464844e-06,
              "mae": 5.392164439399494e-07,
              "rmse": 6.807886165915988e-07,
              "relative_l2": 2.3681438960920786e-06
            },
            {
              "time_index": 3,
              "max_abs": 5.304813385009766e-06,
              "mae": 8.126352213366772e-07,
              "rmse": 1.0232195108983433e-06,
              "relative_l2": 2.7043124646297656e-06
            },
            {
              "time_index": 4,
              "max_abs": 6.198883056640625e-06,
              "mae": 1.1051506589865312e-06,
              "rmse": 1.3804509535475518e-06,
              "relative_l2": 2.920573479059385e-06
            },
            {
              "time_index": 5,
              "max_abs": 8.463859558105469e-06,
              "mae": 1.5038349374663085e-06,
              "rmse": 1.8648274817678612e-06,
              "relative_l2": 3.2799011933093425e-06
            },
            {
              "time_index": 6,
              "max_abs": 1.049041748046875e-05,
              "mae": 1.8115478042091127e-06,
              "rmse": 2.2377241748472443e-06,
              "relative_l2": 3.363744099260657e-06
            },
            {
              "time_index": 7,
              "max_abs": 1.1324882507324219e-05,
              "mae": 2.3556731321150437e-06,
              "rmse": 2.9360751341300784e-06,
              "relative_l2": 3.851573637803085e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.4901161193847656e-05,
              "mae": 2.9163143153709825e-06,
              "rmse": 3.611701458794414e-06,
              "relative_l2": 4.202088803140214e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.633167266845703e-05,
              "mae": 3.3679898479022086e-06,
              "rmse": 4.186754722468322e-06,
              "relative_l2": 4.376224296720466e-06
            },
            {
              "time_index": 10,
              "max_abs": 2.0503997802734375e-05,
              "mae": 3.880855729221366e-06,
              "rmse": 4.8146707740670536e-06,
              "relative_l2": 4.568681561067933e-06
            },
            {
              "time_index": 11,
              "max_abs": 8.130073547363281e-05,
              "mae": 8.459672244498506e-06,
              "rmse": 1.1663609257084318e-05,
              "relative_l2": 1.0134647709492128e-05
            },
            {
              "time_index": 12,
              "max_abs": 0.00021123886108398438,
              "mae": 2.2109641577117145e-05,
              "rmse": 3.313501656521112e-05,
              "relative_l2": 2.6556623197393492e-05
            },
            {
              "time_index": 13,
              "max_abs": 0.0003236532211303711,
              "mae": 2.6379777409601957e-05,
              "rmse": 4.080391954630613e-05,
              "relative_l2": 3.03528577205725e-05
            },
            {
              "time_index": 14,
              "max_abs": 0.00020170211791992188,
              "mae": 1.946501652128063e-05,
              "rmse": 2.9285851269378327e-05,
              "relative_l2": 2.0329056496848352e-05
            },
            {
              "time_index": 15,
              "max_abs": 0.0004404783248901367,
              "mae": 2.7256688554189168e-05,
              "rmse": 4.3893393012695014e-05,
              "relative_l2": 2.8569351343321614e-05
            },
            {
              "time_index": 16,
              "max_abs": 0.0010313987731933594,
              "mae": 5.7531753554940224e-05,
              "rmse": 0.00010808077786350623,
              "relative_l2": 6.624720117542893e-05
            },
            {
              "time_index": 17,
              "max_abs": 0.00310516357421875,
              "mae": 0.00018855862435884774,
              "rmse": 0.0003919786249753088,
              "relative_l2": 0.00022716888634022325
            },
            {
              "time_index": 18,
              "max_abs": 0.010097026824951172,
              "mae": 0.000735947978682816,
              "rmse": 0.0014187977649271488,
              "relative_l2": 0.0007805573404766619
            },
            {
              "time_index": 19,
              "max_abs": 0.04043865203857422,
              "mae": 0.002788239624351263,
              "rmse": 0.005720988381654024,
              "relative_l2": 0.0029998188838362694
            },
            {
              "time_index": 20,
              "max_abs": 0.0337071418762207,
              "mae": 0.0023239958100020885,
              "rmse": 0.004210355691611767,
              "relative_l2": 0.002112359507009387
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2026_sample0.png"
        ]
      },
      {
        "condition_index": 1,
        "seed": 2027,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.721714852028526,
            "second_run_seconds": 5.498761135037057,
            "memory": {
              "sample_count": 20,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.467579260934144,
            "second_run_seconds": 3.67750084400177,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13505.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 42.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.10175776481628418,
          "mae": 0.0005000843084417284,
          "rmse": 0.002950145862996578,
          "relative_l2": 0.002542247762903571,
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
              "max_abs": 1.6093254089355469e-06,
              "mae": 2.2186192438766739e-07,
              "rmse": 2.7903152499675343e-07,
              "relative_l2": 1.6957794741756516e-06
            },
            {
              "time_index": 2,
              "max_abs": 2.2649765014648438e-06,
              "mae": 3.764057510124985e-07,
              "rmse": 4.7527083779641544e-07,
              "relative_l2": 1.9617950783867855e-06
            },
            {
              "time_index": 3,
              "max_abs": 3.933906555175781e-06,
              "mae": 6.544888719872688e-07,
              "rmse": 8.231079391407548e-07,
              "relative_l2": 2.4796299840090796e-06
            },
            {
              "time_index": 4,
              "max_abs": 6.258487701416016e-06,
              "mae": 1.009814923236263e-06,
              "rmse": 1.2671627018789877e-06,
              "relative_l2": 2.9745167466899147e-06
            },
            {
              "time_index": 5,
              "max_abs": 6.67572021484375e-06,
              "mae": 1.320269348070724e-06,
              "rmse": 1.6487593939018552e-06,
              "relative_l2": 3.158683966830722e-06
            },
            {
              "time_index": 6,
              "max_abs": 8.046627044677734e-06,
              "mae": 1.661246187723009e-06,
              "rmse": 2.046326471827342e-06,
              "relative_l2": 3.3066096420952817e-06
            },
            {
              "time_index": 7,
              "max_abs": 1.0132789611816406e-05,
              "mae": 2.023794877459295e-06,
              "rmse": 2.462087877574959e-06,
              "relative_l2": 3.4378529107925715e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.1920928955078125e-05,
              "mae": 2.471700327077997e-06,
              "rmse": 3.0331980269693304e-06,
              "relative_l2": 3.7279032767401077e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.3828277587890625e-05,
              "mae": 2.967224190797424e-06,
              "rmse": 3.663333927761414e-06,
              "relative_l2": 4.0206127778219525e-06
            },
            {
              "time_index": 10,
              "max_abs": 1.8835067749023438e-05,
              "mae": 3.5502007449395023e-06,
              "rmse": 4.389408331917366e-06,
              "relative_l2": 4.352206815383397e-06
            },
            {
              "time_index": 11,
              "max_abs": 2.4199485778808594e-05,
              "mae": 4.416047886479646e-06,
              "rmse": 5.488462647917913e-06,
              "relative_l2": 4.963348146702629e-06
            },
            {
              "time_index": 12,
              "max_abs": 5.519390106201172e-05,
              "mae": 6.53107235848438e-06,
              "rmse": 8.718769095139578e-06,
              "relative_l2": 7.24868914403487e-06
            },
            {
              "time_index": 13,
              "max_abs": 0.0002593994140625,
              "mae": 1.634934415051248e-05,
              "rmse": 2.7831149054691195e-05,
              "relative_l2": 2.141719414794352e-05
            },
            {
              "time_index": 14,
              "max_abs": 0.0009658336639404297,
              "mae": 6.007962656440213e-05,
              "rmse": 0.00012138950114604086,
              "relative_l2": 8.69760406203568e-05
            },
            {
              "time_index": 15,
              "max_abs": 0.0017457008361816406,
              "mae": 0.0001143185218097642,
              "rmse": 0.00023033849720377475,
              "relative_l2": 0.00015446792531292886
            },
            {
              "time_index": 16,
              "max_abs": 0.0017325878143310547,
              "mae": 0.00013148343714419752,
              "rmse": 0.0002608713984955102,
              "relative_l2": 0.00016452047566417605
            },
            {
              "time_index": 17,
              "max_abs": 0.005273580551147461,
              "mae": 0.0005193323595449328,
              "rmse": 0.0009673239546827972,
              "relative_l2": 0.0005762565415352583
            },
            {
              "time_index": 18,
              "max_abs": 0.009470939636230469,
              "mae": 0.0006764803547412157,
              "rmse": 0.001319814589805901,
              "relative_l2": 0.0007458289037458599
            },
            {
              "time_index": 19,
              "max_abs": 0.028338193893432617,
              "mae": 0.00237285066395998,
              "rmse": 0.004750889725983143,
              "relative_l2": 0.0025578895583748817
            },
            {
              "time_index": 20,
              "max_abs": 0.10175776481628418,
              "mae": 0.006583671551197767,
              "rmse": 0.012545325793325901,
              "relative_l2": 0.006464722100645304
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2027_sample0.png"
        ]
      },
      {
        "condition_index": 2,
        "seed": 2028,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.7264970110263675,
            "second_run_seconds": 5.4516088790260255,
            "memory": {
              "sample_count": 20,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.595310288947076,
            "second_run_seconds": 3.712838766979985,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.06946611404418945,
          "mae": 0.00041282904567196965,
          "rmse": 0.002464012475684285,
          "relative_l2": 0.0019839738961309195,
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
              "max_abs": 2.205371856689453e-06,
              "mae": 3.2800625149320695e-07,
              "rmse": 4.1614680412749294e-07,
              "relative_l2": 1.8073091041515e-06
            },
            {
              "time_index": 2,
              "max_abs": 3.4868717193603516e-06,
              "mae": 6.191038437464158e-07,
              "rmse": 7.765802934045496e-07,
              "relative_l2": 2.4082837626338005e-06
            },
            {
              "time_index": 3,
              "max_abs": 4.827976226806641e-06,
              "mae": 9.112807219935348e-07,
              "rmse": 1.128848339249089e-06,
              "relative_l2": 2.7004305138689233e-06
            },
            {
              "time_index": 4,
              "max_abs": 6.854534149169922e-06,
              "mae": 1.3419887636700878e-06,
              "rmse": 1.6524517150173779e-06,
              "relative_l2": 3.208691168765654e-06
            },
            {
              "time_index": 5,
              "max_abs": 7.808208465576172e-06,
              "mae": 1.6978551684587728e-06,
              "rmse": 2.0900374693155754e-06,
              "relative_l2": 3.4116271763195982e-06
            },
            {
              "time_index": 6,
              "max_abs": 1.043081283569336e-05,
              "mae": 2.2018466552253813e-06,
              "rmse": 2.699164724617731e-06,
              "relative_l2": 3.7986189909133827e-06
            },
            {
              "time_index": 7,
              "max_abs": 1.2516975402832031e-05,
              "mae": 2.5641281808930216e-06,
              "rmse": 3.1429190130438656e-06,
              "relative_l2": 3.886630111082923e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.2874603271484375e-05,
              "mae": 2.98633881357091e-06,
              "rmse": 3.6780809296033112e-06,
              "relative_l2": 4.056173111166572e-06
            },
            {
              "time_index": 9,
              "max_abs": 2.0325183868408203e-05,
              "mae": 3.5974780985270627e-06,
              "rmse": 4.462333436094923e-06,
              "relative_l2": 4.440455086296424e-06
            },
            {
              "time_index": 10,
              "max_abs": 0.00010335445404052734,
              "mae": 1.019411411107285e-05,
              "rmse": 1.701455948932562e-05,
              "relative_l2": 1.5425410310854204e-05
            },
            {
              "time_index": 11,
              "max_abs": 0.00016486644744873047,
              "mae": 2.0033199689351022e-05,
              "rmse": 3.050421946682036e-05,
              "relative_l2": 2.5399185688002035e-05
            },
            {
              "time_index": 12,
              "max_abs": 0.0006045103073120117,
              "mae": 6.319754902506247e-05,
              "rmse": 0.00011372459266567603,
              "relative_l2": 8.75659316079691e-05
            },
            {
              "time_index": 13,
              "max_abs": 0.0010688304901123047,
              "mae": 0.00013015377044212073,
              "rmse": 0.000212673403439112,
              "relative_l2": 0.0001523365208413452
            },
            {
              "time_index": 14,
              "max_abs": 0.0014581680297851562,
              "mae": 0.0001367992372252047,
              "rmse": 0.0002414517366560176,
              "relative_l2": 0.00016174597840290517
            },
            {
              "time_index": 15,
              "max_abs": 0.0016245841979980469,
              "mae": 0.00012544961646199226,
              "rmse": 0.00020762540225405246,
              "relative_l2": 0.00013070579734630883
            },
            {
              "time_index": 16,
              "max_abs": 0.003035306930541992,
              "mae": 0.00025399812147952616,
              "rmse": 0.0005039487150497735,
              "relative_l2": 0.00029948263545520604
            },
            {
              "time_index": 17,
              "max_abs": 0.005622386932373047,
              "mae": 0.0003691929450724274,
              "rmse": 0.0006756520015187562,
              "relative_l2": 0.0003806985623668879
            },
            {
              "time_index": 18,
              "max_abs": 0.012232303619384766,
              "mae": 0.0012886555632576346,
              "rmse": 0.002208824036642909,
              "relative_l2": 0.001185371889732778
            },
            {
              "time_index": 19,
              "max_abs": 0.01640772819519043,
              "mae": 0.0013272399082779884,
              "rmse": 0.002650247188284993,
              "relative_l2": 0.0013605122221633792
            },
            {
              "time_index": 20,
              "max_abs": 0.06946611404418945,
              "mae": 0.004928247537463903,
              "rmse": 0.010710950940847397,
              "relative_l2": 0.005282730795443058
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2028_sample0.png"
        ]
      },
      {
        "condition_index": 3,
        "seed": 2029,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 6.016493416042067,
            "second_run_seconds": 5.522303814999759,
            "memory": {
              "sample_count": 21,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.5319033729610965,
            "second_run_seconds": 3.7025027900235727,
            "memory": {
              "sample_count": 12,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.0013551712036132812,
          "mae": 1.577849434397649e-05,
          "rmse": 5.235729258856736e-05,
          "relative_l2": 4.397787051857449e-05,
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
              "max_abs": 1.4156103134155273e-06,
              "mae": 2.018285840676981e-07,
              "rmse": 2.542555250784062e-07,
              "relative_l2": 1.591733507666504e-06
            },
            {
              "time_index": 2,
              "max_abs": 1.9669532775878906e-06,
              "mae": 3.436764757225319e-07,
              "rmse": 4.366240773379104e-07,
              "relative_l2": 1.7543493413541e-06
            },
            {
              "time_index": 3,
              "max_abs": 3.6656856536865234e-06,
              "mae": 6.480546517195762e-07,
              "rmse": 8.16260524061363e-07,
              "relative_l2": 2.373752067796886e-06
            },
            {
              "time_index": 4,
              "max_abs": 5.841255187988281e-06,
              "mae": 1.0719777492340654e-06,
              "rmse": 1.333172576778452e-06,
              "relative_l2": 3.023919362021843e-06
            },
            {
              "time_index": 5,
              "max_abs": 7.927417755126953e-06,
              "mae": 1.3798621694149915e-06,
              "rmse": 1.7057780041795922e-06,
              "relative_l2": 3.166160695400322e-06
            },
            {
              "time_index": 6,
              "max_abs": 8.106231689453125e-06,
              "mae": 1.7111754004872637e-06,
              "rmse": 2.0869108539045556e-06,
              "relative_l2": 3.2759394343884196e-06
            },
            {
              "time_index": 7,
              "max_abs": 1.0371208190917969e-05,
              "mae": 2.2185085981618613e-06,
              "rmse": 2.7318512820784235e-06,
              "relative_l2": 3.714193326231907e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.2159347534179688e-05,
              "mae": 2.6104212338395882e-06,
              "rmse": 3.2175114483834477e-06,
              "relative_l2": 3.857643605442718e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.3887882232666016e-05,
              "mae": 2.989417225762736e-06,
              "rmse": 3.670533715194324e-06,
              "relative_l2": 3.935761924367398e-06
            },
            {
              "time_index": 10,
              "max_abs": 1.6927719116210938e-05,
              "mae": 3.520453901728615e-06,
              "rmse": 4.337071459303843e-06,
              "relative_l2": 4.206076482660137e-06
            },
            {
              "time_index": 11,
              "max_abs": 1.8477439880371094e-05,
              "mae": 4.1077601053984836e-06,
              "rmse": 5.056713689555181e-06,
              "relative_l2": 4.476366939343279e-06
            },
            {
              "time_index": 12,
              "max_abs": 3.516674041748047e-05,
              "mae": 5.112193775858032e-06,
              "rmse": 6.516343091789167e-06,
              "relative_l2": 5.305902504915139e-06
            },
            {
              "time_index": 13,
              "max_abs": 4.184246063232422e-05,
              "mae": 6.631410542468075e-06,
              "rmse": 8.437457836407702e-06,
              "relative_l2": 6.360402494465234e-06
            },
            {
              "time_index": 14,
              "max_abs": 9.97781753540039e-05,
              "mae": 1.4903635019436479e-05,
              "rmse": 1.9587454517022707e-05,
              "relative_l2": 1.3746146578341722e-05
            },
            {
              "time_index": 15,
              "max_abs": 0.00012373924255371094,
              "mae": 1.6154594050021842e-05,
              "rmse": 2.2374499167199247e-05,
              "relative_l2": 1.4688855117128696e-05
            },
            {
              "time_index": 16,
              "max_abs": 0.00030732154846191406,
              "mae": 2.5848286895779893e-05,
              "rmse": 3.912172542186454e-05,
              "relative_l2": 2.4128536097123288e-05
            },
            {
              "time_index": 17,
              "max_abs": 0.00031185150146484375,
              "mae": 1.929873542394489e-05,
              "rmse": 3.2304458727594465e-05,
              "relative_l2": 1.8788574379868805e-05
            },
            {
              "time_index": 18,
              "max_abs": 0.0008220672607421875,
              "mae": 5.543762745219283e-05,
              "rmse": 0.00010938202467514202,
              "relative_l2": 6.01970859861467e-05
            },
            {
              "time_index": 19,
              "max_abs": 0.0009002685546875,
              "mae": 8.505272853653878e-05,
              "rmse": 0.00015032854571472853,
              "relative_l2": 7.85285810707137e-05
            },
            {
              "time_index": 20,
              "max_abs": 0.0013551712036132812,
              "mae": 8.210603846237063e-05,
              "rmse": 0.00013909077097196132,
              "relative_l2": 6.917124846950173e-05
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2029_sample0.png"
        ]
      },
      {
        "condition_index": 4,
        "seed": 2030,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.728610194986686,
            "second_run_seconds": 5.472716053016484,
            "memory": {
              "sample_count": 20,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.548073407029733,
            "second_run_seconds": 3.6717236069962382,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.014816045761108398,
          "mae": 0.00021950594964437187,
          "rmse": 0.0006553914863616228,
          "relative_l2": 0.000577522732783109,
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
              "max_abs": 1.8030405044555664e-06,
              "mae": 3.1649716447645915e-07,
              "rmse": 3.9971328646970505e-07,
              "relative_l2": 1.976272869796958e-06
            },
            {
              "time_index": 2,
              "max_abs": 2.2649765014648438e-06,
              "mae": 4.276008667147835e-07,
              "rmse": 5.352767971089634e-07,
              "relative_l2": 2.152099796148832e-06
            },
            {
              "time_index": 3,
              "max_abs": 3.933906555175781e-06,
              "mae": 6.214418135641608e-07,
              "rmse": 7.777188670843316e-07,
              "relative_l2": 2.4264493276859866e-06
            },
            {
              "time_index": 4,
              "max_abs": 4.559755325317383e-06,
              "mae": 8.408979965679464e-07,
              "rmse": 1.0430176189402118e-06,
              "relative_l2": 2.5798285605560523e-06
            },
            {
              "time_index": 5,
              "max_abs": 6.67572021484375e-06,
              "mae": 1.2902314665552694e-06,
              "rmse": 1.6030039660108741e-06,
              "relative_l2": 3.2454042866447708e-06
            },
            {
              "time_index": 6,
              "max_abs": 7.331371307373047e-06,
              "mae": 1.6116762253659545e-06,
              "rmse": 1.993425257751369e-06,
              "relative_l2": 3.39796451953589e-06
            },
            {
              "time_index": 7,
              "max_abs": 9.238719940185547e-06,
              "mae": 1.9355570657353383e-06,
              "rmse": 2.3814384348952444e-06,
              "relative_l2": 3.496374574751826e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.1555850505828857e-05,
              "mae": 2.5135846044577193e-06,
              "rmse": 3.093779469054425e-06,
              "relative_l2": 3.9837477743276395e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.6689300537109375e-05,
              "mae": 2.941641469078604e-06,
              "rmse": 3.677463610074483e-06,
              "relative_l2": 4.213938609609613e-06
            },
            {
              "time_index": 10,
              "max_abs": 7.56978988647461e-05,
              "mae": 8.061316293606069e-06,
              "rmse": 1.222477021656232e-05,
              "relative_l2": 1.2613895705726463e-05
            },
            {
              "time_index": 11,
              "max_abs": 0.0005831718444824219,
              "mae": 5.1570008508861065e-05,
              "rmse": 8.237772271968424e-05,
              "relative_l2": 7.72878629504703e-05
            },
            {
              "time_index": 12,
              "max_abs": 0.0018575191497802734,
              "mae": 0.00021248494158498943,
              "rmse": 0.0003077089204452932,
              "relative_l2": 0.00026463967515155673
            },
            {
              "time_index": 13,
              "max_abs": 0.003119349479675293,
              "mae": 0.00032901496160775423,
              "rmse": 0.00045683086500503123,
              "relative_l2": 0.00036263943184167147
            },
            {
              "time_index": 14,
              "max_abs": 0.003432750701904297,
              "mae": 0.00033251565764658153,
              "rmse": 0.00048617683933116496,
              "relative_l2": 0.00035833826404996216
            },
            {
              "time_index": 15,
              "max_abs": 0.0015134811401367188,
              "mae": 0.0001554472401039675,
              "rmse": 0.00021994263806845993,
              "relative_l2": 0.00015129988605622202
            },
            {
              "time_index": 16,
              "max_abs": 0.0023758411407470703,
              "mae": 0.00018913220264948905,
              "rmse": 0.0002913848147727549,
              "relative_l2": 0.00018793894560076296
            },
            {
              "time_index": 17,
              "max_abs": 0.010077238082885742,
              "mae": 0.0008525988087058067,
              "rmse": 0.0014274617424234748,
              "relative_l2": 0.0008668000809848309
            },
            {
              "time_index": 18,
              "max_abs": 0.014816045761108398,
              "mae": 0.0009430245263502002,
              "rmse": 0.001660486334003508,
              "relative_l2": 0.0009528921218588948
            },
            {
              "time_index": 19,
              "max_abs": 0.010415792465209961,
              "mae": 0.0006880215369164944,
              "rmse": 0.0012669635470956564,
              "relative_l2": 0.000689636159222573
            },
            {
              "time_index": 20,
              "max_abs": 0.011013507843017578,
              "mae": 0.000835254555568099,
              "rmse": 0.0013929245760664344,
              "relative_l2": 0.0007219032850116491
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2030_sample0.png"
        ]
      },
      {
        "condition_index": 5,
        "seed": 2031,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 6.034176251967438,
            "second_run_seconds": 5.941829425981268,
            "memory": {
              "sample_count": 21,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.529059361084364,
            "second_run_seconds": 3.664630846004002,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.017209529876708984,
          "mae": 0.00012855387467425317,
          "rmse": 0.0007813085103407502,
          "relative_l2": 0.0006860801368020475,
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
              "max_abs": 1.1771917343139648e-06,
              "mae": 1.90955063317233e-07,
              "rmse": 2.415225139884569e-07,
              "relative_l2": 1.7929620526047074e-06
            },
            {
              "time_index": 2,
              "max_abs": 1.9371509552001953e-06,
              "mae": 3.314690673050791e-07,
              "rmse": 4.1959583541029133e-07,
              "relative_l2": 2.0174156816210598e-06
            },
            {
              "time_index": 3,
              "max_abs": 2.7120113372802734e-06,
              "mae": 5.323611276253359e-07,
              "rmse": 6.651848138972127e-07,
              "relative_l2": 2.2385383999790065e-06
            },
            {
              "time_index": 4,
              "max_abs": 4.149973392486572e-06,
              "mae": 8.479878488287795e-07,
              "rmse": 1.053470100487175e-06,
              "relative_l2": 2.6915349735645577e-06
            },
            {
              "time_index": 5,
              "max_abs": 5.4836273193359375e-06,
              "mae": 1.1756371804949595e-06,
              "rmse": 1.4570204029951128e-06,
              "relative_l2": 2.9874800020479597e-06
            },
            {
              "time_index": 6,
              "max_abs": 7.748603820800781e-06,
              "mae": 1.7219308574567549e-06,
              "rmse": 2.1234752693999326e-06,
              "relative_l2": 3.63005415238149e-06
            },
            {
              "time_index": 7,
              "max_abs": 1.1861324310302734e-05,
              "mae": 2.0036391106259543e-06,
              "rmse": 2.4680327896930976e-06,
              "relative_l2": 3.6151618587609846e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.245737075805664e-05,
              "mae": 2.6621532924764324e-06,
              "rmse": 3.283648766227998e-06,
              "relative_l2": 4.206526227790164e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.4185905456542969e-05,
              "mae": 3.2445439046568936e-06,
              "rmse": 3.997948624601122e-06,
              "relative_l2": 4.550455741991755e-06
            },
            {
              "time_index": 10,
              "max_abs": 1.7642974853515625e-05,
              "mae": 3.779093731282046e-06,
              "rmse": 4.665147116611479e-06,
              "relative_l2": 4.7772937250556424e-06
            },
            {
              "time_index": 11,
              "max_abs": 2.0384788513183594e-05,
              "mae": 4.634229298972059e-06,
              "rmse": 5.7348165682924446e-06,
              "relative_l2": 5.337782567949034e-06
            },
            {
              "time_index": 12,
              "max_abs": 2.2292137145996094e-05,
              "mae": 4.88137811771594e-06,
              "rmse": 6.0484776440716814e-06,
              "relative_l2": 5.16032696395996e-06
            },
            {
              "time_index": 13,
              "max_abs": 3.528594970703125e-05,
              "mae": 5.618301656795666e-06,
              "rmse": 7.083433501975378e-06,
              "relative_l2": 5.5789541875128634e-06
            },
            {
              "time_index": 14,
              "max_abs": 6.496906280517578e-05,
              "mae": 6.8476783781079575e-06,
              "rmse": 8.97012978384737e-06,
              "relative_l2": 6.561870577570517e-06
            },
            {
              "time_index": 15,
              "max_abs": 0.00017189979553222656,
              "mae": 1.0741641744971275e-05,
              "rmse": 1.6683679859852418e-05,
              "relative_l2": 1.1395715773687698e-05
            },
            {
              "time_index": 16,
              "max_abs": 0.0002410411834716797,
              "mae": 1.6439535102108493e-05,
              "rmse": 2.500477057765238e-05,
              "relative_l2": 1.602296470082365e-05
            },
            {
              "time_index": 17,
              "max_abs": 0.00035393238067626953,
              "mae": 2.4582634068792686e-05,
              "rmse": 4.0864058973966166e-05,
              "relative_l2": 2.4671364371897653e-05
            },
            {
              "time_index": 18,
              "max_abs": 0.0017604827880859375,
              "mae": 0.00014071605983190238,
              "rmse": 0.0002725055383052677,
              "relative_l2": 0.00015563127817586064
            },
            {
              "time_index": 19,
              "max_abs": 0.005814313888549805,
              "mae": 0.0004763022006954998,
              "rmse": 0.0009261975646950305,
              "relative_l2": 0.0005023426492698491
            },
            {
              "time_index": 20,
              "max_abs": 0.017209529876708984,
              "mae": 0.0019923776853829622,
              "rmse": 0.0034473706036806107,
              "relative_l2": 0.0017825650284066796
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2031_sample0.png"
        ]
      },
      {
        "condition_index": 6,
        "seed": 2032,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.98792454705108,
            "second_run_seconds": 5.6852532679913566,
            "memory": {
              "sample_count": 21,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.510116713005118,
            "second_run_seconds": 3.6771090138936415,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.04060244560241699,
          "mae": 0.00020590287749655545,
          "rmse": 0.0014107686001807451,
          "relative_l2": 0.0011754777515307069,
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
              "max_abs": 1.4230608940124512e-06,
              "mae": 2.3105886270968767e-07,
              "rmse": 2.898668753914535e-07,
              "relative_l2": 1.6282992874039337e-06
            },
            {
              "time_index": 2,
              "max_abs": 2.473592758178711e-06,
              "mae": 4.560180002499692e-07,
              "rmse": 5.709935066988692e-07,
              "relative_l2": 2.129916083504213e-06
            },
            {
              "time_index": 3,
              "max_abs": 3.7550926208496094e-06,
              "mae": 7.357180038525257e-07,
              "rmse": 9.127015232479607e-07,
              "relative_l2": 2.512849960112362e-06
            },
            {
              "time_index": 4,
              "max_abs": 6.258487701416016e-06,
              "mae": 1.0939011190203018e-06,
              "rmse": 1.3481377436619368e-06,
              "relative_l2": 2.929367610704503e-06
            },
            {
              "time_index": 5,
              "max_abs": 7.152557373046875e-06,
              "mae": 1.5488707276745117e-06,
              "rmse": 1.913864707603352e-06,
              "relative_l2": 3.429571961532929e-06
            },
            {
              "time_index": 6,
              "max_abs": 8.344650268554688e-06,
              "mae": 1.9175799934600946e-06,
              "rmse": 2.3446766590495827e-06,
              "relative_l2": 3.5726964142668294e-06
            },
            {
              "time_index": 7,
              "max_abs": 9.5367431640625e-06,
              "mae": 2.2238591554923914e-06,
              "rmse": 2.729155767156044e-06,
              "relative_l2": 3.6162889500701567e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.3828277587890625e-05,
              "mae": 2.708424290176481e-06,
              "rmse": 3.3345813790219836e-06,
              "relative_l2": 3.908538019459229e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.4066696166992188e-05,
              "mae": 3.21685865856125e-06,
              "rmse": 3.962196387874428e-06,
              "relative_l2": 4.163664925727062e-06
            },
            {
              "time_index": 10,
              "max_abs": 1.6927719116210938e-05,
              "mae": 3.6242186070012394e-06,
              "rmse": 4.4697171688312665e-06,
              "relative_l2": 4.256807187630329e-06
            },
            {
              "time_index": 11,
              "max_abs": 1.990795135498047e-05,
              "mae": 4.261079538991908e-06,
              "rmse": 5.266139851300977e-06,
              "relative_l2": 4.585976512316847e-06
            },
            {
              "time_index": 12,
              "max_abs": 4.780292510986328e-05,
              "mae": 7.908848601800855e-06,
              "rmse": 1.0784593541757204e-05,
              "relative_l2": 8.65233778313268e-06
            },
            {
              "time_index": 13,
              "max_abs": 0.00014340877532958984,
              "mae": 1.5454861568287015e-05,
              "rmse": 2.691496047191322e-05,
              "relative_l2": 2.0021812815684825e-05
            },
            {
              "time_index": 14,
              "max_abs": 0.0001537799835205078,
              "mae": 1.3264064364193473e-05,
              "rmse": 2.1172963897697628e-05,
              "relative_l2": 1.4685976566397585e-05
            },
            {
              "time_index": 15,
              "max_abs": 0.00038242340087890625,
              "mae": 3.288868538220413e-05,
              "rmse": 6.318197847576812e-05,
              "relative_l2": 4.1066217818297446e-05
            },
            {
              "time_index": 16,
              "max_abs": 0.0009069442749023438,
              "mae": 7.193048077169806e-05,
              "rmse": 0.00013139851216692477,
              "relative_l2": 8.03910443210043e-05
            },
            {
              "time_index": 17,
              "max_abs": 0.0016987323760986328,
              "mae": 0.00013563691754825413,
              "rmse": 0.00026924695703200996,
              "relative_l2": 0.00015571806579828262
            },
            {
              "time_index": 18,
              "max_abs": 0.003229856491088867,
              "mae": 0.00038854271406307817,
              "rmse": 0.0006591412238776684,
              "relative_l2": 0.00036186096258461475
            },
            {
              "time_index": 19,
              "max_abs": 0.013209819793701172,
              "mae": 0.0007431538542732596,
              "rmse": 0.0016344612231478095,
              "relative_l2": 0.0008552246727049351
            },
            {
              "time_index": 20,
              "max_abs": 0.04060244560241699,
              "mae": 0.002893162425607443,
              "rmse": 0.0062124524265527725,
              "relative_l2": 0.0031120881903916597
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2032_sample0.png"
        ]
      },
      {
        "condition_index": 7,
        "seed": 2033,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.750562057015486,
            "second_run_seconds": 5.49938412103802,
            "memory": {
              "sample_count": 20,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.433666631928645,
            "second_run_seconds": 3.703510414925404,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.021722793579101562,
          "mae": 0.00012042727030348033,
          "rmse": 0.0007508370908908546,
          "relative_l2": 0.000650654430501163,
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
              "max_abs": 1.5050172805786133e-06,
              "mae": 2.1177746134526387e-07,
              "rmse": 2.696228875720408e-07,
              "relative_l2": 1.7410731061318074e-06
            },
            {
              "time_index": 2,
              "max_abs": 2.205371856689453e-06,
              "mae": 3.6018755622535537e-07,
              "rmse": 4.521373568877607e-07,
              "relative_l2": 1.9731769498321228e-06
            },
            {
              "time_index": 3,
              "max_abs": 3.933906555175781e-06,
              "mae": 6.116865733929444e-07,
              "rmse": 7.663944643354625e-07,
              "relative_l2": 2.4126227344822837e-06
            },
            {
              "time_index": 4,
              "max_abs": 5.4836273193359375e-06,
              "mae": 9.160440868072328e-07,
              "rmse": 1.1402484005884617e-06,
              "relative_l2": 2.772699417619151e-06
            },
            {
              "time_index": 5,
              "max_abs": 6.973743438720703e-06,
              "mae": 1.3423393738776213e-06,
              "rmse": 1.6614606010989519e-06,
              "relative_l2": 3.276967163401423e-06
            },
            {
              "time_index": 6,
              "max_abs": 9.834766387939453e-06,
              "mae": 1.7949298580788309e-06,
              "rmse": 2.2217031983018387e-06,
              "relative_l2": 3.679229394037975e-06
            },
            {
              "time_index": 7,
              "max_abs": 9.775161743164062e-06,
              "mae": 2.0733309611387085e-06,
              "rmse": 2.5458275558776222e-06,
              "relative_l2": 3.6304868444858585e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.2874603271484375e-05,
              "mae": 2.5259062113036634e-06,
              "rmse": 3.1061852041602833e-06,
              "relative_l2": 3.888122137141181e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.4185905456542969e-05,
              "mae": 2.941874299722258e-06,
              "rmse": 3.611061629271717e-06,
              "relative_l2": 4.027255727123702e-06
            },
            {
              "time_index": 10,
              "max_abs": 1.8358230590820312e-05,
              "mae": 3.3308483580185566e-06,
              "rmse": 4.092777544428827e-06,
              "relative_l2": 4.115697265660856e-06
            },
            {
              "time_index": 11,
              "max_abs": 2.849102020263672e-05,
              "mae": 4.256011834513629e-06,
              "rmse": 5.400871032179566e-06,
              "relative_l2": 4.94515825266717e-06
            },
            {
              "time_index": 12,
              "max_abs": 7.665157318115234e-05,
              "mae": 7.368360456894152e-06,
              "rmse": 1.1151777471241076e-05,
              "relative_l2": 9.372979548061267e-06
            },
            {
              "time_index": 13,
              "max_abs": 0.00010633468627929688,
              "mae": 9.12336508918088e-06,
              "rmse": 1.326153414993314e-05,
              "relative_l2": 1.0302132068318315e-05
            },
            {
              "time_index": 14,
              "max_abs": 0.0001341104507446289,
              "mae": 1.311014511884423e-05,
              "rmse": 1.9280612832517363e-05,
              "relative_l2": 1.3925733583164401e-05
            },
            {
              "time_index": 15,
              "max_abs": 0.00025594234466552734,
              "mae": 2.224873605882749e-05,
              "rmse": 3.6481567804003134e-05,
              "relative_l2": 2.4625016521895304e-05
            },
            {
              "time_index": 16,
              "max_abs": 0.00034356117248535156,
              "mae": 2.26869378820993e-05,
              "rmse": 3.600777927204035e-05,
              "relative_l2": 2.281947308802046e-05
            },
            {
              "time_index": 17,
              "max_abs": 0.0005528926849365234,
              "mae": 3.251961970818229e-05,
              "rmse": 5.617964779958129e-05,
              "relative_l2": 3.356778324814513e-05
            },
            {
              "time_index": 18,
              "max_abs": 0.002205371856689453,
              "mae": 0.00012719372170977294,
              "rmse": 0.0002528931072447449,
              "relative_l2": 0.00014303340867627412
            },
            {
              "time_index": 19,
              "max_abs": 0.005005359649658203,
              "mae": 0.00044823403004556894,
              "rmse": 0.0008167516789399087,
              "relative_l2": 0.00043899554293602705
            },
            {
              "time_index": 20,
              "max_abs": 0.021722793579101562,
              "mae": 0.00182612263597548,
              "rmse": 0.0033318621572107077,
              "relative_l2": 0.0017087471205741167
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2033_sample0.png"
        ]
      },
      {
        "condition_index": 8,
        "seed": 2034,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.924811311997473,
            "second_run_seconds": 5.673146925983019,
            "memory": {
              "sample_count": 21,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.517768289078958,
            "second_run_seconds": 3.6814221210079268,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.006269931793212891,
          "mae": 7.514281605836004e-05,
          "rmse": 0.00028905668295919895,
          "relative_l2": 0.0002549085475038737,
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
              "max_abs": 9.611248970031738e-07,
              "mae": 1.8437127380366292e-07,
              "rmse": 2.3235560320244986e-07,
              "relative_l2": 1.8905501519839163e-06
            },
            {
              "time_index": 2,
              "max_abs": 1.9073486328125e-06,
              "mae": 2.999320827257179e-07,
              "rmse": 3.7458329416040215e-07,
              "relative_l2": 1.9121746390737826e-06
            },
            {
              "time_index": 3,
              "max_abs": 3.1888484954833984e-06,
              "mae": 5.23145558872784e-07,
              "rmse": 6.560322276527586e-07,
              "relative_l2": 2.297236505910405e-06
            },
            {
              "time_index": 4,
              "max_abs": 4.112720489501953e-06,
              "mae": 7.56307656502031e-07,
              "rmse": 9.463551577937324e-07,
              "relative_l2": 2.488831114533241e-06
            },
            {
              "time_index": 5,
              "max_abs": 6.020069122314453e-06,
              "mae": 1.1736944998119725e-06,
              "rmse": 1.4659799489891157e-06,
              "relative_l2": 3.074219648624421e-06
            },
            {
              "time_index": 6,
              "max_abs": 7.271766662597656e-06,
              "mae": 1.5312108416765113e-06,
              "rmse": 1.8955208815896185e-06,
              "relative_l2": 3.3000960684148595e-06
            },
            {
              "time_index": 7,
              "max_abs": 9.357929229736328e-06,
              "mae": 1.8750274648482446e-06,
              "rmse": 2.2845504190627253e-06,
              "relative_l2": 3.397885620870511e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.1205673217773438e-05,
              "mae": 2.2880178676132346e-06,
              "rmse": 2.8067934181308374e-06,
              "relative_l2": 3.6427222767088097e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.52587890625e-05,
              "mae": 3.076401299040299e-06,
              "rmse": 3.796059900196269e-06,
              "relative_l2": 4.369368525658501e-06
            },
            {
              "time_index": 10,
              "max_abs": 1.633167266845703e-05,
              "mae": 3.642433057393646e-06,
              "rmse": 4.5109400161891244e-06,
              "relative_l2": 4.664474545279518e-06
            },
            {
              "time_index": 11,
              "max_abs": 4.100799560546875e-05,
              "mae": 5.153099664312322e-06,
              "rmse": 7.042104243737413e-06,
              "relative_l2": 6.6100319600082e-06
            },
            {
              "time_index": 12,
              "max_abs": 0.00015401840209960938,
              "mae": 1.2821898053516634e-05,
              "rmse": 2.292037424922455e-05,
              "relative_l2": 1.9697581592481583e-05
            },
            {
              "time_index": 13,
              "max_abs": 0.0006316900253295898,
              "mae": 5.053877248428762e-05,
              "rmse": 0.00010127169662155211,
              "relative_l2": 8.026049908949062e-05
            },
            {
              "time_index": 14,
              "max_abs": 0.0006922483444213867,
              "mae": 7.416163862217218e-05,
              "rmse": 0.00011887420987477526,
              "relative_l2": 8.741662895772606e-05
            },
            {
              "time_index": 15,
              "max_abs": 0.0008205175399780273,
              "mae": 8.934833022067323e-05,
              "rmse": 0.0001520428922958672,
              "relative_l2": 0.00010429972462588921
            },
            {
              "time_index": 16,
              "max_abs": 0.001184701919555664,
              "mae": 0.00011769971752073616,
              "rmse": 0.00019879956380464137,
              "relative_l2": 0.0001278156996704638
            },
            {
              "time_index": 17,
              "max_abs": 0.001961827278137207,
              "mae": 0.00018491913215257227,
              "rmse": 0.00037676264764741063,
              "relative_l2": 0.000227991899009794
            },
            {
              "time_index": 18,
              "max_abs": 0.0025534629821777344,
              "mae": 0.00020637773559428751,
              "rmse": 0.0004016182792838663,
              "relative_l2": 0.00022963246738072485
            },
            {
              "time_index": 19,
              "max_abs": 0.003057718276977539,
              "mae": 0.00028430551174096763,
              "rmse": 0.0005115088424645364,
              "relative_l2": 0.0002773670421447605
            },
            {
              "time_index": 20,
              "max_abs": 0.006269931793212891,
              "mae": 0.0005373228341341019,
              "rmse": 0.0010498022893443704,
              "relative_l2": 0.0005418917862698436
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2034_sample0.png"
        ]
      },
      {
        "condition_index": 9,
        "seed": 2035,
        "runs": {
          "pytorch_solvers_py": {
            "case": "ns_2d",
            "framework": "pytorch_solvers_py",
            "backend": {
              "torch_cuda_available": 1,
              "torch_device": "NVIDIA A100-SXM4-80GB",
              "device_used": "cuda"
            },
            "output_shape": [
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 5.822474800050259,
            "second_run_seconds": 5.700148008996621,
            "memory": {
              "sample_count": 21,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13591.0,
              "sampled_global_gpu_peak_delta_mib": 64.0,
              "torch_peak_allocated_mib": 33.8232421875,
              "torch_peak_reserved_mib": 58.0
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
              1,
              256,
              256,
              21
            ],
            "first_run_seconds": 4.59606063994579,
            "second_run_seconds": 3.7112507390556857,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13547.0,
              "sampled_global_gpu_peak_delta_mib": 46.0
            }
          }
        },
        "comparison": {
          "max_abs": 0.012409687042236328,
          "mae": 0.00010532473243074492,
          "rmse": 0.0004459177143871784,
          "relative_l2": 0.00037433375837281346,
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
              "max_abs": 1.5497207641601562e-06,
              "mae": 2.364822364597785e-07,
              "rmse": 3.0122149041744706e-07,
              "relative_l2": 1.695884634500544e-06
            },
            {
              "time_index": 2,
              "max_abs": 2.3543834686279297e-06,
              "mae": 4.741648638173501e-07,
              "rmse": 5.938812250860792e-07,
              "relative_l2": 2.2686731426802e-06
            },
            {
              "time_index": 3,
              "max_abs": 4.410743713378906e-06,
              "mae": 7.485274409191334e-07,
              "rmse": 9.402363048138795e-07,
              "relative_l2": 2.6555665044725174e-06
            },
            {
              "time_index": 4,
              "max_abs": 5.602836608886719e-06,
              "mae": 1.0083588222187245e-06,
              "rmse": 1.2574486163430265e-06,
              "relative_l2": 2.7980568120256066e-06
            },
            {
              "time_index": 5,
              "max_abs": 8.821487426757812e-06,
              "mae": 1.4970463553254376e-06,
              "rmse": 1.8616656234371476e-06,
              "relative_l2": 3.408990323805483e-06
            },
            {
              "time_index": 6,
              "max_abs": 9.059906005859375e-06,
              "mae": 1.8151682752431952e-06,
              "rmse": 2.2545482352143154e-06,
              "relative_l2": 3.503686230033054e-06
            },
            {
              "time_index": 7,
              "max_abs": 1.1801719665527344e-05,
              "mae": 2.2368933514371747e-06,
              "rmse": 2.783566515063285e-06,
              "relative_l2": 3.755454144993564e-06
            },
            {
              "time_index": 8,
              "max_abs": 1.4185905456542969e-05,
              "mae": 2.6493071345612407e-06,
              "rmse": 3.2955069855233887e-06,
              "relative_l2": 3.92735501009156e-06
            },
            {
              "time_index": 9,
              "max_abs": 1.4662742614746094e-05,
              "mae": 3.005585085702478e-06,
              "rmse": 3.7123643323866418e-06,
              "relative_l2": 3.96142786485143e-06
            },
            {
              "time_index": 10,
              "max_abs": 3.325939178466797e-05,
              "mae": 4.4180605982546695e-06,
              "rmse": 5.7532092796463985e-06,
              "relative_l2": 5.557617896556621e-06
            },
            {
              "time_index": 11,
              "max_abs": 0.00012028217315673828,
              "mae": 1.311461346631404e-05,
              "rmse": 2.1142321202205494e-05,
              "relative_l2": 1.8656175598152913e-05
            },
            {
              "time_index": 12,
              "max_abs": 0.00025653839111328125,
              "mae": 3.1089333788258955e-05,
              "rmse": 4.821050970349461e-05,
              "relative_l2": 3.915464549208991e-05
            },
            {
              "time_index": 13,
              "max_abs": 0.000754237174987793,
              "mae": 7.744575850665569e-05,
              "rmse": 0.0001210240661748685,
              "relative_l2": 9.104889613809064e-05
            },
            {
              "time_index": 14,
              "max_abs": 0.0011059045791625977,
              "mae": 0.00010316708358004689,
              "rmse": 0.00015786945004947484,
              "relative_l2": 0.00011062990961363539
            },
            {
              "time_index": 15,
              "max_abs": 0.0008199214935302734,
              "mae": 0.00010653890785761178,
              "rmse": 0.00015920934674795717,
              "relative_l2": 0.00010442906204843894
            },
            {
              "time_index": 16,
              "max_abs": 0.0010999441146850586,
              "mae": 0.00010919115447904915,
              "rmse": 0.00016331356891896576,
              "relative_l2": 0.00010069851850857958
            },
            {
              "time_index": 17,
              "max_abs": 0.0020155906677246094,
              "mae": 0.0001592208573129028,
              "rmse": 0.00025678836391307414,
              "relative_l2": 0.00014942660345695913
            },
            {
              "time_index": 18,
              "max_abs": 0.0029697418212890625,
              "mae": 0.00023640147992409766,
              "rmse": 0.00039042584830895066,
              "relative_l2": 0.00021520773589145392
            },
            {
              "time_index": 19,
              "max_abs": 0.007318019866943359,
              "mae": 0.0005466327420435846,
              "rmse": 0.0011030593886971474,
              "relative_l2": 0.0005780693027190864
            },
            {
              "time_index": 20,
              "max_abs": 0.012409687042236328,
              "mae": 0.0008109278278425336,
              "rmse": 0.001626697601750493,
              "relative_l2": 0.0008135737734846771
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_full_tfinal_multi10/ns_2d/ns_seed2035_sample0.png"
        ]
      }
    ],
    "statistics": {
      "comparison": {
        "max_abs": {
          "count": 10,
          "mean": 0.03260481357574463,
          "std": 0.03165087315441495,
          "min": 0.0013551712036132812,
          "max": 0.10175776481628418
        },
        "mae": {
          "count": 10,
          "mean": 0.00020795745567738776,
          "std": 0.00015429999658222115,
          "min": 1.577849434397649e-05,
          "max": 0.0005000843084417284
        },
        "rmse": {
          "count": 10,
          "mean": 0.0011383054454199737,
          "std": 0.0009547062656185571,
          "min": 5.235729258856736e-05,
          "max": 0.002950145862996578
        },
        "relative_l2": {
          "count": 10,
          "mean": 0.0009607211952243234,
          "std": 0.0007974205755424086,
          "min": 4.397787051857449e-05,
          "max": 0.002542247762903571
        }
      },
      "runs": {
        "pytorch_solvers_py": {
          "first_run_seconds": {
            "count": 10,
            "mean": 5.847254904813599,
            "std": 0.12974585545432094,
            "min": 5.721714852028526,
            "max": 6.034176251967438
          },
          "second_run_seconds": {
            "count": 10,
            "mean": 5.594920262706,
            "std": 0.1538666214957685,
            "min": 5.4516088790260255,
            "max": 5.941829425981268
          },
          "sampled_process_gpu_peak_mib": {
            "count": 10,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 10,
            "mean": 64.0,
            "std": 0.0,
            "min": 64.0,
            "max": 64.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 10,
            "mean": 13591.0,
            "std": 0.0,
            "min": 13591.0,
            "max": 13591.0
          },
          "torch_peak_allocated_mib": {
            "count": 10,
            "mean": 33.8232421875,
            "std": 0.0,
            "min": 33.8232421875,
            "max": 33.8232421875
          },
          "torch_peak_reserved_mib": {
            "count": 10,
            "mean": 58.0,
            "std": 0.0,
            "min": 58.0,
            "max": 58.0
          }
        },
        "jax_exponax": {
          "first_run_seconds": {
            "count": 10,
            "mean": 4.533258879987988,
            "std": 0.05568986092523811,
            "min": 4.433666631928645,
            "max": 4.603050834964961
          },
          "second_run_seconds": {
            "count": 10,
            "mean": 3.688995548384264,
            "std": 0.01726264883192882,
            "min": 3.664630846004002,
            "max": 3.712838766979985
          },
          "sampled_process_gpu_peak_mib": {
            "count": 10,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 10,
            "mean": 45.6,
            "std": 1.2649110640673518,
            "min": 42.0,
            "max": 46.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 10,
            "mean": 13547.0,
            "std": 0.0,
            "min": 13547.0,
            "max": 13547.0
          },
          "torch_peak_allocated_mib": {
            "count": 0,
            "mean": null,
            "std": null,
            "min": null,
            "max": null
          },
          "torch_peak_reserved_mib": {
            "count": 0,
            "mean": null,
            "std": null,
            "min": null,
            "max": null
          }
        }
      },
      "paired_tests_pytorch_minus_jax": {
        "first_run_seconds": {
          "count": 10,
          "mean_left_minus_right": 1.3139960248256102,
          "t_statistic": 28.711492094299494,
          "p_value": 3.6734623736511654e-10
        },
        "second_run_seconds": {
          "count": 10,
          "mean_left_minus_right": 1.9059247143217362,
          "t_statistic": 37.206229139249885,
          "p_value": 3.62924177616094e-11
        },
        "sampled_process_gpu_peak_mib": {
          "count": 10,
          "mean_left_minus_right": 0.0,
          "t_statistic": 0.0,
          "p_value": 1.0
        },
        "sampled_global_gpu_peak_delta_mib": {
          "count": 10,
          "mean_left_minus_right": 18.4,
          "t_statistic": 45.99999999999999,
          "p_value": 5.426370812324831e-12
        },
        "sampled_global_gpu_peak_mib": {
          "count": 10,
          "mean_left_minus_right": 44.0,
          "t_statistic": null,
          "p_value": 0.0
        }
      }
    }
  }
}
```
