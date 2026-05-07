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
    2030
  ],
  "num_conditions": 5,
  "plot_dir": "benchmark_results/torch_vs_exponax_solver_visualizations_multi5",
  "max_visualized_samples_per_seed": 1,
  "burgers": {
    "batch": 1,
    "nx": 256,
    "t_final": 0.1,
    "dt": 0.001,
    "nu": 0.001,
    "domain_extent": 2.0
  },
  "ns": {
    "batch": 1,
    "nx": 64,
    "t_final": 1,
    "dt": 0.005,
    "nu": 1e-05,
    "domain_extent": 1.0
  }
}
```

## Error Summary Across Conditions

| case | conditions | relative L2 mean | relative L2 std | max abs mean | max abs max |
|---|---:|---:|---:|---:|---:|
| burgers_1d | 5 | 8.76936e-07 | 3.48161e-08 | 6.08712e-07 | 1.01328e-06 |
| ns_2d | 5 | 2.07652e-06 | 1.49914e-07 | 2.23815e-06 | 2.5332e-06 |

## Runtime And Memory Summary

`p_value` is from a paired t-test across the repeated initial conditions. The paired test is `PyTorch - Exponax/JAX`, so a negative mean difference means PyTorch was smaller/faster.

| case | metric | PyTorch mean | PyTorch std | Exponax/JAX mean | Exponax/JAX std | mean PyTorch-JAX | paired p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| burgers_1d | cold seconds | 0.181547 | 0.0142117 | 0.324792 | 0.0239004 | -0.143245 | 0.000300302 |
| burgers_1d | warm seconds | 0.0656167 | 0.00301236 | 0.0263265 | 0.00186119 | 0.0392902 | 1.6824e-05 |
| burgers_1d | global GPU delta MiB | 8 | 0 | 20 | 0 | -12 | 0 |
| burgers_1d | process GPU peak MiB | 0 | 0 | 0 | 0 | 0 | 1 |
| ns_2d | cold seconds | 0.509029 | 0.0317351 | 0.895446 | 0.0597711 | -0.386417 | 0.000139069 |
| ns_2d | warm seconds | 0.289929 | 0.0200805 | 0.176344 | 0.000443733 | 0.113585 | 0.000212029 |
| ns_2d | global GPU delta MiB | 10 | 0 | 12 | 0 | -2 | 0 |
| ns_2d | process GPU peak MiB | 0 | 0 | 0 | 0 | 0 | 1 |

## Visualization Files

- Plot directory: `benchmark_results/torch_vs_exponax_solver_visualizations_multi5`
- `burgers_1d` plots: 5
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2026_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2027_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2028_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2029_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2030_sample0.png`
- `ns_2d` plots: 5
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2026_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2027_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2028_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2029_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2030_sample0.png`

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
              256
            ],
            "first_run_seconds": 0.17859767109621316,
            "second_run_seconds": 0.06310996599495411,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.1220703125,
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
              256
            ],
            "first_run_seconds": 0.3658235518960282,
            "second_run_seconds": 0.027161168050952256,
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
          "mae": 2.1116284187883139e-07,
          "rmse": 2.6711853706729016e-07,
          "relative_l2": 8.614076136836957e-07
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2026_sample0.png"
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
              256
            ],
            "first_run_seconds": 0.20477809000294656,
            "second_run_seconds": 0.07039361703209579,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.1220703125,
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
              256
            ],
            "first_run_seconds": 0.3167989390203729,
            "second_run_seconds": 0.02715876791626215,
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
          "max_abs": 4.172325134277344e-07,
          "mae": 1.0446819942444563e-07,
          "rmse": 1.314821531650523e-07,
          "relative_l2": 9.175710147246718e-07
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2027_sample0.png"
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
              256
            ],
            "first_run_seconds": 0.16646870505064726,
            "second_run_seconds": 0.06294916709885001,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.1220703125,
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
              256
            ],
            "first_run_seconds": 0.30325482599437237,
            "second_run_seconds": 0.027185008046217263,
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
          "max_abs": 6.556510925292969e-07,
          "mae": 1.5405930753331631e-07,
          "rmse": 1.909618845274963e-07,
          "relative_l2": 9.082848464458948e-07
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2028_sample0.png"
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
              256
            ],
            "first_run_seconds": 0.17589433502871543,
            "second_run_seconds": 0.06581771292258054,
            "memory": {
              "sample_count": 4,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.1220703125,
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
              256
            ],
            "first_run_seconds": 0.31710957002360374,
            "second_run_seconds": 0.02299730305094272,
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
          "max_abs": 3.6135315895080566e-07,
          "mae": 8.704955689609051e-08,
          "rmse": 1.1522929810325877e-07,
          "relative_l2": 8.624887186670094e-07
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2029_sample0.png"
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
              256
            ],
            "first_run_seconds": 0.1819954669335857,
            "second_run_seconds": 0.06581308401655406,
            "memory": {
              "sample_count": 3,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13535.0,
              "sampled_global_gpu_peak_delta_mib": 8.0,
              "torch_peak_allocated_mib": 0.1220703125,
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
              256
            ],
            "first_run_seconds": 0.32097388594411314,
            "second_run_seconds": 0.0271303589688614,
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
          "max_abs": 5.960464477539062e-07,
          "mae": 1.6242120182141662e-07,
          "rmse": 2.061899948557766e-07,
          "relative_l2": 8.349261975126865e-07
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/burgers_1d/burgers_seed2030_sample0.png"
        ]
      }
    ],
    "statistics": {
      "comparison": {
        "max_abs": {
          "count": 5,
          "mean": 6.087124347686768e-07,
          "std": 2.5685059408299675e-07,
          "min": 3.6135315895080566e-07,
          "max": 1.0132789611816406e-06
        },
        "mae": {
          "count": 5,
          "mean": 1.438322215108201e-07,
          "std": 4.938965770332383e-08,
          "min": 8.704955689609051e-08,
          "max": 2.1116284187883139e-07
        },
        "rmse": {
          "count": 5,
          "mean": 1.8219637354377483e-07,
          "std": 6.107541085752426e-08,
          "min": 1.1522929810325877e-07,
          "max": 2.6711853706729016e-07
        },
        "relative_l2": {
          "count": 5,
          "mean": 8.769356782067916e-07,
          "std": 3.48160624538836e-08,
          "min": 8.349261975126865e-07,
          "max": 9.175710147246718e-07
        }
      },
      "runs": {
        "pytorch_solvers_py": {
          "first_run_seconds": {
            "count": 5,
            "mean": 0.18154685362242162,
            "std": 0.01421171001365043,
            "min": 0.16646870505064726,
            "max": 0.20477809000294656
          },
          "second_run_seconds": {
            "count": 5,
            "mean": 0.0656167094130069,
            "std": 0.003012364368683559,
            "min": 0.06294916709885001,
            "max": 0.07039361703209579
          },
          "sampled_process_gpu_peak_mib": {
            "count": 5,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 5,
            "mean": 8.0,
            "std": 0.0,
            "min": 8.0,
            "max": 8.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 5,
            "mean": 13535.0,
            "std": 0.0,
            "min": 13535.0,
            "max": 13535.0
          },
          "torch_peak_allocated_mib": {
            "count": 5,
            "mean": 0.1220703125,
            "std": 0.0,
            "min": 0.1220703125,
            "max": 0.1220703125
          },
          "torch_peak_reserved_mib": {
            "count": 5,
            "mean": 2.0,
            "std": 0.0,
            "min": 2.0,
            "max": 2.0
          }
        },
        "jax_exponax": {
          "first_run_seconds": {
            "count": 5,
            "mean": 0.3247921545756981,
            "std": 0.023900443845960795,
            "min": 0.30325482599437237,
            "max": 0.3658235518960282
          },
          "second_run_seconds": {
            "count": 5,
            "mean": 0.026326521206647156,
            "std": 0.0018611903630689,
            "min": 0.02299730305094272,
            "max": 0.027185008046217263
          },
          "sampled_process_gpu_peak_mib": {
            "count": 5,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 5,
            "mean": 20.0,
            "std": 0.0,
            "min": 20.0,
            "max": 20.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 5,
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
          "count": 5,
          "mean_left_minus_right": -0.14324530095327645,
          "t_statistic": -11.748314380304217,
          "p_value": 0.00030030182671238724
        },
        "second_run_seconds": {
          "count": 5,
          "mean_left_minus_right": 0.039290188206359745,
          "t_statistic": 24.369178152076763,
          "p_value": 1.682395017721046e-05
        },
        "sampled_process_gpu_peak_mib": {
          "count": 5,
          "mean_left_minus_right": 0.0,
          "t_statistic": 0.0,
          "p_value": 1.0
        },
        "sampled_global_gpu_peak_delta_mib": {
          "count": 5,
          "mean_left_minus_right": -12.0,
          "t_statistic": null,
          "p_value": 0.0
        },
        "sampled_global_gpu_peak_mib": {
          "count": 5,
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
              64,
              64,
              2
            ],
            "first_run_seconds": 0.5481275579659268,
            "second_run_seconds": 0.324506948934868,
            "memory": {
              "sample_count": 10,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13537.0,
              "sampled_global_gpu_peak_delta_mib": 10.0,
              "torch_peak_allocated_mib": 1.86376953125,
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
              1,
              64,
              64,
              2
            ],
            "first_run_seconds": 0.981583061045967,
            "second_run_seconds": 0.17689563299063593,
            "memory": {
              "sample_count": 11,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13513.0,
              "sampled_global_gpu_peak_delta_mib": 12.0
            }
          }
        },
        "comparison": {
          "max_abs": 2.3245811462402344e-06,
          "mae": 2.2786360887039336e-07,
          "rmse": 4.066379801770381e-07,
          "relative_l2": 2.287972847625497e-06,
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
              "max_abs": 2.3245811462402344e-06,
              "mae": 4.5572721774078673e-07,
              "rmse": 5.750729314968339e-07,
              "relative_l2": 2.807229748214013e-06
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2026_sample0.png"
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
              64,
              64,
              2
            ],
            "first_run_seconds": 0.4779102810425684,
            "second_run_seconds": 0.2878903830423951,
            "memory": {
              "sample_count": 8,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13537.0,
              "sampled_global_gpu_peak_delta_mib": 10.0,
              "torch_peak_allocated_mib": 1.86376953125,
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
              1,
              64,
              64,
              2
            ],
            "first_run_seconds": 0.8775086940731853,
            "second_run_seconds": 0.1763000440550968,
            "memory": {
              "sample_count": 10,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13513.0,
              "sampled_global_gpu_peak_delta_mib": 12.0
            }
          }
        },
        "comparison": {
          "max_abs": 2.4884939193725586e-06,
          "mae": 1.5676306475143065e-07,
          "rmse": 2.849789666470315e-07,
          "relative_l2": 1.9552703633962665e-06,
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
              "max_abs": 2.4884939193725586e-06,
              "mae": 3.135261295028613e-07,
              "rmse": 4.030211471217626e-07,
              "relative_l2": 2.4493015189364087e-06
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2027_sample0.png"
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
              64,
              64,
              2
            ],
            "first_run_seconds": 0.4832154450705275,
            "second_run_seconds": 0.2778302460210398,
            "memory": {
              "sample_count": 8,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13537.0,
              "sampled_global_gpu_peak_delta_mib": 10.0,
              "torch_peak_allocated_mib": 1.86376953125,
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
              1,
              64,
              64,
              2
            ],
            "first_run_seconds": 0.9298405040754005,
            "second_run_seconds": 0.17567751510068774,
            "memory": {
              "sample_count": 13,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13513.0,
              "sampled_global_gpu_peak_delta_mib": 12.0
            }
          }
        },
        "comparison": {
          "max_abs": 2.5331974029541016e-06,
          "mae": 2.2558424461749382e-07,
          "rmse": 4.0424296798846626e-07,
          "relative_l2": 2.0881682303297566e-06,
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
              "max_abs": 2.5331974029541016e-06,
              "mae": 4.5116848923498765e-07,
              "rmse": 5.716858595405938e-07,
              "relative_l2": 2.4828043478919426e-06
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2028_sample0.png"
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
              64,
              64,
              2
            ],
            "first_run_seconds": 0.4991570180281997,
            "second_run_seconds": 0.27433208003640175,
            "memory": {
              "sample_count": 9,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13537.0,
              "sampled_global_gpu_peak_delta_mib": 10.0,
              "torch_peak_allocated_mib": 1.86376953125,
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
              1,
              64,
              64,
              2
            ],
            "first_run_seconds": 0.8513253768905997,
            "second_run_seconds": 0.1765339740086347,
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
          "max_abs": 1.5497207641601562e-06,
          "mae": 1.5770561390127114e-07,
          "rmse": 2.8092216552977334e-07,
          "relative_l2": 2.137755018338794e-06,
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
              "mae": 3.1541122780254227e-07,
              "rmse": 3.9728394085614127e-07,
              "relative_l2": 2.4871251298463903e-06
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2029_sample0.png"
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
              64,
              64,
              2
            ],
            "first_run_seconds": 0.5367347849532962,
            "second_run_seconds": 0.28508386795874685,
            "memory": {
              "sample_count": 8,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13527.0,
              "sampled_global_gpu_peak_mib": 13537.0,
              "sampled_global_gpu_peak_delta_mib": 10.0,
              "torch_peak_allocated_mib": 1.86376953125,
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
              1,
              64,
              64,
              2
            ],
            "first_run_seconds": 0.8369703060016036,
            "second_run_seconds": 0.1763112440239638,
            "memory": {
              "sample_count": 10,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 13501.0,
              "sampled_global_gpu_peak_mib": 13513.0,
              "sampled_global_gpu_peak_delta_mib": 12.0
            }
          }
        },
        "comparison": {
          "max_abs": 2.294778823852539e-06,
          "mae": 2.1497091040600935e-07,
          "rmse": 3.8464645513158757e-07,
          "relative_l2": 1.9134324702463346e-06,
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
              "max_abs": 2.294778823852539e-06,
              "mae": 4.299418208120187e-07,
              "rmse": 5.4397219173552e-07,
              "relative_l2": 2.689531811483903e-06
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_multi5/ns_2d/ns_seed2030_sample0.png"
        ]
      }
    ],
    "statistics": {
      "comparison": {
        "max_abs": {
          "count": 5,
          "mean": 2.238154411315918e-06,
          "std": 3.9822638892652176e-07,
          "min": 1.5497207641601562e-06,
          "max": 2.5331974029541016e-06
        },
        "mae": {
          "count": 5,
          "mean": 1.9657748850931966e-07,
          "std": 3.6244800216558366e-08,
          "min": 1.5676306475143065e-07,
          "max": 2.2786360887039336e-07
        },
        "rmse": {
          "count": 5,
          "mean": 3.5228570709477937e-07,
          "std": 6.388250334928513e-08,
          "min": 2.8092216552977334e-07,
          "max": 4.066379801770381e-07
        },
        "relative_l2": {
          "count": 5,
          "mean": 2.0765197859873296e-06,
          "std": 1.499139188394128e-07,
          "min": 1.9134324702463346e-06,
          "max": 2.287972847625497e-06
        }
      },
      "runs": {
        "pytorch_solvers_py": {
          "first_run_seconds": {
            "count": 5,
            "mean": 0.5090290174121037,
            "std": 0.03173514640974489,
            "min": 0.4779102810425684,
            "max": 0.5481275579659268
          },
          "second_run_seconds": {
            "count": 5,
            "mean": 0.2899287051986903,
            "std": 0.020080522751531556,
            "min": 0.27433208003640175,
            "max": 0.324506948934868
          },
          "sampled_process_gpu_peak_mib": {
            "count": 5,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 5,
            "mean": 10.0,
            "std": 0.0,
            "min": 10.0,
            "max": 10.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 5,
            "mean": 13537.0,
            "std": 0.0,
            "min": 13537.0,
            "max": 13537.0
          },
          "torch_peak_allocated_mib": {
            "count": 5,
            "mean": 1.86376953125,
            "std": 0.0,
            "min": 1.86376953125,
            "max": 1.86376953125
          },
          "torch_peak_reserved_mib": {
            "count": 5,
            "mean": 4.0,
            "std": 0.0,
            "min": 4.0,
            "max": 4.0
          }
        },
        "jax_exponax": {
          "first_run_seconds": {
            "count": 5,
            "mean": 0.8954455884173512,
            "std": 0.05977114335415203,
            "min": 0.8369703060016036,
            "max": 0.981583061045967
          },
          "second_run_seconds": {
            "count": 5,
            "mean": 0.1763436820358038,
            "std": 0.0004437329644274273,
            "min": 0.17567751510068774,
            "max": 0.17689563299063593
          },
          "sampled_process_gpu_peak_mib": {
            "count": 5,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
          },
          "sampled_global_gpu_peak_delta_mib": {
            "count": 5,
            "mean": 12.0,
            "std": 0.0,
            "min": 12.0,
            "max": 12.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 5,
            "mean": 13513.0,
            "std": 0.0,
            "min": 13513.0,
            "max": 13513.0
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
          "count": 5,
          "mean_left_minus_right": -0.3864165710052475,
          "t_statistic": -14.29624195195011,
          "p_value": 0.00013906850828476611
        },
        "second_run_seconds": {
          "count": 5,
          "mean_left_minus_right": 0.1135850231628865,
          "t_statistic": 12.841023092039878,
          "p_value": 0.00021202882278123467
        },
        "sampled_process_gpu_peak_mib": {
          "count": 5,
          "mean_left_minus_right": 0.0,
          "t_statistic": 0.0,
          "p_value": 1.0
        },
        "sampled_global_gpu_peak_delta_mib": {
          "count": 5,
          "mean_left_minus_right": -2.0,
          "t_statistic": null,
          "p_value": 0.0
        },
        "sampled_global_gpu_peak_mib": {
          "count": 5,
          "mean_left_minus_right": 24.0,
          "t_statistic": null,
          "p_value": 0.0
        }
      }
    }
  }
}
```
