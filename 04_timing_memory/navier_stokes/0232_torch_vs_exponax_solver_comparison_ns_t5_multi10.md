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
  "plot_dir": "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10",
  "max_visualized_samples_per_seed": 1,
  "burgers": {
    "batch": 4,
    "nx": 256,
    "t_final": 0.1,
    "dt": 0.001,
    "nu": 0.001,
    "domain_extent": 2.0
  },
  "ns": {
    "batch": 1,
    "nx": 256,
    "t_final": 5,
    "dt": 0.005,
    "nu": 1e-05,
    "domain_extent": 1.0
  }
}
```

## Error Summary Across Conditions

| case | conditions | relative L2 mean | relative L2 std | max abs mean | max abs max |
|---|---:|---:|---:|---:|---:|
| ns_2d | 10 | 2.83019e-06 | 1.30342e-07 | 7.20024e-06 | 8.82149e-06 |

## Runtime And Memory Summary

`p_value` is from a paired t-test across the repeated initial conditions. The paired test is `PyTorch - Exponax/JAX`, so a negative mean difference means PyTorch was smaller/faster.

| case | metric | PyTorch mean | PyTorch std | Exponax/JAX mean | Exponax/JAX std | mean PyTorch-JAX | paired p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| ns_2d | cold seconds | 1.51588 | 0.0366539 | 1.20852 | 0.0453401 | 0.307364 | 4.7346e-10 |
| ns_2d | warm seconds | 1.32809 | 0.044262 | 0.455068 | 0.00605931 | 0.87302 | 3.66261e-13 |
| ns_2d | global GPU delta MiB | 60 | 0 | 13.6 | 1.26491 | 46.4 | 1.33525e-15 |
| ns_2d | process GPU peak MiB | 0 | 0 | 0 | 0 | 0 | 1 |

## Visualization Files

- Plot directory: `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10`
- `ns_2d` plots: 10
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2026_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2027_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2028_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2029_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2030_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2031_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2032_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2033_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2034_sample0.png`
  - `benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2035_sample0.png`

## Full Run Details

```json
{
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
              6
            ],
            "first_run_seconds": 1.5788980820216238,
            "second_run_seconds": 1.3911698709707707,
            "memory": {
              "sample_count": 25,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.2558577780146152,
            "second_run_seconds": 0.4537961418973282,
            "memory": {
              "sample_count": 14,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 8.463859558105469e-06,
          "mae": 7.086129585331946e-07,
          "rmse": 1.0824388709806954e-06,
          "relative_l2": 2.9004281714151148e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2026_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.5291766909649596,
            "second_run_seconds": 1.3237861800007522,
            "memory": {
              "sample_count": 24,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.2040645299712196,
            "second_run_seconds": 0.4722066599642858,
            "memory": {
              "sample_count": 14,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 6.67572021484375e-06,
          "mae": 5.971401151327882e-07,
          "rmse": 9.403329954693618e-07,
          "relative_l2": 2.8238621325726854e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2027_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.531413039076142,
            "second_run_seconds": 1.3311792620224878,
            "memory": {
              "sample_count": 24,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.2983872700715438,
            "second_run_seconds": 0.4539832330774516,
            "memory": {
              "sample_count": 15,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 7.808208465576172e-06,
          "mae": 8.163724487530999e-07,
          "rmse": 1.2348688187557855e-06,
          "relative_l2": 3.033577513633645e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2028_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.550065167946741,
            "second_run_seconds": 1.327464036992751,
            "memory": {
              "sample_count": 24,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.2502429169835523,
            "second_run_seconds": 0.45344983297400177,
            "memory": {
              "sample_count": 13,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 7.927417755126953e-06,
          "mae": 6.075666192373319e-07,
          "rmse": 9.668328857515007e-07,
          "relative_l2": 2.831891833920963e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2029_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.5230341999558732,
            "second_run_seconds": 1.4051283469889313,
            "memory": {
              "sample_count": 25,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.1580660950858146,
            "second_run_seconds": 0.4540409620385617,
            "memory": {
              "sample_count": 14,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 6.67572021484375e-06,
          "mae": 5.82778227453673e-07,
          "rmse": 8.858748969942098e-07,
          "relative_l2": 2.6855989290197613e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2030_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.4580265569966286,
            "second_run_seconds": 1.2604921449674293,
            "memory": {
              "sample_count": 26,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.175063184928149,
            "second_run_seconds": 0.45303693402092904,
            "memory": {
              "sample_count": 14,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 5.4836273193359375e-06,
          "mae": 5.130684144205588e-07,
          "rmse": 8.072146329141106e-07,
          "relative_l2": 2.657008280948503e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2031_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.4839264870388433,
            "second_run_seconds": 1.3027344680158421,
            "memory": {
              "sample_count": 23,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.1654355979990214,
            "second_run_seconds": 0.4523768249200657,
            "memory": {
              "sample_count": 14,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 7.152557373046875e-06,
          "mae": 6.775944711989723e-07,
          "rmse": 1.0585708878352307e-06,
          "relative_l2": 2.954650881292764e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2032_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.532301482046023,
            "second_run_seconds": 1.3457470380235463,
            "memory": {
              "sample_count": 24,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.202670434024185,
            "second_run_seconds": 0.45261138502974063,
            "memory": {
              "sample_count": 14,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 6.973743438720703e-06,
          "mae": 5.736724801863602e-07,
          "rmse": 9.060079833034251e-07,
          "relative_l2": 2.8207853119965876e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2033_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.4806106019532308,
            "second_run_seconds": 1.287079417030327,
            "memory": {
              "sample_count": 24,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.188530631014146,
            "second_run_seconds": 0.45207411504816264,
            "memory": {
              "sample_count": 13,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 473.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 14.0
            }
          }
        },
        "comparison": {
          "max_abs": 6.020069122314453e-06,
          "mae": 4.895751430922246e-07,
          "rmse": 7.820235623512417e-07,
          "relative_l2": 2.65942799160257e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2034_sample0.png"
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
              6
            ],
            "first_run_seconds": 1.4913692600093782,
            "second_run_seconds": 1.3060977660352364,
            "memory": {
              "sample_count": 23,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 499.0,
              "sampled_global_gpu_peak_mib": 559.0,
              "sampled_global_gpu_peak_delta_mib": 60.0,
              "torch_peak_allocated_mib": 30.9482421875,
              "torch_peak_reserved_mib": 54.0
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
              6
            ],
            "first_run_seconds": 1.1868669940158725,
            "second_run_seconds": 0.4531008650083095,
            "memory": {
              "sample_count": 14,
              "sampled_process_gpu_peak_mib": 0.0,
              "sampled_global_gpu_start_mib": 477.0,
              "sampled_global_gpu_peak_mib": 487.0,
              "sampled_global_gpu_peak_delta_mib": 10.0
            }
          }
        },
        "comparison": {
          "max_abs": 8.821487426757812e-06,
          "mae": 6.607633054045436e-07,
          "rmse": 1.0307319371349877e-06,
          "relative_l2": 2.934698613898945e-06,
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
            }
          ]
        },
        "visualizations": [
          "benchmark_results/torch_vs_exponax_solver_visualizations_ns_t5_multi10/ns_2d/ns_seed2035_sample0.png"
        ]
      }
    ],
    "statistics": {
      "comparison": {
        "max_abs": {
          "count": 10,
          "mean": 7.200241088867187e-06,
          "std": 1.057616265226723e-06,
          "min": 5.4836273193359375e-06,
          "max": 8.821487426757812e-06
        },
        "mae": {
          "count": 10,
          "mean": 6.227144183412747e-07,
          "std": 9.659385452675649e-08,
          "min": 4.895751430922246e-07,
          "max": 8.163724487530999e-07
        },
        "rmse": {
          "count": 10,
          "mean": 9.694897471490549e-07,
          "std": 1.3671744886296924e-07,
          "min": 7.820235623512417e-07,
          "max": 1.2348688187557855e-06
        },
        "relative_l2": {
          "count": 10,
          "mean": 2.830192966030154e-06,
          "std": 1.3034227510808864e-07,
          "min": 2.657008280948503e-06,
          "max": 3.033577513633645e-06
        }
      },
      "runs": {
        "pytorch_solvers_py": {
          "first_run_seconds": {
            "count": 10,
            "mean": 1.5158821568009444,
            "std": 0.0366539128383324,
            "min": 1.4580265569966286,
            "max": 1.5788980820216238
          },
          "second_run_seconds": {
            "count": 10,
            "mean": 1.3280878531048075,
            "std": 0.04426197953064711,
            "min": 1.2604921449674293,
            "max": 1.4051283469889313
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
            "mean": 60.0,
            "std": 0.0,
            "min": 60.0,
            "max": 60.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 10,
            "mean": 559.0,
            "std": 0.0,
            "min": 559.0,
            "max": 559.0
          },
          "torch_peak_allocated_mib": {
            "count": 10,
            "mean": 30.9482421875,
            "std": 0.0,
            "min": 30.9482421875,
            "max": 30.9482421875
          },
          "torch_peak_reserved_mib": {
            "count": 10,
            "mean": 54.0,
            "std": 0.0,
            "min": 54.0,
            "max": 54.0
          }
        },
        "jax_exponax": {
          "first_run_seconds": {
            "count": 10,
            "mean": 1.2085185432108119,
            "std": 0.04534008541061222,
            "min": 1.1580660950858146,
            "max": 1.2983872700715438
          },
          "second_run_seconds": {
            "count": 10,
            "mean": 0.45506769539788366,
            "std": 0.006059307394103272,
            "min": 0.45207411504816264,
            "max": 0.4722066599642858
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
            "mean": 13.6,
            "std": 1.2649110640673518,
            "min": 10.0,
            "max": 14.0
          },
          "sampled_global_gpu_peak_mib": {
            "count": 10,
            "mean": 487.0,
            "std": 0.0,
            "min": 487.0,
            "max": 487.0
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
          "mean_left_minus_right": 0.3073636135901324,
          "t_statistic": 27.905222982366496,
          "p_value": 4.734600847115474e-10
        },
        "second_run_seconds": {
          "count": 10,
          "mean_left_minus_right": 0.8730201577069238,
          "t_statistic": 62.11775145701685,
          "p_value": 3.662605068675334e-13
        },
        "sampled_process_gpu_peak_mib": {
          "count": 10,
          "mean_left_minus_right": 0.0,
          "t_statistic": 0.0,
          "p_value": 1.0
        },
        "sampled_global_gpu_peak_delta_mib": {
          "count": 10,
          "mean_left_minus_right": 46.4,
          "t_statistic": 115.99999999999999,
          "p_value": 1.3352451446376922e-15
        },
        "sampled_global_gpu_peak_mib": {
          "count": 10,
          "mean_left_minus_right": 72.0,
          "t_statistic": null,
          "p_value": 0.0
        }
      }
    }
  }
}
```
