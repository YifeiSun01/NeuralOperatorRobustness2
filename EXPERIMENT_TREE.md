# Experiment Tree

- `00_project_environment`: environment, setup, dependency, Git/GitHub, R2/runbook, and infrastructure notes.
- `00_literature_and_theory`: literature review, theory notes, framework notes, and solver/operator-learning background.
- `00_shared_data_and_models`: dataset generation, model-training, and shared model/solver notes.
- `01_adversarial_attack`: attack experiments, PGD/Adam-PGD, loss-objective variants, optimizer ablations, epsilon sweeps.
  - `burgers`: Burgers FNO and DeepONet attack notes, including nu-specific runs.
  - `darcy_flow`: Darcy/C-flow binary coefficient attack and epsilon sweep notes.
  - `navier_stokes`: recurrent NS attack, periodic/warp evidence, external forcing, and optimizer/epsilon variants.
  - `cross_benchmark`: three-system summaries and attack-objective comparisons.
- `02_adversarial_training`: solver-integrated adversarial training, random/clean baselines, 52-dataset generalization, relative-L2 summaries.
- `03_robustness_metrics`: Jacobian, SVD, subspace, spectrum, angle, and correlation diagnostics.
- `04_timing_memory`: runtime, memory, batch-size, rematerialization, and timing tables.
- `05_mechanism_and_ablations`: Loss 3 geometry, loss landscape, JVP/VJP, gradient-path, and ablation studies.
- `09_uncategorized`: markdown notes that do not match a stronger experiment category.
