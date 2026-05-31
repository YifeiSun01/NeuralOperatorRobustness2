# Darcy batch sweep 2026-05-31

Purpose: find a Darcy full10 retry configuration that avoids the CUDA OOM seen at batch 448 / optimizer batch 32.

Sweep conditions matched the formal run pre-eval path: `--darcy-attack-steps 10`, `--eval-max-samples 50`, `--max-generalization-eval 50`, one train step after baseline eval.

Result: batch 512 OOMed. Batch 448 and 480 can pass one-step probes but run with very small memory margin. Recommended formal retry is `--darcy-batch-size 416 --darcy-optimizer-batch-size 128`; conservative fallback is batch 384 / optimizer batch 128.

See `summary_metrics.csv` for raw timing and memory values.
