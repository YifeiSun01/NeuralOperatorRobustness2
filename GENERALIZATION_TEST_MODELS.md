# Generalization Test Models

This file records the canonical trained FNO checkpoints to use for future
generalization tests and training-distribution-related evaluations.

Only these model artifacts should be used as the default pretrained models for
the 1D Burgers, 2D Darcy Flow, and 2D Navier-Stokes generalization experiments.

## 1D Burgers FNO

- Model: PyTorch FNO1d, 500 epochs
- Checkpoint:
  `1D_Burgers/trained_models/attack_ready/burgers_nu0.001_fno1d_500/checkpoints/pytorch_fno1d_500.pt`

## 2D Darcy Flow FNO

- Model: Darcy N1500, nx85, modes64, width60, 500 epochs, 20260528
- Checkpoint:
  `2D_Darcy_FNO2d/saved_models/2D/darcy_N1500_nx85_m64_w60_e500_20260528/best.pt`

## 2D Navier-Stokes FNO

- Model: modes64, width60, epochs500, Tin10, T10, recurrent PyTorch,
  20260521_090136_UTC
- Checkpoint:
  `2D_NS_FNO2d_recurrent/saved_models/2D/modes64_modes64_width60_epochs500_Tin10_T10_recurrent_pytorch_20260521_090136_UTC/checkpoints/best.pt`

## Usage Note

These checkpoints are the selected baseline models for evaluating
out-of-distribution and generalization datasets. Future inference, robustness,
or generalization-test scripts should point to these paths unless an experiment
explicitly states that it is comparing against another training run.
