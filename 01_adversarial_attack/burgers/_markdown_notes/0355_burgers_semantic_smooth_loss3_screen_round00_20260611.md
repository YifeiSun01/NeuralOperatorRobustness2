# Burgers Smooth Semantic Loss3 Screen Round 00 - 2026-06-11

This is the corrected smooth-dominant Burgers semantic generalization dataset after rejecting the sawtooth-heavy round. The selected root is:

`generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611/round_00`

## Policy

- Semantic OOD only: no attack-generated samples are used as generalization data.
- Smooth families dominate: Gaussian, Matern, power-law Fourier, and sine mixture.
- Sawtooth and square wave are capped to tiny diagnostic counts only.
- Spike train is removed.
- Values are bounded by the hard range [-0.7, 1.7]; the selected set actually stays inside [-0.600, 1.500].
- Every selected dataset has a unique parameter signature.

## Selected Set

- Selected datasets: 50
- Unique parameter signatures: 50
- Duplicate parameter signatures: 0
- Loss3 strict best count: 50 / 50

Family counts:

```json
{
  "gaussian": 15,
  "matern": 14,
  "powerlaw_fourier": 10,
  "sine_mixture": 8,
  "sawtooth": 2,
  "square_wave": 1
}
```

Target range counts:

```json
{
  "(-0.6, 1.4)": 3,
  "(-0.5, 1.5)": 5,
  "(-0.3, 1.3)": 10,
  "(-0.25, 1.25)": 6,
  "(-0.2, 1.2)": 5,
  "(-0.1, 1.1)": 3,
  "(-0.05, 0.95)": 2,
  "(0.0, 1.0)": 1,
  "(0.0, 1.2)": 5,
  "(0.0, 1.5)": 2,
  "(0.05, 1.05)": 2,
  "(0.1, 1.1)": 1,
  "(0.15, 1.25)": 5
}
```

Actual x stats:

- global min: -0.600000
- global max: 1.500000
- mean of dataset minima: -0.185000
- mean of dataset maxima: 1.257000
- average x mean: 0.532388
- average x std: 0.313337

## Loss3 Advantage

Loss3 RMSE divided by the better of loss1/loss2 on each dataset:

```json
{
  "min": 0.1191543189525211,
  "q25": 0.5769883775407153,
  "median": 0.5964455232847341,
  "mean": 0.5796924400515849,
  "q75": 0.6249347832131085,
  "max": 0.6573507980238451
}
```

Counts by threshold:

```json
{
  "lt_0p9": 50,
  "lt_0p8": 50,
  "lt_0p7": 50,
  "lt_0p6": 29
}
```

By family:

```json
{
  "gaussian": {
    "n": 15,
    "strict": 15,
    "rmse_ratio_min": 0.5625822874445868,
    "rmse_ratio_mean": 0.5987491474886338,
    "rmse_ratio_max": 0.6322448294844162
  },
  "matern": {
    "n": 14,
    "strict": 14,
    "rmse_ratio_min": 0.5692131571991893,
    "rmse_ratio_mean": 0.6132694556564589,
    "rmse_ratio_max": 0.6573507980238451
  },
  "powerlaw_fourier": {
    "n": 10,
    "strict": 10,
    "rmse_ratio_min": 0.5699195985318822,
    "rmse_ratio_mean": 0.5882411862409367,
    "rmse_ratio_max": 0.598577794673881
  },
  "sawtooth": {
    "n": 2,
    "strict": 2,
    "rmse_ratio_min": 0.35568357895181313,
    "rmse_ratio_mean": 0.3825936757787708,
    "rmse_ratio_max": 0.40950377260572846
  },
  "sine_mixture": {
    "n": 8,
    "strict": 8,
    "rmse_ratio_min": 0.4635923804493025,
    "rmse_ratio_mean": 0.5813573597674861,
    "rmse_ratio_max": 0.6447528532845425
  },
  "square_wave": {
    "n": 1,
    "strict": 1,
    "rmse_ratio_min": 0.1191543189525211,
    "rmse_ratio_mean": 0.1191543189525211,
    "rmse_ratio_max": 0.1191543189525211
  }
}
```

Candidate pool strict-loss3 counts before selection:

```json
{
  "square_wave": {
    "total": 33,
    "within_bounds": 33,
    "strict_loss3": 23,
    "within_bounds_strict_loss3": 23
  },
  "sawtooth": {
    "total": 42,
    "within_bounds": 42,
    "strict_loss3": 25,
    "within_bounds_strict_loss3": 25
  },
  "sine_mixture": {
    "total": 120,
    "within_bounds": 120,
    "strict_loss3": 82,
    "within_bounds_strict_loss3": 82
  },
  "gaussian": {
    "total": 195,
    "within_bounds": 195,
    "strict_loss3": 125,
    "within_bounds_strict_loss3": 125
  },
  "matern": {
    "total": 180,
    "within_bounds": 180,
    "strict_loss3": 115,
    "within_bounds_strict_loss3": 115
  },
  "powerlaw_fourier": {
    "total": 150,
    "within_bounds": 150,
    "strict_loss3": 150,
    "within_bounds_strict_loss3": 150
  }
}
```

## Diagnostic Plot

Signal and Fourier-spectrum examples are saved at:

`forensics/burgers_semantic_smooth_loss3_screen_round00_20260611/selected_family_signal_and_spectrum_examples.png`

Run status:

```json
{
  "status": "complete",
  "selected_count": 50,
  "select_count": 50,
  "candidate_count": 720,
  "output_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611/round_00",
  "candidate_root": "/workspace/NeuralOperatorRobustness2/generalization_datasets_burgers_semantic_smooth_loss3_screen_20260611/round_00_candidate_pool",
  "loss3_strict_best_count": 50,
  "require_strict": false
}
```
