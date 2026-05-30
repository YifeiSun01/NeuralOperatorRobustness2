# NS2D Generalization Ground Truth Repair and Re-evaluation

## What was wrong

The original generated NS2D generalization set contained solver explosions. With the `abs(y) <= 5` stability rule on solver-generated frames (`t > 0`), 254 samples were unstable before repair. Including the initial frame gave 293 unstable samples, because one generated distribution had initial conditions outside the allowed amplitude range.

## Repair action

- Repaired 243 unstable samples in place across 46 normal NS2D dataset files.
- Every repaired normal sample was accepted at `fixed_step = 0.005`, i.e. one halving from the original `0.01`.
- Quarantined `ns_mid_spectrum_alpha0p8_tau5.pt` because all 50 initial conditions exceeded `abs(x)>5`, 11 solver rollouts were unstable, and 5 samples remained unstable even after halving down to `dt=0.00015625`.
- Removed `(alpha=0.8, tau=5.0)` from future NS generated specs.

## Verification

Normal NS2D set after quarantine: 49 files.

- Solver-frame scan (`t>0`): 0 unstable samples.
- All-frame scan including initial frame: 0 unstable samples.

The quarantined file is in `generalization_datasets/ns2d_quarantine/ns_mid_spectrum_alpha0p8_tau5.pt`; it is no longer picked up by the normal evaluator glob.

## Re-evaluated NS2D losses

Using `generalization_eval_repaired_ns/metrics.csv`:

- train RMSE: 0.0713213, relative L2: 0.0512957
- test RMSE: 0.133571, relative L2: 0.0957001
- generated rows: 49
- soft-focus rows excluding sign cases: 47
- soft-focus generated below test RMSE: 31/47
- soft-focus generated below test relative L2: 31/47

The low-loss effect therefore remains after fixing exploded targets. It is not only a raw-RMSE scale artifact: the same 31/47 soft-focus cases are below test by relative L2 too.

## Pattern finding after repair

The strongest soft-focus correlations with RMSE were target final range, target final TV, target final high-frequency fraction, and target final spectral centroid. Low-loss NS cases are visually/spectrally smoother, with lower target high-frequency energy and smaller target spatial variation. High-loss cases have larger final ranges, higher TV, and more high-frequency content.

Generated visual panels and tables are in `analysis_outputs/generalization_loss_pattern_diagnostics_repaired_ns/`.
