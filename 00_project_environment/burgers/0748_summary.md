# Burgers First/Master SVD R2 Live Lookup 20260609

UTC time: `2026-06-09T00:54:31.046121+00:00`

## Scope

Live targeted R2 lookup under the selected prefix for a completed first/master semantic Burgers Jacobian/SVD artifact.

No credentials are recorded in this artifact.

## Observed Evidence

- R2 `forensics/` top-level directory count: `127`.
- SVD/Jacobian-related top-level directories found: `27`.
- First/master-related top-level directories found: `0`.
- Direct lookup path: `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608/`.
- Direct lookup stdout empty: `True`.
- Direct lookup stderr empty: `True`.

## SVD/Jacobian Top-Level Directories Found

- `burgers_adv_training_jacobian_svd_20260531_representative20_same_points/`
- `burgers_jacobian_downsample_svd_probe_20260607/`
- `burgers_loss12_vs_loss3_jacobian_svd_comparison_20260603/`
- `burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604/`
- `burgers_loss3_aligned_round01_final_jacobian_svd_smoke_20260604/`
- `burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605/`
- `burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605/`
- `burgers_loss3_selective_round03_loss123_continuation_final_jacobian_svd_rep20_top100_20260606/`
- `burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606/`
- `burgers_p2q2_checkpoint_series_jacobian_svd_20260601/`
- `burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603/`
- `burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top20_20260603/`
- `burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603/`
- `burgers_round03_loss123_downsample_svd_probe_20260607/`
- `burgers_round03_loss123_svd_vector_frequency_probe_20260608/`
- `burgers_zero_adv_training_jacobian_svd_20260601_rep20subset10_top100/`
- `comprehensive_svd_diagnostics_20260515/`
- `comprehensive_svd_diagnostics_20260515_no_std/`
- `deeponet_solver_jacobian_similarity_20260515/`
- `fno_deeponet_nu0p01_comprehensive_svd_diagnostics_20260515_no_std/`
- `fno_nu0p001_nu0p01_deeponet_nu0p01_ninerow_svd_diagnostics_20260515_no_std/`
- `fno_nu0p01_solver_jacobian_similarity_20260515/`
- `fno_solver_jacobian_similarity_20260514/`
- `fno_solver_jacobian_similarity_20260514_raw_recomputed/`
- `local_jacobian_frequency_20260514/`
- `loss3_current_core4_jacobian_svd_probe_20260521/`
- `loss3_jacobian_subspace_rotation_path_20260516/`

## Conclusion

The selected R2 prefix contains completed SVD/Jacobian evidence for second/ns50, round01/aligned, and round03/selective families, but the live targeted lookup did not find a completed first/master semantic SVD artifact.

Therefore the first/master full attack result cannot currently be joined against first/master error-Jacobian spectral norms. The available round01/aligned SVD correlation result must not be substituted for the true first/master root.

## Required Next Step

To compute the requested first/master `||J_model-J_solver||_2` vs attack damage correlation, we need either to locate a first/master SVD artifact outside this selected R2 prefix or run the first/master SVD audit using the existing first/master sample manifest.
