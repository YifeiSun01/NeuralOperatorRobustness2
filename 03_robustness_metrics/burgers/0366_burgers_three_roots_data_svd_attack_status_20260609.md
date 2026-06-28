# Burgers Three Generalization Roots: Data, SVD, And Attack/Tag Status (2026-06-09)

## Scope

This note records what is evidenced, what is not evidenced, and what is currently running for the three Burgers generalization roots discussed in the recent audit.

Roots:

1. First/master semantic root: `generalization_datasets/burgers`
2. Second/ns50 target-band root: `generalization_datasets_rmse_1p5_3x_all_ns50/burgers`
3. Third/round03 selective stress root: `generalization_datasets_burgers_loss3_selective_search/round_03/burgers`

No credentials or secrets are recorded here.

## Short Answer

| item | first/master semantic | second/ns50 target-band | third/round03 selective |
|---|---|---|---|
| 50 Burgers `.pt` datasets on R2 | yes | yes | yes |
| clean/generalization evaluation | yes, local record exists | yes, local record exists | yes, stress-set clean/generated records exist |
| representative Jacobian/SVD evidence | not evidenced in selected R2 metadata | yes | yes |
| full 52-dataset / 10200-sample P2Q2 attack/tag | not found on R2; local run currently generating | yes, complete | not found on R2; queued in local run |
| caveat | SVD still unclear/missing for this exact root | strongest clean-vs-robustness mismatch evidence | selected/attack-generated stress set, not neutral generalization |

## Data Roots On R2

Observed from completed R2 listings under `neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected`:

| root | Burgers `.pt` count | evidence file |
|---|---:|---|
| `generalization_datasets/burgers` | 50 | `forensics/r2_burgers_missing_roots_inventory_20260608/r2_generalization_datasets_lsf_pst.tsv` |
| `generalization_datasets_rmse_1p5_3x_all_ns50/burgers` | 50 | `forensics/r2_burgers_missing_roots_inventory_20260608/r2_generalization_datasets_rmse_1p5_3x_all_ns50_lsf_pst.tsv` |
| `generalization_datasets_burgers_loss3_selective_search/round_03/burgers` | 50 | `forensics/r2_burgers_missing_roots_inventory_20260608/r2_generalization_datasets_burgers_loss3_selective_search_lsf_pst.tsv` |

The third root also has a `round_03_candidate_pool/burgers` pool with 60 `.pt` candidate datasets.

## SVD Status

Observed from R2 SVD metadata audit:

- R2 selected prefix contains 27 `svd`/`jacobian` top-level artifact directories under `forensics/`.
- Only metadata files were copied locally for provenance; large `.npz` SVD matrices were not downloaded.
- Local metadata mirror: `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_top_metadata/`.
- Derived provenance summaries:
  - `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_manifest_provenance_summary.csv`
  - `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_manifest_provenance_summary.md`

Evidenced SVD roots:

- Second/ns50 has SVD evidence. Example artifacts:
  - `burgers_adv_training_jacobian_svd_20260531_representative20_same_points`
  - `burgers_p2q2_checkpoint_series_jacobian_svd_20260601`
  - `burgers_p2q2_loss1_epoch2000_jacobian_svd_rep20_top100_20260603`
  - `burgers_p2q2_loss2_epoch900_jacobian_svd_rep20_top100_20260603`
- Round01/aligned has SVD evidence:
  - `burgers_loss3_aligned_round01_final_jacobian_svd_rep20_top100_20260604`
- Third/round03 selective has SVD evidence. Example artifacts:
  - `burgers_loss3_selective_round03_final_jacobian_svd_gen5_top20_20260605`
  - `burgers_loss3_selective_round03_long_final_jacobian_svd_rep20_top100_20260605`
  - `burgers_loss3_selective_round03_loss123_final_extension_jacobian_svd_rep20_top100_20260606`

Not evidenced:

- Exact first/master semantic root string `generalization_datasets/burgers` was not found in the downloaded R2 SVD metadata.
- Local `forensics/burgers_first_master_finalmodels_jacobian_svd_rep20_top100_20260608` appears to contain preflight/config/manifest/sample-start files, but no completed summary was evidenced in the current audit.

Corrected interpretation:

- It is incorrect to say that SVD was not done broadly. SVD was done for several Burgers generalization families, including second/ns50 and third/round03.
- The unclear/missing SVD evidence is specifically for the first/master semantic root.

## Full Attack/Tag Status

The target full attack/tag protocol is:

- train first 50 + full test + 50 generalization datasets
- 52 datasets total
- 10200 samples total
- 4 models: baseline, `loss1_epoch8000`, `loss2_epoch2000`, `loss3_epoch1500`
- P2Q2 final-only attack/tag
- `steps=20`, `epsilon_rms=0.12`, `alpha_rms=0.012`, `batch_size=500`

Observed complete artifact:

- Second/ns50 full attack/tag is complete:
  - `forensics/burgers_round03_full_p2q2_52datasets_4models_finalonly_20step_20260607`

Not found on R2:

- No first/master-specific full P2Q2 final-only artifact was found in R2 top-level artifact listings.
- No third/round03-specific full P2Q2 final-only artifact was found in R2 top-level artifact listings.

Currently running locally:

- tmux session: `burgers_missing_roots_fulltag_20260608`
- launcher: `tools/run_burgers_missing_roots_full_p2q2_attack_20260608.sh`
- first-root output target:
  - `forensics/burgers_first_master_full_p2q2_52datasets_4models_finalonly_20step_20260608`
- third-root output target after first finishes:
  - `forensics/burgers_round03_selective_full_p2q2_52datasets_4models_finalonly_20step_20260608`

Important clarification:

- The running background job is full P2Q2 attack/tag, not Jacobian/SVD.
- It should have runtime closer to the previous full-tag run, not dense SVD runtime. It still takes time because it attacks 10200 samples for four models per root.

## Clean/Generalization And Robustness Interpretation So Far

Observed from previous records:

- Second/ns50 is the strongest complete clean-vs-robustness mismatch evidence:
  - clean/generalization favors loss1/loss2, especially loss1
  - full attack/tag robustness strongly favors loss3
- Third/round03 supports loss3 on the selected stress set and has SVD/representative robustness evidence, but it is selection-biased and should not be used as neutral generalization proof.
- First/master semantic clean/generalization has local records, but full robustness/tag and exact-root completed SVD evidence are not currently evidenced.

## Remaining Work

1. Let the current first-root full attack/tag run finish.
2. Let the queued third-root full attack/tag run finish.
3. Summarize first/third full attack/tag results into per-dataset and per-sample winner/count tables.
4. Revisit first/master SVD only if a separate exact-root SVD result can be found in another R2 prefix/old machine, or if the user explicitly decides to pay the long SVD runtime again.
5. Upload new completed full attack/tag artifacts to R2 if needed.

## Source Records

Detailed supporting records:

- `docs/r2_burgers_missing_roots_inventory_20260608.md`
- `docs/burgers_missing_roots_full_robustness_run_20260608.md`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_manifest_provenance_summary.md`
- `forensics/r2_burgers_missing_roots_inventory_20260608/r2_svd_manifest_provenance_summary.csv`
