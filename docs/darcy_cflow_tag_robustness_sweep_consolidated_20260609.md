# Darcy/C-flow Tag Robustness Sweep - Consolidated Results - 2026-06-09

## Scope

This record consolidates the Darcy/C-flow tag experiments run on 2026-06-09 for five formal models:

- `baseline`
- `loss1`
- `loss2`
- `loss3`
- `physical_source`

The baseline model is not loss-trained. All attack/tag comparisons use a shared solver-consistent `loss3` objective so that post-tag loss growth is directly comparable across models.

No Cloudflare or GitHub credentials are recorded in this document.

## Artifact Roots

Local artifacts:

- One-batch 1-step run: `forensics/darcy_five_model_one_batch_tag_20260609/`
- One-dataset 20-step timing run: `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/`
- Full-50 20-step run: `forensics/darcy_five_model_full50_tag20_20260609/`
- Protocol sweep: `forensics/darcy_five_model_tag_protocol_sweep_20260609/`
- Darcy image bundle: `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/`
- Protocol sweep summary figures: `visualizations/darcy_five_model_tag_protocol_sweep_20260609/`

R2 upload target prefix:

- `r2://neural-operator-robustness/machine-sync/NeuralOperatorRobustness2-selected/`

## GPU Evidence

Observed from the generated preflight JSON files:

- GPU: Tesla V100-SXM2-32GB
- PyTorch: `2.8.0+cu126`
- PyTorch CUDA: `12.6`
- CUDA architecture list includes `sm_70`
- JAX: `0.10.0`
- JAX backend: `gpu`
- JAX device: `cuda:0`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` for the sweep runs that used JAX solver paths

## One-Batch 1-Step Result

Source: `forensics/darcy_five_model_one_batch_tag_20260609/one_batch_summary_by_model.csv`.

Protocol:

- Dataset: `darcy_lossdrop_pool_soft_l4_h10_b10_02.pt`
- Batch size: `48`
- Steps: `1`
- Epsilon fraction: `0.025`

| rank | model | clean loss mean | final tag loss mean | loss gain mean | relative gain mean |
|---:|---|---:|---:|---:|---:|
| 1 | loss3 | 9.599584697520906e-08 | 1.1024506788951764e-07 | 1.424922091430858e-08 | 0.17596435946567604 |
| 2 | physical_source | 1.5705951295075238e-07 | 1.835484744934964e-07 | 2.648896154274401e-08 | 0.18621729942969978 |
| 3 | loss1 | 2.9563728087822483e-07 | 3.371411810467369e-07 | 4.1503900168512096e-08 | 0.1633932989401122 |
| 4 | loss2 | 2.424386837951431e-07 | 2.854224743960761e-07 | 4.298379060093301e-08 | 0.2081057421552638 |
| 5 | baseline | 3.277261191314551e-07 | 4.460621176131478e-07 | 1.1833599848169267e-07 | 0.41604626675446826 |

Observed result: `loss3` is most robust for this one weak one-step batch.

## One-Dataset 20-Step Timing Result

Source: `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/timing_by_model.csv`.

Protocol:

- Dataset: `darcy_lossdrop_pool_soft_l4_h10_b10_02.pt`
- Batch size: `48`
- Steps: `20`
- Epsilon fraction: `0.025`
- Total wall time across all five models: `29.999275830574334` seconds
- Direct 50-dataset estimate: `1499.9637915287167` seconds, or `0.41665660875797683` hours

| rank | model | attack wall sec | clean loss mean | final tag loss mean | loss gain mean |
|---:|---|---:|---:|---:|---:|
| 1 | physical_source | 2.775472982786596 | 1.5705951295075238e-07 | 2.386436874779463e-07 | 8.158417452719391e-08 |
| 2 | loss2 | 2.7978224605321884 | 2.424386837951431e-07 | 3.602693124567698e-07 | 1.178306286616267e-07 |
| 3 | loss1 | 2.7644626293331385 | 2.9563728087822483e-07 | 4.2230398289433424e-07 | 1.266667020161094e-07 |
| 4 | loss3 | 2.8103448692709208 | 9.599584697520906e-08 | 3.8296407585865683e-07 | 2.869682288834478e-07 |
| 5 | baseline | 4.633883801288903 | 3.277261191314551e-07 | 8.32303507299533e-07 | 5.045773837271857e-07 |

Observed result: at 20 steps with base budget on this one dataset, `physical_source` is most robust and `loss3` is no longer robust.

## Full-50 20-Step Base-Budget Result

Source: `forensics/darcy_five_model_full50_tag20_20260609/summary_by_model.csv`.

Protocol:

- Datasets: `50`
- Samples per model: `2400`
- Batch size: `48`
- Steps: `20`
- Epsilon fraction: `0.025`

| rank | model | clean loss mean | final tag loss mean | mean loss gain | relative gain mean |
|---:|---|---:|---:|---:|---:|
| 1 | physical_source | 1.5452080666037796e-07 | 2.2655414384065153e-07 | 7.203333720543863e-08 | 0.48388714697166807 |
| 2 | loss2 | 2.388385874141363e-07 | 3.4006990865679637e-07 | 1.012313212545024e-07 | 0.4850481873371427 |
| 3 | loss1 | 2.9298726013167926e-07 | 3.9749649036302517e-07 | 1.0450923022542469e-07 | 0.3765278256293641 |
| 4 | loss3 | 9.476862238363044e-08 | 4.0037479414500866e-07 | 3.0560617149788525e-07 | 4.059539115148639 |
| 5 | baseline | 3.2841694900328143e-07 | 7.814819267082385e-07 | 4.530649779432849e-07 | 1.733155802086306 |

Observed per-dataset winners:

- `physical_source`: `50 / 50`
- `loss3`: `0 / 50`

Observed result: for the full 50-dataset, 20-step, base-budget protocol, `physical_source` is the most robust model. `loss3` has the lowest clean loss but a much larger post-tag loss gain.

## Protocol Sweep Summary

Source: `forensics/darcy_five_model_tag_protocol_sweep_20260609/combined_summary_by_model.csv` and per-protocol `summary_by_dataset.csv` files.

| protocol | epsilon | steps | winner by mean gain | winner datasets | loss3 rank / notes |
|---|---:|---:|---|---|---|
| `eps0p025_steps001` | 0.025 | 1 | loss3 | loss3 `50 / 50` | loss3 rank 1 on `50 / 50` |
| `eps0p025_steps005` | 0.025 | 5 | physical_source | physical_source `48 / 50`, loss3 `2 / 50` | loss3 rank 1 on `2`, rank 2 on `30`, rank 3 on `5`, rank 4 on `13` |
| `eps0p025_steps010` | 0.025 | 10 | physical_source | physical_source `50 / 50` | loss3 rank 4 on `50 / 50` |
| `eps0p00625_steps001` | 0.00625 | 1 | loss3 | loss3 `50 / 50` | loss3 rank 1 on `50 / 50` |
| `eps0p00625_steps005` | 0.00625 | 5 | loss3 | loss3 `50 / 50` | loss3 rank 1 on `50 / 50` |
| `eps0p00625_steps010` | 0.00625 | 10 | loss3 | loss3 `46 / 50`, physical_source `4 / 50` | loss3 rank 1 on `46`, rank 2 on `4` |
| `eps0p00625_steps020` | 0.00625 | 20 | physical_source | physical_source `47 / 50`, loss3 `3 / 50` | loss3 rank 1 on `3`, rank 2 on `33`, rank 3 on `8`, rank 4 on `6` |

## Protocol Sweep Full Model Table

| protocol | model | epsilon | steps | mean adv loss | mean gain | relative gain | final flip fraction | boundary ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eps0p00625_steps001 | loss3 | 0.00625 | 1 | 9.877191460431571e-08 | 4.003292220685258e-09 | 0.04797323691588948 | 0.0062283738516271114 | 0.9965398162603378 |
| eps0p00625_steps001 | physical_source | 0.00625 | 1 | 1.6926859771023337e-07 | 1.4747791049855388e-08 | 0.1055029466120565 | 0.0062283738516271114 | 0.9965398162603378 |
| eps0p00625_steps001 | loss1 | 0.00625 | 1 | 3.1636686430580597e-07 | 2.337960417412669e-08 | 0.09164197524965857 | 0.0062283738516271114 | 0.9965398162603378 |
| eps0p00625_steps001 | loss2 | 0.00625 | 1 | 2.625712583093076e-07 | 2.373267089517128e-08 | 0.11329223169712349 | 0.0062283738516271114 | 0.9965398162603378 |
| eps0p00625_steps001 | baseline | 0.00625 | 1 | 3.7460086044364497e-07 | 4.618391144036356e-08 | 0.16087381922329466 | 0.0062283738516271114 | 0.9965398162603378 |
| eps0p00625_steps005 | loss3 | 0.00625 | 5 | 1.1440517353615103e-07 | 1.9636551152520572e-08 | 0.24987284244696942 | 0.024361533969640733 | 3.897845435142517 |
| eps0p00625_steps005 | physical_source | 0.00625 | 5 | 1.9077309159361752e-07 | 3.6252284933239544e-08 | 0.25055614702667905 | 0.02685253754258156 | 4.296406006813049 |
| eps0p00625_steps005 | loss1 | 0.00625 | 5 | 3.5248601127690903e-07 | 5.9498751145229773e-08 | 0.22678496850501687 | 0.02784625131636858 | 4.455400210618973 |
| eps0p00625_steps005 | loss2 | 0.00625 | 5 | 3.014496265970242e-07 | 6.261103918288787e-08 | 0.30316215877537617 | 0.029575720839202405 | 4.732115334272384 |
| eps0p00625_steps005 | baseline | 0.00625 | 5 | 4.833271426522382e-07 | 1.5491019365043712e-07 | 0.5608634165534749 | 0.031087716184556483 | 4.974034589529038 |
| eps0p00625_steps010 | loss3 | 0.00625 | 10 | 1.3594514303309305e-07 | 4.117652063762023e-08 | 0.5352666823095554 | 0.040733679458498956 | 6.517388713359833 |
| eps0p00625_steps010 | physical_source | 0.00625 | 10 | 2.0761541746091203e-07 | 5.309461080201435e-08 | 0.36613631833007576 | 0.03938010357320309 | 6.3008165717124935 |
| eps0p00625_steps010 | loss1 | 0.00625 | 10 | 3.7980398479871745e-07 | 8.681672466999875e-08 | 0.3301820947355494 | 0.0424400806427002 | 6.790412902832031 |
| eps0p00625_steps010 | loss2 | 0.00625 | 10 | 3.314450045207214e-07 | 9.260641709030182e-08 | 0.4552258371084463 | 0.04881199508905411 | 7.809919214248657 |
| eps0p00625_steps010 | baseline | 0.00625 | 10 | 5.78211264929962e-07 | 2.497943158571066e-07 | 0.9068755687897404 | 0.06098667807877064 | 9.757868492603302 |
| eps0p00625_steps020 | physical_source | 0.00625 | 20 | 2.1556954710770052e-07 | 6.104874045620434e-08 | 0.41945920999897984 | 0.04497866213321686 | 7.196585941314697 |
| eps0p00625_steps020 | loss3 | 0.00625 | 20 | 1.8294976677897997e-07 | 8.818114431689376e-08 | 1.166315531759839 | 0.06938223756849765 | 11.101158010959626 |
| eps0p00625_steps020 | loss1 | 0.00625 | 20 | 3.892225402720347e-07 | 9.623528013443424e-08 | 0.36346263329707046 | 0.04776337951421738 | 7.64214072227478 |
| eps0p00625_steps020 | loss2 | 0.00625 | 20 | 3.429667603634575e-07 | 1.0412817296708473e-07 | 0.5122109646051346 | 0.0572243944555521 | 9.155903112888335 |
| eps0p00625_steps020 | baseline | 0.00625 | 20 | 7.13501291068989e-07 | 3.8508434207903027e-07 | 1.426191907163399 | 0.1058377155661583 | 16.934034490585326 |
| eps0p025_steps001 | loss3 | 0.025 | 1 | 1.090436134750424e-07 | 1.4274991091411949e-08 | 0.1810436872584281 | 0.02505190297961235 | 1.002076119184494 |
| eps0p025_steps001 | physical_source | 0.025 | 1 | 1.796702562846993e-07 | 2.5149449624321344e-08 | 0.1817267506739366 | 0.02505190297961235 | 1.002076119184494 |
| eps0p025_steps001 | loss1 | 0.025 | 1 | 3.3386720547016563e-07 | 4.087994533848634e-08 | 0.16231988534998285 | 0.02505190297961235 | 1.002076119184494 |
| eps0p025_steps001 | loss2 | 0.025 | 1 | 2.80856116348834e-07 | 4.201752893469764e-08 | 0.20883804410999193 | 0.02505190297961235 | 1.002076119184494 |
| eps0p025_steps001 | baseline | 0.025 | 1 | 4.438385077065732e-07 | 1.1542155870773267e-07 | 0.41612709817942234 | 0.02505190297961235 | 1.002076119184494 |
| eps0p025_steps005 | physical_source | 0.025 | 5 | 2.105818647650063e-07 | 5.606105812091163e-08 | 0.37273422070099815 | 0.07479792430996896 | 2.9919169723987578 |
| eps0p025_steps005 | loss3 | 0.025 | 5 | 1.749845939199209e-07 | 8.021597157921907e-08 | 1.0620691056107854 | 0.08642445176839829 | 3.456978070735931 |
| eps0p025_steps005 | loss2 | 0.025 | 5 | 3.215823535818174e-07 | 8.27437661602796e-08 | 0.39724206191987227 | 0.08346470579504967 | 3.3385882318019866 |
| eps0p025_steps005 | loss1 | 0.025 | 5 | 3.8185313542271617e-07 | 8.886587527327331e-08 | 0.3193357711926607 | 0.07563835054636002 | 3.0255340218544005 |
| eps0p025_steps005 | baseline | 0.025 | 5 | 6.238141669001607e-07 | 2.953972180878376e-07 | 1.0818868246488273 | 0.11187623977661133 | 4.475049591064453 |
| eps0p025_steps010 | physical_source | 0.025 | 10 | 2.222101413214735e-07 | 6.768933463445019e-08 | 0.4527978131692241 | 0.08615547820925712 | 3.446219128370285 |
| eps0p025_steps010 | loss2 | 0.025 | 10 | 3.372281656766063e-07 | 9.838957827135176e-08 | 0.47067887702647565 | 0.0969005186855793 | 3.876020747423172 |
| eps0p025_steps010 | loss1 | 0.025 | 10 | 3.942109712469962e-07 | 1.0122371113604108e-07 | 0.363223818110079 | 0.08605357527732849 | 3.4421430110931395 |
| eps0p025_steps010 | loss3 | 0.025 | 10 | 2.724338306829092e-07 | 1.7766520836293153e-07 | 2.360242200995972 | 0.14875201836228372 | 5.9500807344913484 |
| eps0p025_steps010 | baseline | 0.025 | 10 | 7.586826978685698e-07 | 4.3026574925164596e-07 | 1.6314568789800008 | 0.16402756661176682 | 6.561102664470672 |

## Boundary Interpretation

For a single step at epsilon `0.025`, final flip fraction is about `0.025`, matching the step budget. For multi-step attacks, final flip fraction is below the naive `steps * epsilon` sum because later steps can revisit or undo earlier flips.

Examples:

- `eps=0.025`, `10 steps`, `loss3` final flip fraction: `0.14875201836228372`
- `eps=0.025`, `10 steps`, `physical_source` final flip fraction: `0.08615547820925712`
- `eps=0.00625`, `20 steps`, `loss3` final flip fraction: `0.06938223756849765`
- `eps=0.00625`, `20 steps`, `physical_source` final flip fraction: `0.04497866213321686`

## Interpretation

Observed evidence supports a threshold behavior:

- `loss3` is the robust winner for weak attacks: base budget at 1 step, and quarter budget through 10 steps.
- `physical_source` becomes the robust winner once the attack accumulates enough strength: base budget by 5 steps, quarter budget by 20 steps.
- `baseline` is consistently the least robust model among the completed comparisons.
- `loss3` has the lowest clean loss in several comparisons, but low clean loss does not imply strong multi-step tag robustness.

The robust winner therefore depends on attack strength. For weak tag protocols use `loss3`; for stronger/more accumulated tag protocols, `physical_source` is the robust model.

## Files To Push To GitHub

Code and Markdown record files:

- `tools/run_darcy_five_model_generalization_tag_20260609.py`
- `tools/run_darcy_five_model_one_batch_tag_20260609.py`
- `tools/time_darcy_five_model_one_dataset_tag20_20260609.py`
- `tools/run_darcy_five_model_tag_protocol_sweep_20260609.py`
- `tools/plot_darcy_tag_protocol_sweep_20260609.py`
- `docs/darcy_five_model_one_batch_tag_20260609.md`
- `docs/darcy_five_model_one_dataset_tag20_timing_20260609.md`
- `docs/darcy_five_model_full50_tag20_20260609.md`
- `docs/darcy_cflow_tag_robustness_sweep_consolidated_20260609.md`
- `EXPERIMENT_LEDGER.md`

## Files To Upload To R2

Selected data and visualization artifacts:

- `forensics/darcy_five_model_one_batch_tag_20260609/`
- `forensics/darcy_five_model_one_dataset_tag20_timing_20260609/`
- `forensics/darcy_five_model_full50_tag20_20260609/`
- `forensics/darcy_five_model_tag_protocol_sweep_20260609/`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/`
- `visualizations/darcy_five_model_tag_protocol_sweep_20260609/`

## Loss3-Advantage Six-Sample Visualization

A dense six-sample visualization was generated after the protocol sweep to make the weak-attack `loss3` advantage visually inspectable.

Protocol used for sample selection and plotting:

- Epsilon fraction: `0.00625`
- Attack steps: `10`
- Shared objective: `loss3`
- Selection source: `forensics/darcy_five_model_tag_protocol_sweep_20260609/eps0p00625_steps010/summary_by_dataset.csv`
- Selection rule: choose datasets where `loss3` won, then choose the sample with the largest per-sample margin between `loss3` gain and the next-best non-loss3 gain.

Generated files:

- Figure: `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_six_sample_tag_plate.png`
- Selected sample table: `forensics/darcy_loss3_advantage_six_sample_plate_20260609/selected_samples.csv`
- Plot script: `tools/plot_darcy_loss3_advantage_six_sample_plate_20260609.py`

Observed selected samples and margins:

| rank | dataset | sample | loss3 gain | next best non-loss3 gain | margin |
|---:|---|---:|---:|---:|---:|
| 1 | `darcy_lossdrop_pool_soft_l4_h10_b12_10.pt` | 25 | 3.120010205748258e-08 | 2.1155113927306957e-07 | 1.8035103721558698e-07 |
| 2 | `darcy_lossdrop_pool_soft_l4_h10_b14_07.pt` | 30 | 1.1386561027393327e-08 | 1.6295378202357824e-07 | 1.5156722099618491e-07 |
| 3 | `darcy_lossdrop_pool_soft_l4_h10_b10_03.pt` | 25 | 8.325713451995398e-09 | 1.5869332514739654e-07 | 1.5036761169540114e-07 |
| 4 | `darcy_lossdrop_pool_soft_l4_h10_b8_14.pt` | 45 | 1.33276500946522e-09 | 1.4893954869421577e-07 | 1.4760678368475055e-07 |
| 5 | `darcy_lossdrop_pool_soft_l4_h10_b12_11.pt` | 20 | 1.182712594527402e-08 | 1.5562221733489423e-07 | 1.437950913896202e-07 |
| 6 | `darcy_lossdrop_pool_soft_l4_h10_b12_07.pt` | 40 | 1.1290808288322296e-09 | 1.4337588538637647e-07 | 1.4224680455754424e-07 |

Figure layout:

- Six rows correspond to six selected generalization samples.
- The left column shows the clean Darcy input field.
- The five model columns show `baseline`, `loss1`, `loss2`, `loss3`, and `physical_source`.
- Each model cell contains perturbation, model output, solver output, and absolute model-solver error.
- Green borders mark the `loss3` column.
- Per-cell annotations show final MSE, loss gain, and gain ratio relative to the best model in that row.

Inference from these selected examples: under the `eps=0.00625`, `10-step` weak-attack protocol, the chosen examples make the `loss3` advantage visually and numerically explicit. These are intentionally selected examples, not a replacement for the full-50 aggregate statistics above.

## Large Loss3-Advantage Visualization Variants

Additional large-format variants were generated to make the selected examples easier to inspect visually.

Observed from `tools/plot_darcy_loss3_advantage_large_variants_20260609.py` and `forensics/darcy_loss3_advantage_six_sample_plate_20260609/large_variant_metrics.csv`:

- The same six selected samples from `forensics/darcy_loss3_advantage_six_sample_plate_20260609/selected_samples.csv` were reused.
- The same weak-tag protocol was rerun: `epsilon_fraction=0.00625`, `steps=10`, shared `loss3` attack objective.
- GPU verification for this plotting rerun is stored at `forensics/darcy_loss3_advantage_six_sample_plate_20260609/large_variants_gpu_preflight.json`.
- The delta/error-only figures remove model and solver heatmaps so the perturbation and absolute model-solver error are larger.
- The four-panel figures use the actual row-shared model/solver output range. This corrects the earlier display issue where model and solver panels were visually flattened by an overly wide fixed color range.
- The four-panel heatmaps are forced to square aspect. No `delta`, `model`, `solver`, or `error` panel is stretched into a rectangle.
- Each row's absolute-error panels use one shared color range across all five models so the `loss3` error reduction is visually comparable within the row.
- `large_variant_metrics.csv` now records final changed-pixel counts and nonzero delta magnitudes for each displayed sample/model.
- The displayed `delta` is not a fixed-amplitude additive perturbation. In this binary-replace Darcy attack, selected pixels are replaced by the sample's low or high coefficient value, so `delta = replaced_value - original_value` can have many magnitudes even though each selected pixel is discretely replaced.
- For these `85 x 85` samples, `epsilon_fraction=0.00625` gives about `45` pixel replacements per step. Over `10` steps the maximum no-repeat final count is `450`, but pixels can be revisited or toggled back, so the final nonzero count can be much smaller.

Generated large-format figures:

- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_six_sample_delta_error_large.png`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_three_sample_delta_error_large.png`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_two_sample_delta_error_large.png`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_one_sample_delta_error_huge.png`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_three_sample_four_panel_large.png`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_two_sample_four_panel_large.png`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_one_sample_four_panel_huge.png`
- `visualizations/darcy_cflow_loss123_physics_baseline_image_only_20260608/darcy_cflow_loss3_advantage_one_sample_four_panel_split_rows_huge.png`

Observed final changed-pixel counts for the selected six rows:

| row | baseline | loss1 | loss2 | loss3 | physical_source |
|---:|---:|---:|---:|---:|---:|
| 1 | 450 | 418 | 423 | 186 | 446 |
| 2 | 450 | 426 | 449 | 125 | 431 |
| 3 | 450 | 416 | 443 | 163 | 435 |
| 4 | 429 | 412 | 439 | 188 | 435 |
| 5 | 450 | 395 | 384 | 136 | 439 |
| 6 | 450 | 393 | 440 | 62 | 423 |

Inference from these selected examples: `loss3` visually shows fewer nonzero perturbation pixels because the final attack state actually contains fewer changed pixels for these selected weak-tag samples, not because of an image scaling artifact.
