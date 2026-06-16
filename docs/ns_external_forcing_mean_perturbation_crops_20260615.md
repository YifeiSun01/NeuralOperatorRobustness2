# NS External Forcing Mean Perturbation Crop Stitch 20260615

Status: crop-stitch figures generated from the user-uploaded screenshots stored
in the Codex session JSONL, plus local rendered notebook images used during the
initial search. The raw NS external-forcing attack tensors are not present in
this workspace.

## Question

Build a visual summary for the claim that, in 2D Navier-Stokes with external
forcing, adversarial attack perturbations averaged over many random-field
initial conditions acquire a coherent pattern close to the external forcing.
For the `none` forcing case, the expected averaged perturbation should not show
a coherent forcing template.

The requested fallback was to crop and stitch existing rendered figures because
the original tensor data appears to be unavailable locally. After the user
pointed back to the first message, the uploaded screenshots were recovered from
the local Codex session log as `data:image/png;base64` payloads.

## Observed Local Evidence

- Observed from
  `2D_NS_FNO2d_recurrent/external_forcing_patterns/draw_patterns.ipynb`:
  formulas for six named forcing patterns: `ringsCos`, `sBands`,
  `isoCircles`, `petals`, `ringsL1`, and `ringsLinf`.
- Observed from `solvers.py` and `EXPONAX_SOLVER_CODE_WALKTHROUGH.md`: the
  default 45-degree forcing has the form
  `0.1 * (sin(2*pi*(x+y)) + cos(2*pi*(x+y)))`.
- Observed from the first user message decoded from
  `/root/.codex/sessions/2026/06/15/rollout-2026-06-15T12-17-17-019ecb36-f7f7-76e3-ac1d-6f9e68bb15f6.jsonl`:
  uploaded averaged `pert/orig/diff` screenshots are available for
  `diagonalinit0` / 45-degree default, `ringsCos`, `sBands`, `isoCircles`,
  `petals`, and `none`.
- Observed from rendered notebook outputs copied under
  `analysis_outputs/ns_external_forcing_summary_20260615/raw_notebook_images/`:
  duplicate local crop sources for averaged first/last-frame perturbation or
  difference images are available for `ringsCos`, `isoCircles`, and `petals`.
- Observed from
  `2D_NS_FNO2d_recurrent/external_forcing_patterns/datasets/all_patterns/none/read_data.ipynb`:
  local `none` crops are random-field sequence images, not attack-average
  perturbation images.
- Observed from local filesystem search: no local `.pt`, `.pth`, `.npz`,
  `.npy`, or `.pkl` tensors for the external-forcing attack datasets are
  present under `2D_NS_FNO2d_recurrent/external_forcing_patterns/`.

## Remote Evidence From Logs

The local expansion logs point to completed attack datasets on the historical
remote `/blue/...` path, for example:

- `2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/data_expand_12518522.out`:
  `isoCircles_attack_N1150_train.pt` and `petals_attack_N1150_train.pt`.
- `2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/data_expand_14493732.out`:
  `ringsCos_attack_N1150_train.pt`.
- `2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/data_expand_14536799.out`:
  `sBands_attack_N1150_train.pt`.
- `2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/data_expand_14536809.out`:
  `ringsL1_attack_N1150_train.pt`.
- `2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/data_expand_14608143.out`:
  `ringsLinf_attack_N1150_train.pt`.
- `2D_NS_FNO2d_recurrent/external_forcing_patterns/dataset_expansion/logs/data_expand_14493728.out`:
  `none_attack_N1150_train.pt`.

These are log records only in the current workspace; the actual remote tensors
are not mounted here.

## Generated Files

- Repro script:
  `tools/make_ns_external_forcing_crop_stitch.py`.
- Repro script for the uploaded screenshot version:
  `tools/make_ns_external_forcing_from_uploaded_screenshots.py`.
- Cropped tiles:
  `analysis_outputs/ns_external_forcing_summary_20260615/cropped_tiles/`.
- Cropped tiles from the uploaded screenshots:
  `analysis_outputs/ns_external_forcing_summary_20260615/uploaded_screenshot_crops/`.
- Available-evidence figure:
  `analysis_outputs/ns_external_forcing_summary_20260615/ns_external_forcing_attack_perturbation_crops_available_only.png`.
- Full status figure with missing cells marked:
  `analysis_outputs/ns_external_forcing_summary_20260615/ns_external_forcing_attack_perturbation_crops_full_status.png`.
- Uploaded-screenshot 8-row figure, using uploaded external-forcing tiles and
  `t=0/t=19` attack frames:
  `analysis_outputs/ns_external_forcing_summary_20260615/ns_external_forcing_attack_perturbation_uploaded_8row.png`.
- Uploaded-screenshot available-row figure with no empty attack-average rows,
  using uploaded external-forcing tiles and `t=0/t=19` attack frames:
  `analysis_outputs/ns_external_forcing_summary_20260615/ns_external_forcing_attack_perturbation_uploaded_available_rows.png`.
- Polished uploaded-screenshot figures now use publication-facing column
  labels (`original initial condition`, `perturbed initial condition`, `added
  perturbation`, `original final condition`, `perturbed final condition`, and
  the final-condition difference), include explicit `T = 0` / `T = 19` labels,
  and draw full-column black outlines around the external-forcing column and the
  added-perturbation column.

## Interpretation

Observed from the uploaded-screenshot crop-stitch figures: external forcing is
now cropped directly from the uploaded `Field f` / six-forcing grid images, not
recomputed from formulas. The standalone 45-degree `Field f` crop is vertically
flipped to match the attack-grid image orientation; the six-forcing-grid tiles
keep their original uploaded orientation. The attack panels use the first frame
(`t=0`) and the penultimate meaningful frame (`t=19`), avoiding the `t=20`
cells that are constant/blank in the 45-degree uploaded grid. The averaged
delta/difference crops for the recovered rows
(`diagonalinit0` / 45-degree default, `ringsCos`, `sBands`, `isoCircles`,
`petals`) have coherent large-scale structures that visually align with their
external-forcing families. The `none` row is random-looking rather than a fixed
external-forcing template.

The figure subtitle now states the averaging basis explicitly: each row averages
attack results over 1150 random-field initial conditions.

Inference from the available evidence and expansion logs: a complete 8-row
version should be possible if the historical `/blue/.../expanded_datasets/*.pt`
tensors or their saved rendered average figures are restored. Without those
files, the locally generated 8-row figure leaves the missing attack-average
cells blank instead of inventing them.

## Missing Locally

The recovered first-message upload and all other locally stored Codex session
input images do not contain averaged `pert/orig/diff` screenshots for `ringsL1`
or `ringsLinf`; they contain two `ringsCos avg` screenshots instead. The 8-row
uploaded-screenshot figure therefore marks only `ringsL1` and `ringsLinf` as
missing. The available-row figure omits those two rows and has no empty
attack-average cells.
