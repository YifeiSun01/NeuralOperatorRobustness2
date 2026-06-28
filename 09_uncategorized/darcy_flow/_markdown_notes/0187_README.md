# Darcy Flow Figures Bundle With Loss4

This bundle extends the 20260528 clean Darcy figure bundle by adding loss4 attacks.

Loss definitions:

- `loss1`: ||f(A)-f(A0)||
- `loss2`: ||f(A)-g(A0)||
- `loss3`: ||f(A)-g(A)||
- `loss4`: PDE residual plus homogeneous Dirichlet boundary residual

Generated sections:

- `01_multiK_sameK_loss_comparison_steepest_replace/K*_with_loss4`: same-key panels. First row is original, followed by loss1/loss2/loss3/loss4. Batch-mean panels average all 50 samples. Solver/model columns share one global color range across the whole same-key/multi-K bundle, and model-solver panels share one global symmetric range and show rel L2 loss3.
- `02_loss_curves`: no-std, std-shaded-with-std-range, and std-shaded-with-line-range versions. Every multi-panel figure uses a shared y-axis range across its panels.
- `03_loss4_gifs`: loss4 GIFs for the four core methods, using the old 2x3 step-trace layout with A/ΔA/model/solver/model-solver plus a true-loss3 curve panel and current-step marker. It also includes a multi-key GIF for steepest_replace with four rows for loss1/loss2/loss3/loss4 and a bottom true-loss3 curve panel.
- static sections copied from the original clean bundle where unchanged.

Plot style notes: heatmaps for A/model/solver use `viridis`; signed ΔA and model-solver fields use red-blue `coolwarm`; loss/method lines use the old blue/orange/green/red palette.
In loss2/loss3 optimized-objective panels, some raw/steepest curves can be exactly on top of each other because the binary 3/12 coefficient flip has fixed ±9 magnitude, so those methods select identical or nearly identical flip sets.

Run manifest: `loss4_bundle_manifest.json`.
