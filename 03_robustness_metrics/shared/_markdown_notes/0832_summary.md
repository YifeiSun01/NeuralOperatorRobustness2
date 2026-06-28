# Top Right Singular Vector Comparison

This directory compares the top right singular vectors of:

- FNO `nu=0.001`
- solver `nu=0.001`
- DeepONet `nu=0.01`
- solver `nu=0.01`

Each sample has:

- `*_lines.png`: top-4 right singular vectors as line plots; titles include singular value, `hi128`, and zero crossings.
- `*_fft.png`: Fourier energy spectra of the same vectors, with `k=128` marked.
- `*_overlay.png`: shape-normalized overlay for direct visual comparison.

Display note: singular vectors are sign-ambiguous, so signs are aligned for plotting. Shapes are scaled by max absolute value; singular-value magnitudes are shown separately in the labels.
