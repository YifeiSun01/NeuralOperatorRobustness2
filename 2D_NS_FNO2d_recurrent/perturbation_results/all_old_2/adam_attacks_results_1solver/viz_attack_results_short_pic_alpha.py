import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import pickle
from pathlib import Path
from collections import defaultdict
import json
import re

# -------------------- helpers --------------------

def normalize_attack_tag(tag: str) -> str:
    """Unify tag names & strip optimizer suffix."""
    tag = tag.lower().strip()
    # strip anything like '|adam_b10.9...'
    tag = tag.split("|adam", 1)[0]
    # unify with_solver -> withsolver
    if tag == "with_solver":
        tag = "withsolver"
    return tag

# -------------------- grouping --------------------

def group_pkl_files(folder_path):
    """
    Return a dict grouped by (alpha, epsilon, steps, idx):
      { (alpha=..., epsilon=..., steps=..., idx=...): { attack_key: abs_path, ... }, ... }
    """
    folder = Path(folder_path)
    pkl_files = list(folder.glob("*.pkl"))
    grouped = defaultdict(dict)

    # Strict pattern; accept with_solver or withsolver; accept optional |adam... suffix
    pattern = r"""
        ^pgd_attack_records_
        norm(?P<norm>\d+)_                    # norm
        alpha(?P<alpha>[\d\.]+)_              # alpha
        epsilon(?P<epsilon>[\d\.]+)_          # epsilon
        steps(?P<steps>\d+)_                  # steps
        idx(?P<idx>\d+)_                      # idx
        (?P<attack_type>                      # attack type
            detached\d+(?:to\d+)?(?:_\d+)?
          | constant\d+(?:to\d+)?(?:_\d+)?
          | approximated
          | with_solver
          | withsolver
        )
        (?:\|adam.*)?                         # optional adam suffix
        $
    """

    for path in pkl_files:
        fname = path.stem  # remove .pkl
        m = re.match(pattern, fname, re.VERBOSE | re.IGNORECASE)
        if not m:
            print(f"Warning: cannot parse file name: {fname}")
            continue

        gd = m.groupdict()
        attack_type_raw = gd["attack_type"]
        attack_type = normalize_attack_tag(attack_type_raw)

        param_key = (
            f"alpha={gd['alpha']}",
            f"epsilon={gd['epsilon']}",
            f"steps={gd['steps']}",
            f"idx={gd['idx']}",
        )
        grouped[param_key][attack_type] = str(path.resolve())

    return grouped

def build_index_by_epsilon_tag(grouped):
    """
    Convert { (alpha, epsilon, steps, idx) -> {tag: path} } into
    { (epsilon=..., steps=..., idx=..., tag=...) -> {alpha: path} }.
    """
    index = defaultdict(dict)
    for key_tuple, tag2path in grouped.items():
        # unpack values from key tuple like "alpha=1.0"
        key_map = dict(kv.split("=", 1) for kv in key_tuple)
        alpha = key_map["alpha"]
        epsilon = key_map["epsilon"]
        steps = key_map["steps"]
        idx = key_map["idx"]

        for tag_raw, fpath in tag2path.items():
            tag = normalize_attack_tag(tag_raw)
            new_key = (f"epsilon={epsilon}", f"steps={steps}", f"idx={idx}", f"tag={tag}")
            index[new_key][alpha] = fpath
    return index

# -------------------- plotting --------------------

def visualize_attack_comparison_by_alpha(fixed_key, alpha_to_path, dpi=100, approximated_prefix="surrogate_"):
    """
    One figure per (epsilon, tag, steps, idx). Lines = different alphas.
    Line colors are taken from a continuous colormap at evenly spaced positions
    according to the rank order of alpha (not the numeric distance).
    """
    # Parse fixed key fields
    key_map = dict(kv.split("=", 1) for kv in fixed_key)
    epsilon = key_map["epsilon"]
    steps = int(key_map["steps"])
    idx = key_map["idx"]
    tag = key_map["tag"]

    # Sort alphas and prepare evenly spaced colors by rank
    sorted_alphas = sorted(alpha_to_path.keys(), key=lambda a: float(a))
    n_lines = len(sorted_alphas)
    if n_lines == 0:
        print("No data to plot for:", fixed_key)
        return

    # Use a continuous colormap and sample evenly across it
    cmap = plt.cm.viridis
    # avoid extreme ends to keep contrast
    color_positions = np.linspace(0.07, 0.93, n_lines) if n_lines > 1 else np.array([0.5])
    alpha_color = {a: cmap(p) for a, p in zip(sorted_alphas, color_positions)}

    data_per_alpha = {}
    max_loss = -float("inf")
    min_loss = float("inf")

    print("Loading files...")
    for a in sorted_alphas:
        fpath = alpha_to_path[a]
        with open(fpath, "rb") as f:
            records = pickle.load(f)["steps"]
        data_per_alpha[a] = records

        if tag == "approximated":
            losses = [r[f"{approximated_prefix}loss"] for r in records]
        else:
            losses = [r["loss"] for r in records]

        if losses:
            max_loss = max(max_loss, max(losses))
            min_loss = min(min_loss, min(losses))

    if not data_per_alpha:
        print("No data to plot for:", fixed_key)
        return

    num_steps = len(next(iter(data_per_alpha.values())))
    fig, ax = plt.subplots(figsize=(24, 6))

    # Plot each alpha with rank-based gradient color
    for a in sorted_alphas:
        records = data_per_alpha[a]
        if tag == "approximated":
            losses = [r[f"{approximated_prefix}loss"] for r in records]
            final_loss = records[-1][f"{approximated_prefix}loss"]
        else:
            losses = [r["loss"] for r in records]
            final_loss = records[-1]["loss"]

        ax.plot(
            range(num_steps),
            losses,
            label=f"alpha={a} (final loss={final_loss:.2f})",
            color=alpha_color[a],
            marker="o",
            markersize=3,
            alpha=0.9,
            linewidth=2,
        )

    # Title & axes
    main_title = (
        "PGD Attack Comparison by Alpha\n"
        f"epsilon: {epsilon} | attack: {tag} | steps: {steps} | idx: {idx}"
    )
    ax.set_title(main_title, pad=18)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_xlim(-0.5, num_steps - 0.5)
    if np.isfinite(min_loss) and np.isfinite(max_loss) and min_loss < max_loss:
        ax.set_ylim(min_loss * 0.9, max_loss * 1.1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    ax.grid(True)

    # Optional: a rank-based colorbar for reference
    if n_lines > 1:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=n_lines - 1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015)
        # Show a few representative ticks (min / mid / max) to avoid clutter
        tick_idx = np.unique(np.clip(np.round(np.linspace(0, n_lines - 1, 3)).astype(int), 0, n_lines - 1))
        cbar.set_ticks(tick_idx)
        cbar.set_ticklabels([sorted_alphas[i] for i in tick_idx])
        cbar.set_label("alpha (ranked gradient)")

    plt.tight_layout()

    # Save beside this file
    output_dir = Path(__file__).parent / "comparison_plots_alpha"
    output_dir.mkdir(exist_ok=True)

    suffix = (
        "_surrogate" if tag == "approximated" and approximated_prefix == "surrogate_"
        else "_true" if tag == "approximated" and approximated_prefix == ""
        else ""
    )
    out_name = f"compare_by_alpha_epsilon={epsilon}_tag={tag}_steps={steps}_idx={idx}{suffix}.png"
    out_path = output_dir / out_name
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure to: {out_path}")



# -------------------- main --------------------

if __name__ == "__main__":
    folder_path = Path(__file__).parent / "pickle_files"
    grouped = group_pkl_files(folder_path)

    # Re-index so each plot fixes (epsilon, tag, steps, idx) and varies alpha
    by_eps_tag = build_index_by_epsilon_tag(grouped)

    # Plot all groups
    for fixed_key, alpha_map in by_eps_tag.items():
        visualize_attack_comparison_by_alpha(
            fixed_key,
            alpha_map,
            dpi=100,
            approximated_prefix="surrogate_",  # or "" if plotting true loss for 'approximated'
        )
