#!/usr/bin/env python3
"""Plot loss-gradient path summaries from postprocessed CSV files.

This script intentionally uses only the Python standard library plus
matplotlib, because the Vast container has had a few fragile pandas installs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ATTACK_LABELS = {
    "loss1": "optimize loss1",
    "loss2": "optimize loss2",
    "loss3": "optimize loss3",
}

ATTACK_COLORS = {
    "loss1": "#2a6fbb",
    "loss2": "#00866e",
    "loss3": "#c44e52",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def sort_by_k(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: as_int(r, "k"))


def write_target_table(
    rows_by_attack_k: list[dict[str, str]], out_md: Path, out_csv: Path
) -> None:
    by_attack: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows_by_attack_k:
        by_attack[row["attack_loss"]].append(row)
    for rows in by_attack.values():
        rows.sort(key=lambda r: as_int(r, "k"))

    ks = [as_int(r, "k") for r in by_attack["loss1"]]

    csv_rows: list[dict[str, str]] = []
    for k in ks:
        r1 = next(r for r in by_attack["loss1"] if as_int(r, "k") == k)
        r2 = next(r for r in by_attack["loss2"] if as_int(r, "k") == k)
        r3 = next(r for r in by_attack["loss3"] if as_int(r, "k") == k)
        l1_l3 = as_float(r1, "loss3_value_mean")
        l2_l3 = as_float(r2, "loss3_value_mean")
        direct_l3 = as_float(r3, "loss3_value_mean")
        csv_rows.append(
            {
                "k": str(k),
                "loss1_path_target_loss1": f"{as_float(r1, 'loss1_value_mean'):.6f}",
                "loss1_path_same_delta_loss3": f"{l1_l3:.6f}",
                "loss1_path_budget": f"{as_float(r1, 'delta_budget_ratio_mean'):.6f}",
                "loss2_path_target_loss2": f"{as_float(r2, 'loss2_value_mean'):.6f}",
                "loss2_path_same_delta_loss3": f"{l2_l3:.6f}",
                "loss2_path_budget": f"{as_float(r2, 'delta_budget_ratio_mean'):.6f}",
                "loss3_path_direct_loss3": f"{direct_l3:.6f}",
                "loss3_path_budget": f"{as_float(r3, 'delta_budget_ratio_mean'):.6f}",
                "direct_minus_loss1_path_loss3": f"{direct_l3 - l1_l3:.6f}",
                "direct_minus_loss2_path_loss3": f"{direct_l3 - l2_l3:.6f}",
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# FNO nu=0.001 Target Loss And Same-Delta Loss3 By Step",
        "",
        "Rows are averaged over the five initial conditions. Each attack path is evaluated at the same saved delta_k.",
        "",
        "| k | optimize loss1: target loss1 (same-delta loss3, budget) | optimize loss2: target loss2 (same-delta loss3, budget) | optimize loss3: direct loss3 (budget) | direct loss3 - loss1-path loss3 | direct loss3 - loss2-path loss3 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in csv_rows:
        lines.append(
            "| {k} | `{l1} (L3={l13}, b={b1})` | `{l2} (L3={l23}, b={b2})` | `{l3} (b={b3})` | `{d1}` | `{d2}` |".format(
                k=row["k"],
                l1=f"{float(row['loss1_path_target_loss1']):.4f}",
                l13=f"{float(row['loss1_path_same_delta_loss3']):.4f}",
                b1=f"{float(row['loss1_path_budget']):.3f}",
                l2=f"{float(row['loss2_path_target_loss2']):.4f}",
                l23=f"{float(row['loss2_path_same_delta_loss3']):.4f}",
                b2=f"{float(row['loss2_path_budget']):.3f}",
                l3=f"{float(row['loss3_path_direct_loss3']):.4f}",
                b3=f"{float(row['loss3_path_budget']):.3f}",
                d1=f"{float(row['direct_minus_loss1_path_loss3']):+.4f}",
                d2=f"{float(row['direct_minus_loss2_path_loss3']):+.4f}",
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Early and middle steps are not a clean win for direct loss3, because the loss1/loss2 paths use much more L2 budget.",
            "- At the final saved step k=50, direct loss3 gives the largest mean loss3: 4.6171 vs 4.0678 on the loss1 path and 3.6982 on the loss2 path.",
            "- Therefore the correct statement is not that loss1/loss2 always produce lower loss3. The cleaner statement is: after comparable late-stage attack progress, direct loss3 is the most targeted way to raise loss3, while loss1/loss2 mainly drive model output movement.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines))


def plot_dashboard(rows_by_k: list[dict[str, str]], rows_by_attack_k: list[dict[str, str]], out_path: Path) -> None:
    by_attack: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows_by_attack_k:
        by_attack[row["attack_loss"]].append(row)
    for rows in by_attack.values():
        rows.sort(key=lambda r: as_int(r, "k"))

    ks = [as_int(r, "k") for r in rows_by_k]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("FNO nu=0.001 loss-gradient path summary", fontsize=15, fontweight="bold")

    ax = axes[0][0]
    ax.plot(ks, [as_float(r, "angle_g1_g2_deg_mean") for r in rows_by_k], marker="o", label="angle g1-g2")
    ax.plot(ks, [as_float(r, "angle_g1_g3_deg_mean") for r in rows_by_k], marker="o", label="angle g1-g3")
    ax.plot(ks, [as_float(r, "angle_g2_g3_deg_mean") for r in rows_by_k], marker="o", label="angle g2-g3")
    ax.set_title("Gradient angle by saved step")
    ax.set_xlabel("attack step k")
    ax.set_ylabel("angle (deg)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[0][1]
    ax.plot(ks, [as_float(r, "loss1_value_mean") for r in rows_by_k], marker="o", label="loss1")
    ax.plot(ks, [as_float(r, "loss2_value_mean") for r in rows_by_k], marker="o", label="loss2")
    ax.plot(ks, [as_float(r, "loss3_value_mean") for r in rows_by_k], marker="o", label="loss3")
    ax.set_title("Mean loss values across all paths")
    ax.set_xlabel("attack step k")
    ax.set_ylabel("mean value")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1][0]
    for attack in ("loss1", "loss2", "loss3"):
        rows = by_attack[attack]
        ax.plot(
            [as_int(r, "k") for r in rows],
            [as_float(r, "loss3_value_mean") for r in rows],
            marker="o",
            color=ATTACK_COLORS[attack],
            label=ATTACK_LABELS[attack],
        )
    ax.set_title("Same-delta loss3 by optimized attack path")
    ax.set_xlabel("attack step k")
    ax.set_ylabel("mean loss3")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1][1]
    for attack in ("loss1", "loss2", "loss3"):
        rows = by_attack[attack]
        ax.plot(
            [as_int(r, "k") for r in rows],
            [as_float(r, "delta_budget_ratio_mean") for r in rows],
            marker="o",
            color=ATTACK_COLORS[attack],
            label=ATTACK_LABELS[attack],
        )
    ax.set_title("Mean L2 budget ratio by attack path")
    ax.set_xlabel("attack step k")
    ax.set_ylabel("||delta||_2 / epsilon")
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_target_vs_loss3(rows_by_attack_k: list[dict[str, str]], out_path: Path) -> None:
    by_attack: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows_by_attack_k:
        by_attack[row["attack_loss"]].append(row)
    for rows in by_attack.values():
        rows.sort(key=lambda r: as_int(r, "k"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, constrained_layout=True)
    fig.suptitle("Target objective versus same-delta loss3", fontsize=15, fontweight="bold")

    target_keys = {
        "loss1": "loss1_value_mean",
        "loss2": "loss2_value_mean",
        "loss3": "loss3_value_mean",
    }
    target_names = {
        "loss1": "target loss1",
        "loss2": "target loss2",
        "loss3": "target loss3",
    }

    for ax, attack in zip(axes, ("loss1", "loss2", "loss3")):
        rows = by_attack[attack]
        ks = [as_int(r, "k") for r in rows]
        target = [as_float(r, target_keys[attack]) for r in rows]
        loss3 = [as_float(r, "loss3_value_mean") for r in rows]
        budget = [as_float(r, "delta_budget_ratio_mean") for r in rows]

        ax.plot(ks, target, marker="o", color=ATTACK_COLORS[attack], label=target_names[attack])
        if attack != "loss3":
            ax.plot(ks, loss3, marker="s", color="#444444", label="same-delta loss3")
        ax.set_title(ATTACK_LABELS[attack])
        ax.set_xlabel("attack step k")
        ax.set_ylabel("mean loss value")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="upper left")

        ax2 = ax.twinx()
        ax2.plot(ks, budget, linestyle="--", color="#888888", alpha=0.7, label="budget ratio")
        ax2.set_ylim(0.0, 1.08)
        ax2.set_ylabel("budget ratio")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_angles_by_attack(rows_by_attack_k: list[dict[str, str]], out_path: Path) -> None:
    by_attack: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows_by_attack_k:
        by_attack[row["attack_loss"]].append(row)
    for rows in by_attack.values():
        rows.sort(key=lambda r: as_int(r, "k"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    fig.suptitle("Gradient angles along each optimized attack path", fontsize=15, fontweight="bold")

    for ax, attack in zip(axes, ("loss1", "loss2", "loss3")):
        rows = by_attack[attack]
        ks = [as_int(r, "k") for r in rows]
        ax.plot(ks, [as_float(r, "angle_g1_g2_deg_mean") for r in rows], marker="o", label="g1-g2")
        ax.plot(ks, [as_float(r, "angle_g1_g3_deg_mean") for r in rows], marker="o", label="g1-g3")
        ax.plot(ks, [as_float(r, "angle_g2_g3_deg_mean") for r in rows], marker="o", label="g2-g3")
        ax.set_title(ATTACK_LABELS[attack])
        ax.set_xlabel("attack step k")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("angle (deg)")
        ax.legend(frameon=False)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(
            "results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis"
        ),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--table-md",
        type=Path,
        default=Path("docs/fno_nu0p001_loss_gradient_path_target_loss3_table_20260515.md"),
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=Path(
            "results/fno_nu0p001_loss_gradient_path_steps50_save5_gpu_nocudnn_20260515_200631/gradient_direction_analysis/target_vs_loss3_by_attack_loss_and_k.csv"
        ),
    )
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_md.parent.mkdir(parents=True, exist_ok=True)
    args.table_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_by_k = sort_by_k(read_csv(args.analysis_dir / "summary_by_k.csv"))
    rows_by_attack_k = read_csv(args.analysis_dir / "summary_by_attack_loss_and_k.csv")

    dashboard = args.figure_dir / "fno_nu0p001_loss_gradient_path_dashboard_20260515.png"
    target = args.figure_dir / "fno_nu0p001_loss_gradient_path_target_vs_loss3_20260515.png"
    angles = args.figure_dir / "fno_nu0p001_loss_gradient_path_angles_by_attack_20260515.png"

    plot_dashboard(rows_by_k, rows_by_attack_k, dashboard)
    plot_target_vs_loss3(rows_by_attack_k, target)
    plot_angles_by_attack(rows_by_attack_k, angles)
    write_target_table(rows_by_attack_k, args.table_md, args.table_csv)

    print(f"wrote {dashboard}")
    print(f"wrote {target}")
    print(f"wrote {angles}")
    print(f"wrote {args.table_md}")
    print(f"wrote {args.table_csv}")


if __name__ == "__main__":
    main()
