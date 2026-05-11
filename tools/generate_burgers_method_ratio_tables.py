#!/usr/bin/env python3
"""Generate sorted Burgers method-ratio tables with paired best-vs-second tests.

The input is the clean per-index CSV produced by rebuild_burgers_clean_summary.py.
Rows are grouped by model, nu, norm, epsilon, alpha, and steps.  Within each
group, each sample index is counted once.  If the same setting/index appears in
multiple result roots, the preferred batch100 result root is kept so older
batch50 or one-off probe runs do not overweight a setting.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METHODS: list[tuple[str, str]] = [
    ("loss1", "loss1"),
    ("loss2_fixed", "loss2_fixed"),
    ("loss2_dict_N200", "N200"),
    ("loss2_dict_N2000", "N2000"),
    ("loss2_dict_N20000", "N20000"),
    ("loss3_stopgrad", "loss3_sg"),
    ("loss3", "loss3"),
]


def finite_float(text: str | None) -> float:
    try:
        value = float(text) if text not in (None, "") else float("nan")
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def fmt_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    value_abs = abs(value)
    if value_abs >= 1e4 or (0 < value_abs < 1e-3):
        return f"{value:.{digits}g}"
    if value_abs >= 1000:
        return f"{value:.0f}"
    if value_abs >= 100:
        return f"{value:.1f}"
    if value_abs >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def sample_std(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else float("nan")
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def student_t_pdf(x: float, df: int) -> float:
    log_coeff = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    return math.exp(log_coeff - ((df + 1) / 2) * math.log1p((x * x) / df))


def simpson_integral_pdf(a: float, b: float, df: int, n: int) -> float:
    if b <= a:
        return 0.0
    if n % 2:
        n += 1
    h = (b - a) / n
    total = student_t_pdf(a, df) + student_t_pdf(b, df)
    odd = 0.0
    even = 0.0
    for i in range(1, n):
        val = student_t_pdf(a + i * h, df)
        if i % 2:
            odd += val
        else:
            even += val
    return h * (total + 4 * odd + 2 * even) / 3


def student_t_two_sided_p(t_stat: float, df: int) -> float:
    if not math.isfinite(t_stat) or df <= 0:
        return float("nan")
    t_abs = abs(t_stat)
    if t_abs == 0:
        return 1.0
    if t_abs > 80:
        return 0.0
    # Numerical Simpson integration of the symmetric Student-t density.
    # This avoids scipy as a hard dependency.  It is accurate enough for the
    # n=100 paired checks used by the Burgers sweeps.
    n = max(2000, int(t_abs * 800))
    area = simpson_integral_pdf(0.0, t_abs, df, n)
    cdf = min(1.0, 0.5 + area)
    return max(0.0, min(1.0, 2.0 * (1.0 - cdf)))


def paired_t_test(best_values: list[float], second_values: list[float]) -> dict[str, float]:
    diffs = [a - b for a, b in zip(best_values, second_values) if math.isfinite(a) and math.isfinite(b)]
    n = len(diffs)
    if n < 2:
        return {
            "paired_n": float(n),
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "cohen_dz": float("nan"),
        }
    mu = mean(diffs)
    sd = sample_std(diffs)
    if sd == 0:
        t_stat = math.copysign(float("inf"), mu) if mu != 0 else 0.0
        p_value = 0.0 if mu != 0 else 1.0
        dz = math.copysign(float("inf"), mu) if mu != 0 else 0.0
    else:
        t_stat = mu / (sd / math.sqrt(n))
        p_value = student_t_two_sided_p(t_stat, n - 1)
        dz = mu / sd
    return {
        "paired_n": float(n),
        "mean_diff": mu,
        "std_diff": sd,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohen_dz": dz,
    }


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row["model"],
        row["nu"],
        row["norm"],
        row["epsilon"],
        row["alpha"],
        row.get("steps", ""),
    )


def row_signature(row: dict[str, str]) -> tuple[str, ...]:
    fields = ["model", "index", "nu", "norm", "epsilon", "alpha", "steps"]
    for method, _label in METHODS:
        fields.extend([f"{method}_ratio", f"{method}_final_true_loss"])
    return tuple(row.get(field, "") for field in fields)


def source_priority(row: dict[str, str]) -> tuple[int, int]:
    root = row.get("result_root", "")
    if "burgers_future_loss3_bad_search_batch100_random100_sequential" in root:
        priority = 50
    elif "burgers_loss3_good_bad_7settings_batch100_random100_losses" in root:
        priority = 40
    elif "burgers_loss3_candidate_batch100_random100" in root:
        priority = 30
    elif "burgers_loss3_18setting_batch100_random100_losses_parallel6" in root:
        priority = 20
    elif "batch50_random50" in root:
        priority = 10
    else:
        priority = 0
    is_batch100 = 1 if "batch100" in root or "batch100" in row.get("tag", "") else 0
    return (priority, is_batch100)


def sorted_group_items(groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]]):
    def key_fn(item):
        key, _rows = item
        model, nu, norm, epsilon, alpha, steps = key
        model_order = {"FNO": 0, "DeepONet": 1}.get(model, 99)
        norm_order = {"2": 0, "inf": 1}.get(norm, 99)
        return (
            model_order,
            model,
            finite_float(nu),
            norm_order,
            norm,
            finite_float(epsilon),
            finite_float(alpha),
            finite_float(steps),
        )

    return sorted(groups.items(), key=key_fn)


def summarize_group(key: tuple[str, str, str, str, str, str], rows: list[dict[str, str]]) -> dict[str, str]:
    model, nu, norm, epsilon, alpha, steps = key
    n = len(rows)
    roots = sorted({row.get("result_root", "") for row in rows if row.get("result_root", "")})
    summary: dict[str, str] = {
        "model": model,
        "nu": nu,
        "norm": norm,
        "epsilon": epsilon,
        "alpha": alpha,
        "steps": steps,
        "n": str(n),
        "n_unique_indices": str(len({row.get("index", "") for row in rows})),
        "source_roots": ";".join(roots),
    }

    method_means: dict[str, float] = {}
    method_values: dict[str, list[float]] = {}
    for method, _label in METHODS:
        ratios = [finite_float(row.get(f"{method}_ratio")) for row in rows]
        values = [v for v in ratios if math.isfinite(v)]
        method_values[method] = values
        method_means[method] = mean(values)
        summary[f"{method}_ratio_mean"] = repr(method_means[method])
        summary[f"{method}_ratio_std"] = repr(sample_std(values))
        summary[f"{method}_wins"] = "0"

    # Wins are per-row max ratio.  This matches final true loss ranking because
    # all methods in a row share the same initial true loss.
    win_counts = {method: 0 for method, _label in METHODS}
    for row in rows:
        best_method = max(METHODS, key=lambda item: finite_float(row.get(f"{item[0]}_ratio")))[0]
        win_counts[best_method] += 1
    for method, _label in METHODS:
        summary[f"{method}_wins"] = str(win_counts[method])

    ranked = sorted(METHODS, key=lambda item: method_means[item[0]], reverse=True)
    best_method = ranked[0][0]
    second_method = ranked[1][0] if len(ranked) > 1 else ""
    summary["best_method"] = best_method
    summary["best_ratio_mean"] = repr(method_means[best_method])
    summary["second_method"] = second_method
    summary["second_ratio_mean"] = repr(method_means[second_method]) if second_method else ""

    if second_method:
        test = paired_t_test(
            [finite_float(row.get(f"{best_method}_ratio")) for row in rows],
            [finite_float(row.get(f"{second_method}_ratio")) for row in rows],
        )
        for name, value in test.items():
            summary[name] = repr(value)
        p_value = test["p_value"]
        summary["significant_0p05"] = "yes" if math.isfinite(p_value) and p_value < 0.05 else "no"
    return summary


def method_cell(summary: dict[str, str], method: str) -> str:
    wins = summary.get(f"{method}_wins", "0")
    n = summary["n"]
    ratio_mean = finite_float(summary.get(f"{method}_ratio_mean"))
    ratio_std = finite_float(summary.get(f"{method}_ratio_std"))
    return f"{wins}/{n}; {fmt_float(ratio_mean)} +/- {fmt_float(ratio_std)}x"


def best_cell(summary: dict[str, str]) -> str:
    method = summary["best_method"]
    wins = summary.get(f"{method}_wins", "0")
    n = summary["n"]
    ratio_mean = finite_float(summary["best_ratio_mean"])
    ratio_std = finite_float(summary.get(f"{method}_ratio_std"))
    p_value = finite_float(summary.get("p_value"))
    sig = summary.get("significant_0p05", "no")
    return f"{method}; {wins}/{n}; {fmt_float(ratio_mean)} +/- {fmt_float(ratio_std)}x; p={fmt_float(p_value)}; sig={sig}"


def write_markdown(path: Path, summaries: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Burgers Method Ratio Tables With Paired Best-vs-Second Tests")
    lines.append("")
    lines.append("Each method cell is `wins; mean ratio +/- std ratio`.")
    lines.append("The `best` column ranks methods by mean ratio and reports a paired two-sided t-test against the second-best mean-ratio method.")
    lines.append("")

    section_order: list[tuple[str, str, str]] = []
    for row in summaries:
        section = (row["model"], row["nu"], row["norm"])
        if section not in section_order:
            section_order.append(section)

    for model, nu, norm in section_order:
        norm_label = "Linf" if norm == "inf" else f"L{norm}"
        lines.append(f"## {model} nu={nu} {norm_label}")
        lines.append("")
        headers = ["eps", "alpha", "n", *[label for _method, label in METHODS], "best", "second", "t", "p", "sig"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---:"] * 3 + ["---"] * (len(headers) - 3)) + "|")
        section_rows = [r for r in summaries if (r["model"], r["nu"], r["norm"]) == (model, nu, norm)]
        for row in section_rows:
            values = [
                fmt_float(finite_float(row["epsilon"])),
                fmt_float(finite_float(row["alpha"])),
                row["n"],
                *[method_cell(row, method) for method, _label in METHODS],
                best_cell(row),
                f'{row["second_method"]}; {fmt_float(finite_float(row["second_ratio_mean"]))}x',
                fmt_float(finite_float(row.get("t_stat"))),
                fmt_float(finite_float(row.get("p_value"))),
                row.get("significant_0p05", "no"),
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, summaries: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in summaries:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-index-csv", type=Path, default=Path("results/clean_recomputed_summary/per_index_clean.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("results/clean_recomputed_summary/method_ratio_best_vs_second_tests.md"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/clean_recomputed_summary/method_ratio_best_vs_second_tests.csv"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional model filter, e.g. FNO DeepONet.")
    args = parser.parse_args()

    rows_by_setting_index: dict[tuple[str, str, str, str, str, str, str], dict[str, str]] = {}
    seen_signatures: set[tuple[str, ...]] = set()
    with args.per_index_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if args.models and row.get("model") not in set(args.models):
                continue
            sig = row_signature(row)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            key = (*group_key(row), row.get("index", ""))
            old = rows_by_setting_index.get(key)
            if old is None or source_priority(row) > source_priority(old):
                rows_by_setting_index[key] = row

    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows_by_setting_index.values():
        groups[group_key(row)].append(row)

    summaries = [summarize_group(key, rows) for key, rows in sorted_group_items(groups)]
    write_markdown(args.out_md, summaries)
    write_csv(args.out_csv, summaries)
    print(f"[done] groups: {len(summaries)}")
    print(f"[done] wrote markdown: {args.out_md}")
    print(f"[done] wrote csv: {args.out_csv}")


if __name__ == "__main__":
    main()
