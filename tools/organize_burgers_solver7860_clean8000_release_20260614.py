#!/usr/bin/env python3
"""Build a layered Burgers solver7860/clean8000 release folder.

The script is intentionally non-destructive for source artifacts: it recreates
only the organized output directory, then copies the existing audit, figure,
table, trace, SVD, report, and log files into a readable hierarchy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DATE = "20260614"
RELEASE_NAME = f"burgers_solver7860_clean8000_organized_release_{DATE}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


class Organizer:
    def __init__(self, root: Path, out: Path) -> None:
        self.root = root
        self.out = out
        self.entries: list[dict[str, object]] = []
        self.missing: list[str] = []
        self.source_roots: dict[str, str] = {}

    def note_source(self, label: str, path: Path) -> None:
        self.source_roots[label] = rel(path, self.root) if path.exists() else f"MISSING: {path}"

    def copy_file(self, src: Path, dst: Path, category: str) -> None:
        if not src.exists():
            self.missing.append(rel(src, self.root) if src.is_absolute() else str(src))
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        stat = dst.stat()
        self.entries.append(
            {
                "category": category,
                "source": rel(src, self.root),
                "path": rel(dst, self.out),
                "bytes": stat.st_size,
                "suffix": dst.suffix.lower() or "<none>",
            }
        )

    def copy_tree(self, src: Path, dst: Path, category: str) -> None:
        if not src.exists():
            self.missing.append(rel(src, self.root) if src.is_absolute() else str(src))
            return
        for file_path in sorted(p for p in src.rglob("*") if p.is_file()):
            self.copy_file(file_path, dst / file_path.relative_to(src), category)

    def copy_tree_filtered(
        self,
        src: Path,
        dst: Path,
        category: str,
        excluded_names: set[str] | None = None,
        excluded_prefixes: tuple[str, ...] = (),
    ) -> None:
        if not src.exists():
            self.missing.append(rel(src, self.root) if src.is_absolute() else str(src))
            return
        excluded_names = excluded_names or set()
        for file_path in sorted(p for p in src.rglob("*") if p.is_file()):
            if file_path.name in excluded_names or file_path.name.startswith(excluded_prefixes):
                continue
            self.copy_file(file_path, dst / file_path.relative_to(src), category)

    def copy_files(self, files: Iterable[Path], dst: Path, category: str, base: Path | None = None) -> None:
        for file_path in sorted(files):
            if not file_path.is_file():
                continue
            target = dst / (file_path.relative_to(base) if base else file_path.name)
            self.copy_file(file_path, target, category)

    def write_text(self, dst: Path, text: str, category: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        stat = dst.stat()
        self.entries.append(
            {
                "category": category,
                "source": "<generated>",
                "path": rel(dst, self.out),
                "bytes": stat.st_size,
                "suffix": dst.suffix.lower() or "<none>",
            }
        )

    def upsert_generated_entry(self, path: Path, category: str, byte_count: int) -> None:
        rel_path = rel(path, self.out)
        for entry in self.entries:
            if entry["path"] == rel_path and entry["source"] == "<generated>":
                entry["bytes"] = byte_count
                entry["category"] = category
                entry["suffix"] = path.suffix.lower() or "<none>"
                return
        self.entries.append(
            {
                "category": category,
                "source": "<generated>",
                "path": rel_path,
                "bytes": byte_count,
                "suffix": path.suffix.lower() or "<none>",
            }
        )

    def counts_by_category(self) -> dict[str, int]:
        return dict(sorted(Counter(str(e["category"]) for e in self.entries).items()))

    def counts_by_suffix(self) -> dict[str, int]:
        return dict(sorted(Counter(str(e["suffix"]) for e in self.entries).items()))

    def bytes_by_top_dir(self) -> dict[str, int]:
        totals: defaultdict[str, int] = defaultdict(int)
        for entry in self.entries:
            top = str(entry["path"]).split("/", 1)[0]
            totals[top] += int(entry["bytes"])
        return dict(sorted(totals.items()))

    def render_readme(self) -> str:
        total_bytes = sum(int(e["bytes"]) for e in self.entries)
        lines = [
            "# Burgers Solver7860 Clean8000 Organized Release",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "This folder is a non-destructive, layered copy of the final Burgers time-matched audit artifacts.",
            "It groups the same local results by how you would inspect them: start-here reports, summary tables, figures, dense attack traces, random-model full-suite data, logs, and source-code references.",
            "",
            "## Start Here",
            "",
            "- `00_start_here/`: final markdown reports, audit manifest, and top-level result notes.",
            "- `01_summary_tables/`: clean 52-dataset metrics, attack metrics, robustness/SVD correlations, model versions.",
            "- `02_figures/`: training curves, log-y variants, no-random-clean variants, polished reports, dense six-model attack panels, and summary plots.",
            "- `03_dense_six_model_attack_data/`: five dense groups with manifests, six-model attack curves, summaries, and NPZ traces.",
            "- `04_random_model_full_suite/`: random clean/random solver clean loss, P2Q2 attacks, Jacobian/SVD, and postprocess tables.",
            "- `05_polished_report_data/`: CSV/JSON/log data backing the polished per-model reports.",
            "- `06_logs/`: postprocess and R2 sync logs.",
            "- `07_source_code_and_references/`: scripts and docs that produced or describe the release.",
            "- `08_dense_image_only_bundle_full_copy/`: full copy of the dense image-only bundle as its own subfolder.",
            "",
            "## Counts",
            "",
            f"- Files: {len(self.entries)}",
            f"- Bytes: {total_bytes}",
            f"- Missing expected inputs: {len(self.missing)}",
            "",
            "## Counts By Suffix",
            "",
        ]
        for suffix, count in self.counts_by_suffix().items():
            lines.append(f"- `{suffix}`: {count}")
        lines.extend(["", "## Counts By Category", ""])
        for category, count in self.counts_by_category().items():
            lines.append(f"- `{category}`: {count}")
        lines.extend(["", "## Bytes By Top Directory", ""])
        for top_dir, num_bytes in self.bytes_by_top_dir().items():
            lines.append(f"- `{top_dir}`: {num_bytes}")
        lines.extend(["", "## Source Roots", ""])
        for label, source in sorted(self.source_roots.items()):
            lines.append(f"- `{label}`: `{source}`")
        if self.missing:
            lines.extend(["", "## Missing Expected Inputs", ""])
            for item in self.missing:
                lines.append(f"- `{item}`")
        lines.append("")
        lines.append("Full per-file inventory is in `MANIFEST.json`.")
        lines.append("")
        return "\n".join(lines)

    def write_manifest(self) -> None:
        manifest_path = self.out / "MANIFEST.json"
        readme_path = self.out / "README.md"
        self.upsert_generated_entry(readme_path, "generated_index", 0)
        self.upsert_generated_entry(manifest_path, "generated_manifest", 0)

        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for _ in range(8):
            readme_path.write_text(self.render_readme(), encoding="utf-8")
            self.upsert_generated_entry(readme_path, "generated_index", readme_path.stat().st_size)
            manifest = {
                "release_name": RELEASE_NAME,
                "generated_at_utc": generated_at,
                "output_root": rel(self.out, self.root),
                "source_roots": self.source_roots,
                "missing_expected_inputs": self.missing,
                "file_count": len(self.entries),
                "total_bytes": sum(int(e["bytes"]) for e in self.entries),
                "counts_by_suffix": self.counts_by_suffix(),
                "counts_by_category": self.counts_by_category(),
                "bytes_by_top_dir": self.bytes_by_top_dir(),
                "files": sorted(self.entries, key=lambda e: str(e["path"])),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            old_size = next(int(e["bytes"]) for e in self.entries if e["path"] == "MANIFEST.json")
            new_size = manifest_path.stat().st_size
            self.upsert_generated_entry(manifest_path, "generated_manifest", new_size)
            if old_size == new_size:
                break


def polished_model_label(model_dir: Path) -> str:
    name = model_dir.name
    suffix = "_polished_variable_epoch"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def organize_polished_reports(org: Organizer, polished_root: Path) -> None:
    figures_root = org.out / "02_figures" / "03_polished_reports"
    data_root = org.out / "05_polished_report_data"

    org.copy_tree(polished_root / "run_logs", data_root / "run_logs", "polished_report_generation_logs")

    for model_dir in sorted(polished_root.glob("*_polished_variable_epoch")):
        if not model_dir.is_dir():
            continue
        model = polished_model_label(model_dir)
        report_dir = model_dir / "polished_report"
        for file_path in sorted(report_dir.glob("*")):
            if not file_path.is_file():
                continue
            name = file_path.name
            if file_path.suffix.lower() == ".png":
                if "attack_loss" in name:
                    subdir = "01_attack_loss"
                elif "rmse" in name:
                    subdir = "02_rmse_heatmaps"
                elif "relative_l2" in name:
                    subdir = "03_relative_l2_heatmaps"
                elif "delta_fft" in name:
                    subdir = "04_delta_fft"
                else:
                    subdir = "99_other_figures"
                org.copy_file(file_path, figures_root / model / subdir / name, "polished_report_figures")
            elif file_path.suffix.lower() == ".csv":
                org.copy_file(file_path, data_root / model / "01_polished_report_csv" / name, "polished_report_csv")
            elif file_path.suffix.lower() == ".json":
                org.copy_file(file_path, data_root / model / "02_manifests_json" / name, "polished_report_json")
            else:
                org.copy_file(file_path, data_root / model / "99_other" / name, "polished_report_other")

        for file_path in sorted(model_dir.glob("*")):
            if not file_path.is_file():
                continue
            name = file_path.name
            if file_path.suffix.lower() == ".png":
                if name.startswith("rmse_"):
                    subdir = "05_lineplots_rmse"
                elif name.startswith("relative_l2_"):
                    subdir = "06_lineplots_relative_l2"
                else:
                    subdir = "99_other_figures"
                org.copy_file(file_path, figures_root / model / subdir / name, "polished_lineplot_figures")
            elif file_path.suffix.lower() == ".csv":
                org.copy_file(file_path, data_root / model / "03_lineplot_csv" / name, "polished_lineplot_csv")
            elif file_path.suffix.lower() == ".json":
                org.copy_file(file_path, data_root / model / "02_manifests_json" / name, "polished_report_json")
            else:
                org.copy_file(file_path, data_root / model / "99_other" / name, "polished_report_other")


def build_release(root: Path, out: Path) -> Organizer:
    audit = root / "outputs" / "burgers_timematched_solver7860_clean8000_audit_20260614"
    summary = root / "forensics" / "burgers_six_model_solver7860_clean8000_summary_20260614"
    random_suite = root / "forensics" / "burgers_random_solver7860_clean8000_full_suite_20260614"
    dense_data = root / "forensics" / "burgers_wideparam_loss123_randomsolver7860_clean8000_round00_p2q2_six_model_visuals_20260614"
    dense_figures = root / "visualizations" / "burgers_wideparam_loss123_randomsolver7860_clean8000_comparison_dense_image_only_bundle_20260614"
    polished = root / "visualizations" / "burgers_solver7860_clean8000_polished_reports_20260614"
    logs = root / "run_logs" / "burgers_solver7860_postprocess_20260614"

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    org = Organizer(root, out)
    for label, path in {
        "audit_output": audit,
        "six_model_summary": summary,
        "random_model_full_suite": random_suite,
        "dense_six_model_attack_data": dense_data,
        "dense_six_model_attack_figures": dense_figures,
        "polished_reports": polished,
        "postprocess_logs": logs,
    }.items():
        org.note_source(label, path)

    # 00: start-here reports and manifests.
    start = out / "00_start_here"
    for src in [
        audit / "reports" / "burgers_timematched_full_audit_report.md",
        audit / "manifests" / "audit_manifest.json",
        summary / "README.md",
        random_suite / "postprocess" / "summary.md",
        root / "docs" / "burgers_timematched_full_or_audit_20260614.md",
        root / "docs" / "burgers_solver7860_dense_image_only_bundle_20260614.md",
        root / "docs" / "burgers_solver7860_polished_reports_20260614.md",
        root / "docs" / "burgers_solver7860_clean8000_organized_release_20260614.md",
    ]:
        org.copy_file(src, start / src.name, "start_here_reports")

    # 01: summary tables grouped by question.
    table_root = out / "01_summary_tables"
    table_groups = {
        "01_clean_52dataset": [
            "clean_52dataset_six_models_selected_worktime.csv",
            "clean_generalization_model_summary_selected_worktime.csv",
            "wideparam_retrain_final_summary.csv",
            "wideparam_retrain_eval_split_summary_merged.csv",
            "wideparam_retrain_train_steps_merged.csv",
            "eval_split_curves_by_epoch_wallclock.csv",
            "generalization_dataset_curves_by_epoch_wallclock.csv",
            "wall_clock_by_epoch.csv",
        ],
        "02_attack_52dataset": [
            "attack_52dataset_six_models_selected_worktime_long.csv",
            "p2q2_attack__manifest.json",
        ],
        "03_robustness_25sample": [
            "robustness_25sample_six_models_selected_worktime.csv",
            "jacobian_svd__sample_manifest.csv",
            "jacobian_svd__jacobian_svd_summary.csv",
            "jacobian_svd__top_singular_values_long.csv",
            "jacobian_svd__aggregate_jacobian_svd_summary.csv",
            "jacobian_svd__j_error_times_attack_delta.csv",
            "jacobian_svd__runtime.csv",
        ],
        "04_correlations_and_rankings": [
            "model_level_metric_means_selected_worktime_25sample.csv",
            "metric_correlations_with_attack_selected_worktime_25sample.csv",
            "per_sample_model_rank_similarity_selected_worktime_25sample.csv",
        ],
        "05_model_versions_and_runtime": [
            "selected_model_versions.json",
            "final_metric_summary.json",
            "gpu_preflight.json",
            "done.json",
            "README.md",
        ],
    }
    for group, names in table_groups.items():
        for name in names:
            org.copy_file(audit / "data" / name, table_root / group / name, "summary_tables")

    for src in sorted(summary.glob("*.csv")) + sorted(summary.glob("*.json")):
        org.copy_file(src, table_root / "06_summary_root_originals" / src.name, "summary_root_original_tables")

    # 02: figures.
    fig_root = audit / "figures"
    org.copy_files(fig_root.glob("*.png"), out / "02_figures" / "01_training_curves_linear" / "all_models", "training_curve_figures")
    org.copy_files((fig_root / "no_random_clean").glob("*.png"), out / "02_figures" / "01_training_curves_linear" / "no_random_clean", "training_curve_figures")
    org.copy_files((fig_root / "log_y").glob("*.png"), out / "02_figures" / "02_training_curves_log_y" / "all_models", "training_curve_log_y_figures")
    log_no_random = fig_root / "log_y" / "no_random_clean"
    if log_no_random.exists():
        org.copy_files(log_no_random.glob("*.png"), out / "02_figures" / "02_training_curves_log_y" / "no_random_clean", "training_curve_log_y_figures")

    organize_polished_reports(org, polished)

    dense_image_base = dense_figures / "comparison_dense"
    for group_dir in sorted(dense_image_base.glob("group*")):
        if group_dir.is_dir():
            org.copy_files(
                group_dir.glob("*.png"),
                out / "02_figures" / "04_dense_six_model_attack" / group_dir.name,
                "dense_six_model_attack_figures",
            )

    org.copy_tree(
        dense_figures,
        out / "08_dense_image_only_bundle_full_copy",
        "dense_image_only_bundle_full_copy",
    )

    org.copy_tree(summary / "plots", out / "02_figures" / "05_summary_plots", "summary_figures")

    # 03: dense attack numerical traces.
    org.copy_tree(dense_data, out / "03_dense_six_model_attack_data", "dense_six_model_attack_data")

    # 04: random clean/random solver full suite.
    org.copy_tree(random_suite, out / "04_random_model_full_suite", "random_model_full_suite")

    # 06: logs.
    org.copy_tree_filtered(
        logs,
        out / "06_logs" / "postprocess_and_upload_logs",
        "postprocess_logs",
        excluded_prefixes=("r2_sync_organized_release",),
    )

    # 07: source and references.
    source_root = out / "07_source_code_and_references"
    for src in [
        root / "tools" / "organize_burgers_solver7860_clean8000_release_20260614.py",
        root / "tools" / "audit_burgers_timematched_full_20260614.py",
        root / "tools" / "build_burgers_selected_worktime_six_model_summary_20260613.py",
        root / "tools" / "evaluate_burgers_random_field_final_models_20260613.py",
        root / "tools" / "plot_burgers_training_run_visualizations_variable_epoch.py",
        root / "tools" / "plot_burgers_wideparam_random_field_six_model_one_row_20260613.py",
        root / "tools" / "run_burgers_random_field_final_models_full_suite_20260613.sh",
        root / "tools" / "run_burgers_timematched_strict_random_continuations_20260614.sh",
        root / "tools" / "watch_burgers_solver7860_then_postprocess_20260614.sh",
        root / "EXPERIMENT_LEDGER.md",
    ]:
        org.copy_file(src, source_root / "scripts_and_docs" / src.name, "source_code_references")

    org.write_text(
        source_root / "SOURCE_NOTE.md",
        "\n".join(
            [
                "# Source Note",
                "",
                "This directory contains code and experiment notes relevant to the organized Burgers solver7860/clean8000 release.",
                "The large generated artifacts remain outside git and are copied into this organized release folder for R2 sync.",
                "",
                "Latest known committed result before this organizer was `c916473 Add Burgers polished report refresh`.",
                "",
            ]
        ),
        "source_code_references",
    )

    org.write_manifest()
    return org


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(repo_root() / "outputs" / RELEASE_NAME),
        help="Organized release output directory.",
    )
    args = parser.parse_args()

    root = repo_root()
    out = Path(args.output).resolve()
    if not str(out).startswith(str(root.resolve())):
        raise SystemExit(f"Refusing to write outside repo: {out}")

    org = build_release(root, out)
    print(json.dumps(
        {
            "output": rel(out, root),
            "files": len(org.entries),
            "missing": len(org.missing),
            "counts_by_suffix": org.counts_by_suffix(),
            "counts_by_category": org.counts_by_category(),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
