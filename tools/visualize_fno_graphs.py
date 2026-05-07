#!/usr/bin/env python3
"""Visualize tiny FNO computation graphs for PyTorch and JAX.

This script draws operation-level graphs for the FNO models with either the
default four Fourier layers or a single Fourier layer. It uses small dummy
inputs so that the graphs remain inspectable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from visualize_ns_solver_graphs import (  # noqa: E402
    collect_jaxpr_graph,
    parse_dot_source,
    save_summary,
    write_dot,
    write_svg_graph,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def torch_make_graph(case: str, layers: int, args: argparse.Namespace, out_dir: Path) -> None:
    import torch
    from torchviz import make_dot

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else "cpu")

    if case == "burgers_1d":
        mod = load_module("fno1d_torch_graph_mod", PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d.py")
        model = mod.FNO1d(modes=args.modes_1d, width=args.width_1d, num_layers=layers, dtype=torch.float32).to(device)
        x = torch.randn(args.batch_size, args.nx_1d, 1, device=device, dtype=torch.float32, requires_grad=True)
    elif case == "ns_2d":
        mod = load_module("fno2d_torch_graph_mod", PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d.py")
        model = mod.FNO2d(
            modes1=args.modes_2d,
            modes2=args.modes_2d,
            width=args.width_2d,
            num_layers=layers,
            in_channels=args.in_channels_2d,
        ).to(device)
        x = torch.randn(
            args.batch_size,
            args.nx_2d,
            args.nx_2d,
            args.in_channels_2d,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
    else:
        raise ValueError(case)

    y = model(x)
    loss = torch.mean(y * y)
    dot = make_dot(loss, params=dict(model.named_parameters()))
    nodes, edges = parse_dot_source(dot.source, max_nodes=args.max_nodes)

    stem = f"{case}_fno_torch_backward_graph_layers{layers}"
    title = f"PyTorch {case} FNO backward graph, layers={layers}"
    write_svg_graph(out_dir / f"{stem}.svg", nodes, edges, title)
    write_dot(out_dir / f"{stem}.dot", nodes, edges, title)
    save_summary(out_dir / f"{stem}_summary.json", "pytorch", layers, 0, nodes.values())
    summary_path = out_dir / f"{stem}_metadata.json"
    summary_path.write_text(
        json.dumps(
            {
                "framework": "pytorch",
                "case": case,
                "num_layers": layers,
                "input_shape": tuple(x.shape),
                "output_shape": tuple(y.shape),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_type_counts": dict(Counter(nodes.values()).most_common(30)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    loss.backward()


def jax_make_graph(case: str, layers: int, args: argparse.Namespace, out_dir: Path) -> None:
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(args.seed)

    if case == "burgers_1d":
        mod = load_module(
            "fno1d_jax_graph_mod",
            PROJECT_ROOT / "1D_Burgers" / "models" / "FNO1d_jax_real_imag.py",
        )
        params = mod.init_fno1d_params(
            key,
            modes=args.modes_1d,
            width=args.width_1d,
            num_layers=layers,
            dtype=jnp.float32,
        )
        x = jnp.ones((args.batch_size, args.nx_1d, 1), dtype=jnp.float32) * 0.01

        def apply(p, z):
            return mod.fno1d_apply(
                p,
                z,
                modes=args.modes_1d,
                width=args.width_1d,
                num_layers=layers,
                dtype=jnp.float32,
            )

    elif case == "ns_2d":
        mod = load_module(
            "fno2d_jax_graph_mod",
            PROJECT_ROOT / "2D_NS_FNO2d_recurrent" / "models" / "FNO2d_jax_real_imag.py",
        )
        params = mod.init_fno2d_params(
            key,
            modes1=args.modes_2d,
            modes2=args.modes_2d,
            width=args.width_2d,
            num_layers=layers,
            in_channels=args.in_channels_2d,
            dtype=jnp.float32,
        )
        x = jnp.ones((args.batch_size, args.nx_2d, args.nx_2d, args.in_channels_2d), dtype=jnp.float32) * 0.01

        def apply(p, z):
            return mod.fno2d_apply(
                p,
                z,
                modes1=args.modes_2d,
                modes2=args.modes_2d,
                width=args.width_2d,
                num_layers=layers,
                dtype=jnp.float32,
            )

    else:
        raise ValueError(case)

    def loss_only(p, z):
        y = apply(p, z)
        return jnp.mean(y * y)

    value_and_grad = jax.value_and_grad(loss_only, argnums=(0, 1))
    jaxpr = jax.make_jaxpr(value_and_grad)(params, x).jaxpr
    nodes, edges = collect_jaxpr_graph(jaxpr, max_nodes=args.max_nodes)

    stem = f"{case}_fno_jax_value_and_grad_jaxpr_layers{layers}"
    title = f"JAX {case} FNO value_and_grad JAXPR, layers={layers}"
    write_svg_graph(out_dir / f"{stem}.svg", nodes, edges, title)
    write_dot(out_dir / f"{stem}.dot", nodes, edges, title)
    save_summary(out_dir / f"{stem}_summary.json", "jax", layers, 0, nodes.values())
    (out_dir / f"{stem}.txt").write_text(str(jaxpr) + "\n", encoding="utf-8")
    (out_dir / f"{stem}_metadata.json").write_text(
        json.dumps(
            {
                "framework": "jax",
                "case": case,
                "num_layers": layers,
                "input_shape": tuple(x.shape),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_type_counts": dict(Counter(nodes.values()).most_common(30)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["pytorch", "jax", "both"], default="both")
    parser.add_argument("--case", choices=["burgers_1d", "ns_2d", "both"], default="both")
    parser.add_argument("--layers", default="4,1", help="Comma-separated layer counts to draw.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nx-1d", type=int, default=16)
    parser.add_argument("--modes-1d", type=int, default=4)
    parser.add_argument("--width-1d", type=int, default=4)
    parser.add_argument("--nx-2d", type=int, default=8)
    parser.add_argument("--modes-2d", type=int, default=3)
    parser.add_argument("--width-2d", type=int, default=4)
    parser.add_argument("--in-channels-2d", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=420)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "benchmark_results" / "fno_graphs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = ["burgers_1d", "ns_2d"] if args.case == "both" else [args.case]
    frameworks = ["pytorch", "jax"] if args.framework == "both" else [args.framework]
    layers = [int(value) for value in args.layers.split(",") if value.strip()]

    for case in cases:
        for layer_count in layers:
            for framework in frameworks:
                print(f"Drawing {framework} {case} FNO graph with layers={layer_count}...", flush=True)
                if framework == "pytorch":
                    torch_make_graph(case, layer_count, args, out_dir)
                else:
                    jax_make_graph(case, layer_count, args, out_dir)

    print(f"Wrote FNO graph files to {out_dir}")


if __name__ == "__main__":
    main()
