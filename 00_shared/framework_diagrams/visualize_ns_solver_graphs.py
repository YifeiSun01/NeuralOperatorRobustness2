#!/usr/bin/env python3
"""Visualize tiny NS solver backward graphs for PyTorch and JAX.

The graphs are intentionally built with only a few PDE time steps. A real
`t_final=20, dt=0.005` rollout would contain thousands of repeated blocks and
would be too large to inspect visually.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter, deque
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def escape_label(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_svg_graph(
    path: Path,
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    title: str,
    max_label_chars: int = 38,
) -> None:
    """Write a small dependency graph as a standalone SVG image.

    The layout is deliberately simple and dependency-free: nodes are layered by
    distance from graph roots and placed left-to-right inside each layer.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for src, dst in edges:
        if src in nodes and dst in nodes:
            outgoing[src].append(dst)
            incoming[dst].append(src)

    roots = [node_id for node_id in nodes if not incoming[node_id]]
    if not roots and nodes:
        roots = [next(iter(nodes))]

    layer: dict[str, int] = {node_id: 0 for node_id in roots}
    queue: deque[str] = deque(roots)
    while queue:
        src = queue.popleft()
        for dst in outgoing[src]:
            next_layer = layer[src] + 1
            if dst not in layer or next_layer > layer[dst]:
                layer[dst] = next_layer
                queue.append(dst)

    for node_id in nodes:
        layer.setdefault(node_id, 0)

    grouped: dict[int, list[str]] = {}
    for node_id, depth in layer.items():
        grouped.setdefault(depth, []).append(node_id)

    for ids in grouped.values():
        ids.sort(key=lambda node_id: nodes[node_id])

    box_w = 260
    box_h = 46
    x_gap = 48
    y_gap = 34
    margin_x = 40
    margin_y = 84

    max_cols = max((len(ids) for ids in grouped.values()), default=1)
    max_layer = max(grouped, default=0)
    width = margin_x * 2 + max_cols * box_w + max(0, max_cols - 1) * x_gap
    height = margin_y + (max_layer + 1) * (box_h + y_gap) + 40

    positions: dict[str, tuple[float, float]] = {}
    for depth in range(max_layer + 1):
        ids = grouped.get(depth, [])
        row_w = len(ids) * box_w + max(0, len(ids) - 1) * x_gap
        x0 = max(margin_x, (width - row_w) / 2)
        y = margin_y + depth * (box_h + y_gap)
        for index, node_id in enumerate(ids):
            positions[node_id] = (x0 + index * (box_w + x_gap), y)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<defs>",
        (
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#3f4b5b" />'
            "</marker>"
        ),
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{margin_x}" y="36" font-family="Arial, sans-serif" '
            f'font-size="22" font-weight="700" fill="#172033">{escape_label(title)}</text>'
        ),
        (
            f'<text x="{margin_x}" y="62" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#5d6b7a">nodes={len(nodes)}, edges={len(edges)}</text>'
        ),
    ]

    for src, dst in edges:
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        start_x = x1 + box_w / 2
        start_y = y1 + box_h
        end_x = x2 + box_w / 2
        end_y = y2
        mid_y = (start_y + end_y) / 2
        lines.append(
            f'<path d="M {start_x:.1f} {start_y:.1f} C {start_x:.1f} {mid_y:.1f}, '
            f'{end_x:.1f} {mid_y:.1f}, {end_x:.1f} {end_y:.1f}" '
            'fill="none" stroke="#3f4b5b" stroke-width="1.2" opacity="0.55" '
            'marker-end="url(#arrow)"/>'
        )

    for node_id, label in nodes.items():
        x, y = positions[node_id]
        short = label if len(label) <= max_label_chars else label[: max_label_chars - 1] + "..."
        fill = "#eef4ff" if "loss" in label.lower() or "output" in label.lower() else "#f7f8fa"
        stroke = "#476fbd" if "loss" in label.lower() or "output" in label.lower() else "#c6ced8"
        lines.extend(
            [
                (
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w}" height="{box_h}" '
                    f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
                ),
                (
                    f'<text x="{x + 12:.1f}" y="{y + 28:.1f}" '
                    'font-family="Menlo, Consolas, monospace" font-size="12" '
                    f'fill="#172033">{escape_label(short)}</text>'
                ),
            ]
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dot(path: Path, nodes: dict[str, str], edges: list[tuple[str, str]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "digraph G {",
        "  rankdir=TB;",
        f'  label="{escape_label(title)}";',
        '  node [shape=box, style="rounded,filled", fillcolor="#f7f8fa"];',
    ]
    for node_id, label in nodes.items():
        lines.append(f'  "{node_id}" [label="{escape_label(label)}"];')
    for src, dst in edges:
        lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(path: Path, framework: str, steps: int, nx: int, labels: Iterable[str]) -> None:
    counts = Counter(labels)
    payload = {
        "framework": framework,
        "case": "ns_2d",
        "time_steps": steps,
        "nx": nx,
        "node_type_counts": dict(counts.most_common()),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect_torch_backward_graph(loss, max_nodes: int) -> tuple[dict[str, str], list[tuple[str, str]]]:
    from torchviz import make_dot

    dot = make_dot(loss)
    return parse_dot_source(dot.source, max_nodes=max_nodes)


def parse_dot_source(source: str, max_nodes: int) -> tuple[dict[str, str], list[tuple[str, str]]]:
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    node_re = re.compile(r'^\s*([A-Za-z0-9_]+)\s+\[label="?(.*?)"?\]')
    edge_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)")

    for line in source.splitlines():
        match = node_re.match(line)
        if match and len(nodes) < max_nodes:
            node_id, label = match.groups()
            clean = label.replace("\\n", " ").strip()
            nodes[node_id] = clean or node_id

    for line in source.splitlines():
        match = edge_re.match(line)
        if not match:
            continue
        src, dst = match.groups()
        if src in nodes and dst in nodes:
            edges.append((src, dst))

    return nodes, edges


def build_torch_graph(args: argparse.Namespace, out_dir: Path) -> None:
    import sys

    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from solvers import NavierStokesVorticity2DZongyiETDRK4

    print("Building PyTorch tiny NS graph...", flush=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    solver = NavierStokesVorticity2DZongyiETDRK4(
        num_points=args.nx,
        domain_extent=args.domain,
        dt=args.dt,
        diffusivity=args.nu,
        drag=0.0,
        device=device,
        dtype=torch.float32,
    )

    omega0 = torch.randn(1, args.nx, args.nx, device=device, dtype=torch.float32, requires_grad=True)
    omega = omega0
    for _ in range(args.steps):
        omega = solver.step(omega)

    target = torch.zeros_like(omega)
    loss = torch.mean((omega - target) ** 2)
    print("Collecting PyTorch autograd nodes...", flush=True)
    nodes, edges = collect_torch_backward_graph(loss, args.max_nodes)

    stem = f"ns_torch_backward_graph_steps{args.steps}_nx{args.nx}"
    write_svg_graph(out_dir / f"{stem}.svg", nodes, edges, f"PyTorch NS backward graph, steps={args.steps}, nx={args.nx}")
    write_dot(out_dir / f"{stem}.dot", nodes, edges, f"PyTorch NS backward graph, steps={args.steps}, nx={args.nx}")
    save_summary(out_dir / f"{stem}_summary.json", "pytorch", args.steps, args.nx, nodes.values())

    print("Running PyTorch backward once to verify the graph is differentiable...", flush=True)
    loss.backward()


def collect_jaxpr_graph(jaxpr, max_nodes: int) -> tuple[dict[str, str], list[tuple[str, str]]]:
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    producer: dict[str, str] = {}

    def var_name(var: object) -> str:
        return str(var)

    for index, invar in enumerate(jaxpr.invars):
        var = var_name(invar)
        node = f"in{index}"
        nodes[node] = f"input {index}: {var}"
        producer[var] = node

    for index, eqn in enumerate(jaxpr.eqns):
        if len(nodes) >= max_nodes:
            break
        node = f"eq{index}"
        primitive = eqn.primitive.name
        nodes[node] = primitive
        for invar in eqn.invars:
            src = producer.get(var_name(invar))
            if src is not None:
                edges.append((src, node))
        for outvar in eqn.outvars:
            producer[var_name(outvar)] = node

    for index, outvar in enumerate(jaxpr.outvars):
        if len(nodes) >= max_nodes:
            break
        src = producer.get(var_name(outvar))
        node = f"out{index}"
        nodes[node] = f"output {index}: {var_name(outvar)}"
        if src is not None:
            edges.append((src, node))

    return nodes, edges


def build_jax_graph(args: argparse.Namespace, out_dir: Path) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp

    print("Building JAX tiny NS value_and_grad JAXPR...", flush=True)
    nx = args.nx
    dx = args.domain / nx
    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=dx)
    kx = kx_1d[:, None]
    ky = ky_1d[None, :]

    derivative_x = 1j * kx
    derivative_y = 1j * ky
    laplace = -(kx**2 + ky**2)
    inv_laplacian = jnp.where(laplace != 0, 1.0 / laplace, 0.0)
    linear_operator = args.nu * laplace

    mode_x = jnp.fft.fftfreq(nx, d=1.0) * nx
    mode_y = jnp.fft.rfftfreq(nx, d=1.0) * nx
    cutoff = (2.0 / 3.0) * (nx / 2.0)
    mask = (jnp.abs(mode_x[:, None]) <= cutoff) & (jnp.abs(mode_y[None, :]) <= cutoff)

    grid = jnp.linspace(0.0, 1.0, nx + 1, dtype=jnp.float32)[:-1]
    xx, yy = jnp.meshgrid(grid, grid, indexing="ij")
    forcing = 0.1 * (jnp.sin(2.0 * jnp.pi * (xx + yy)) + jnp.cos(2.0 * jnp.pi * (xx + yy)))
    injection = jnp.fft.rfft2(forcing)

    h_l = args.dt * linear_operator
    exp_term = jnp.exp(h_l)
    half_exp_term = jnp.exp(0.5 * h_l)
    theta = jnp.pi * (jnp.arange(1, 17, dtype=jnp.float32) - 0.5) / 16.0
    roots = jnp.exp(1j * theta)
    lr = h_l[..., None] + roots
    coef_1 = args.dt * jnp.mean((jnp.exp(lr / 2.0) - 1.0) / lr, axis=-1)
    coef_4 = args.dt * jnp.mean((-4.0 - lr + jnp.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=-1)
    coef_5 = args.dt * jnp.mean((2.0 + lr + jnp.exp(lr) * (-2.0 + lr)) / lr**3, axis=-1)
    coef_6 = args.dt * jnp.mean((-4.0 - 3.0 * lr - lr**2 + jnp.exp(lr) * (4.0 - lr)) / lr**3, axis=-1)

    def nonlinear(omega_hat):
        omega_hat_dealiased = omega_hat * mask
        stream_function_hat = inv_laplacian * omega_hat_dealiased
        velocity_x_hat = derivative_y * stream_function_hat
        velocity_y_hat = -derivative_x * stream_function_hat
        omega_x_hat = derivative_x * omega_hat_dealiased
        omega_y_hat = derivative_y * omega_hat_dealiased

        velocity_x = jnp.fft.irfft2(velocity_x_hat * mask, s=(nx, nx))
        velocity_y = jnp.fft.irfft2(velocity_y_hat * mask, s=(nx, nx))
        omega_x = jnp.fft.irfft2(omega_x_hat * mask, s=(nx, nx))
        omega_y = jnp.fft.irfft2(omega_y_hat * mask, s=(nx, nx))
        convection = velocity_x * omega_x + velocity_y * omega_y
        return -jnp.fft.rfft2(convection) + injection

    def step(omega):
        omega_hat = jnp.fft.rfft2(omega)
        k0 = nonlinear(omega_hat)
        a = half_exp_term * omega_hat + coef_1 * k0
        k1 = nonlinear(a)
        b = half_exp_term * omega_hat + coef_1 * k1
        k2 = nonlinear(b)
        c = half_exp_term * a + coef_1 * (2.0 * k2 - k0)
        k3 = nonlinear(c)
        omega_next_hat = exp_term * omega_hat + coef_4 * k0 + coef_5 * 2.0 * (k1 + k2) + coef_6 * k3
        return jnp.fft.irfft2(omega_next_hat, s=(nx, nx))

    def rollout(u0):
        u = u0
        for _ in range(args.steps):
            u = step(u)
        return u

    def loss_only(u0):
        pred = rollout(u0)
        return jnp.mean(pred * pred)

    value_and_grad = jax.value_and_grad(loss_only)
    u0 = jnp.ones((args.nx, args.nx), dtype=jnp.float32) * 0.01
    print("Tracing JAXPR...", flush=True)
    jaxpr = jax.make_jaxpr(value_and_grad)(u0).jaxpr
    print("Collecting JAXPR nodes...", flush=True)
    nodes, edges = collect_jaxpr_graph(jaxpr, args.max_nodes)

    stem = f"ns_jax_value_and_grad_jaxpr_steps{args.steps}_nx{args.nx}"
    write_svg_graph(out_dir / f"{stem}.svg", nodes, edges, f"JAX NS value_and_grad JAXPR, steps={args.steps}, nx={args.nx}")
    write_dot(out_dir / f"{stem}.dot", nodes, edges, f"JAX NS value_and_grad JAXPR, steps={args.steps}, nx={args.nx}")
    save_summary(out_dir / f"{stem}_summary.json", "jax", args.steps, args.nx, nodes.values())
    (out_dir / f"{stem}.txt").write_text(str(jaxpr) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["pytorch", "jax", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2, help="Number of tiny dt steps to unroll.")
    parser.add_argument("--nx", type=int, default=16, help="Small NS grid size for readable graphs.")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--domain", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--device", default="", help="PyTorch device override, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=360)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "benchmark_results" / "solver_graphs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.framework in {"pytorch", "both"}:
        build_torch_graph(args, out_dir)
    if args.framework in {"jax", "both"}:
        build_jax_graph(args, out_dir)

    print(f"Wrote graph files to {out_dir}")


if __name__ == "__main__":
    main()
