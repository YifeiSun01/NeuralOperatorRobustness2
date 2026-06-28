#!/usr/bin/env python3
"""Build crop-stitched NS external-forcing perturbation summary figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("matplotlib is required for colormaps") from exc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "ns_external_forcing_summary_20260615"
TILES = OUT / "cropped_tiles"

AVAILABLE_OUT = OUT / "ns_external_forcing_attack_perturbation_crops_available_only.png"
FULL_STATUS_OUT = OUT / "ns_external_forcing_attack_perturbation_crops_full_status.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
    ):
        path = base / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(36, True)
FONT_SUBTITLE = font(18)
FONT_HEADER = font(25)
FONT_LABEL = font(20)
FONT_NOTE = font(15)
FONT_SMALL = font(14)


def r_cos(x: np.ndarray, y: np.ndarray, x0: float = 0.5, y0: float = 0.5) -> np.ndarray:
    dx = 2 * np.pi * (x - x0)
    dy = 2 * np.pi * (y - y0)
    return np.sqrt((2 - 2 * np.cos(dx)) + (2 - 2 * np.cos(dy)))


def wrap01(u: np.ndarray) -> np.ndarray:
    return u - np.round(u)


def r_torus_lp(
    x: np.ndarray,
    y: np.ndarray,
    p: float,
    x0: float = 0.5,
    y0: float = 0.5,
) -> np.ndarray:
    ax = np.abs(wrap01(x - x0))
    ay = np.abs(wrap01(y - y0))
    if np.isinf(p):
        return np.maximum(ax, ay)
    return (ax**p + ay**p) ** (1.0 / p)


def forcing_field(name: str, n: int = 256) -> np.ndarray | None:
    xs = np.linspace(0.0, 1.0, n, endpoint=False)
    ys = np.linspace(0.0, 1.0, n, endpoint=False)
    x, y = np.meshgrid(xs, ys, indexing="xy")

    if name == "45 deg default":
        return 0.1 * (np.sin(2 * np.pi * (x + y)) + np.cos(2 * np.pi * (x + y)))
    if name == "ringsCos":
        return np.cos(2.5 * r_cos(x, y, 0.5, 0.5))
    if name == "sBands":
        s = 0.12 * np.sin(2 * np.pi * x) + 0.20 * (1.0 - np.cos(2 * np.pi * x))
        phase = 2 * np.pi * (y + s)
        return np.sin(phase) + 0.6 * np.cos(phase + np.pi / 15)
    if name == "isoCircles":
        return np.cos(4 * np.pi * x) + np.cos(4 * np.pi * y) + 0.5 * np.cos(2 * np.pi * (x + y))
    if name == "petals":
        rc = r_cos(x, y, 0.5, 0.5)
        p = np.sin(2 * np.pi * (x - 0.5))
        q = np.sin(2 * np.pi * (y - 0.5))
        z = (p + 1j * q) / np.sqrt(p * p + q * q + 1e-6)
        return np.cos(0.5 * rc) * np.real(z**2)
    if name == "ringsL1":
        return np.cos(2 * np.pi * r_torus_lp(x, y, 1.3, 0.5, 0.5))
    if name == "ringsLinf":
        return np.cos(2 * np.pi * (2.0 * r_torus_lp(x, y, 5.0, 0.5, 0.5)))
    if name == "none":
        return None
    raise KeyError(name)


def array_tile(values: np.ndarray, size: int, cmap_name: str = "viridis") -> Image.Image:
    v = values.astype(np.float64)
    denom = float(np.nanmax(v) - np.nanmin(v))
    if denom <= 1e-12:
        scaled = np.zeros_like(v)
    else:
        scaled = (v - np.nanmin(v)) / denom
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(scaled)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGB").resize((size, size), Image.Resampling.BICUBIC)


def load_crop(path: Path, size: int) -> Image.Image | None:
    if not path.exists():
        return None
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font_obj: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (40, 40, 40),
    spacing: int = 4,
) -> None:
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_obj)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total_h) // 2
    for line, w, h in zip(lines, widths, heights):
        x = box[0] + (box[2] - box[0] - w) // 2
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += h + spacing


def paste_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image | None,
    x: int,
    y: int,
    size: int,
    border: tuple[int, int, int],
    width: int = 3,
    missing: str | None = None,
    caption: str | None = None,
) -> None:
    if image is None:
        draw.rectangle([x, y, x + size, y + size], fill=(248, 248, 248), outline=(238, 238, 238), width=1)
        if missing:
            draw_centered(draw, (x, y, x + size, y + size), missing, FONT_SMALL, fill=(130, 130, 130))
    else:
        canvas.paste(image, (x, y))
        draw.rectangle([x, y, x + size, y + size], outline=border, width=width)
    if caption:
        draw_centered(draw, (x - 20, y + size + 6, x + size + 20, y + size + 46), caption, FONT_SMALL)


def build(rows: list[str], output: Path, full_status: bool) -> None:
    size = 210
    left = 250
    top = 200
    row_gap = 80
    col_gap = 120
    headers = ["External forcing", "Mean delta, t=0", "Mean delta, t=20"]
    cols = len(headers)
    width = left + cols * size + (cols - 1) * col_gap + 90
    height = top + len(rows) * (size + row_gap) + 95

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = "2D NS attacks: mean perturbations follow external forcing"
    subtitle = (
        "Random-field initial conditions are attacked and grouped by forcing tag/time.\n"
        "Averaging many perturbations reveals the forcing pattern; no forcing gives no coherent template."
    )
    draw_centered(draw, (0, 18, width, 70), title, FONT_TITLE, fill=(10, 10, 10))
    draw_centered(draw, (40, 83, width - 40, 135), subtitle, FONT_SUBTITLE, fill=(45, 45, 45))

    x_positions = [left + i * (size + col_gap) for i in range(cols)]
    for x, header in zip(x_positions, headers):
        draw_centered(draw, (x - 20, top - 62, x + size + 20, top - 20), header, FONT_HEADER, fill=(20, 20, 20))

    for r, name in enumerate(rows):
        y = top + r * (size + row_gap)
        label = name
        draw_centered(draw, (25, y, left - 25, y + size), label, FONT_LABEL, fill=(20, 20, 20))

        field = forcing_field(name)
        forcing_img = array_tile(field, size) if field is not None else None
        missing_forcing = "none\n(no external\nforcing)" if name == "none" else None
        paste_cell(canvas, draw, forcing_img, x_positions[0], y, size, (45, 45, 45), width=4, missing=missing_forcing)

        if name == "none":
            t0 = load_crop(TILES / "none_original_t0.png", size)
            t20 = load_crop(TILES / "none_original_t20.png", size)
            paste_cell(
                canvas,
                draw,
                t0,
                x_positions[1],
                y,
                size,
                (232, 98, 0),
                width=5,
                caption="random-field crop\n(no attack avg local)",
            )
            paste_cell(
                canvas,
                draw,
                t20,
                x_positions[2],
                y,
                size,
                (110, 110, 110),
                width=3,
                caption="random-field crop\n(no attack avg local)",
            )
            continue

        t0 = load_crop(TILES / f"{name}_delta_t0.png", size)
        t20 = load_crop(TILES / f"{name}_delta_t20.png", size)
        missing = "not in local\nnotebook images" if full_status else None
        paste_cell(canvas, draw, t0, x_positions[1], y, size, (232, 98, 0), width=5, missing=missing)
        paste_cell(canvas, draw, t20, x_positions[2], y, size, (110, 110, 110), width=3, missing=missing)

    if full_status:
        note = (
            "Orange boxes mark the first-frame mean perturbation/diff. "
            "Blank cells are missing local rendered notebook images; logs point to remote /blue/... attack tensors."
        )
    else:
        note = (
            "Orange boxes mark the first-frame mean perturbation/diff. "
            "The none row uses only local random-field crops because its local attack-average image is missing."
        )
    draw_centered(draw, (30, height - 58, width - 30, height - 15), note, FONT_NOTE, fill=(80, 80, 80))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    build(["ringsCos", "isoCircles", "petals", "none"], AVAILABLE_OUT, full_status=False)
    build(
        [
            "45 deg default",
            "ringsCos",
            "sBands",
            "isoCircles",
            "petals",
            "ringsL1",
            "ringsLinf",
            "none",
        ],
        FULL_STATUS_OUT,
        full_status=True,
    )
    print(AVAILABLE_OUT)
    print(FULL_STATUS_OUT)


if __name__ == "__main__":
    main()
