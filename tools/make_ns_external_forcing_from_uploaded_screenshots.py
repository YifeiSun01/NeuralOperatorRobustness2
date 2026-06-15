#!/usr/bin/env python3
"""Crop the user-uploaded NS forcing screenshots and build summary figures."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import matplotlib
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for colormaps") from exc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "ns_external_forcing_summary_20260615"
CHAT = OUT / "chat_uploaded_images"
CROPS = OUT / "uploaded_screenshot_crops"

EIGHT_ROW_OUT = OUT / "ns_external_forcing_attack_perturbation_uploaded_8row.png"
AVAILABLE_OUT = OUT / "ns_external_forcing_attack_perturbation_uploaded_available_rows.png"

SESSION = Path("/root/.codex/sessions/2026/06/15/rollout-2026-06-15T12-17-17-019ecb36-f7f7-76e3-ac1d-6f9e68bb15f6.jsonl")

UPLOAD_LABELS = [
    "forcing_grid",
    "petals_avg",
    "sBands_avg",
    "isoCircles_avg",
    "ringsCos_avg_a",
    "ringsCos_avg_b",
    "none_avg",
    "diagonalinit0_avg",
    "field_f_45deg",
]

ROW_TO_UPLOAD = {
    "45 deg default": "07_diagonalinit0_avg.png",
    "ringsCos": "04_ringsCos_avg_a.png",
    "sBands": "02_sBands_avg.png",
    "isoCircles": "03_isoCircles_avg.png",
    "petals": "01_petals_avg.png",
    "none": "06_none_avg.png",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation2")):
        path = base / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(31, True)
FONT_SUBTITLE = font(20)
FONT_HEADER = font(17, True)
FONT_LABEL = font(20)
FONT_SMALL = font(14)
FONT_NOTE = font(16)


def extract_uploaded_images() -> None:
    CHAT.mkdir(parents=True, exist_ok=True)
    if all((CHAT / f"{i:02d}_{label}.png").exists() for i, label in enumerate(UPLOAD_LABELS)):
        return
    for line in SESSION.open():
        obj = json.loads(line)
        payload = obj.get("payload", {})
        if payload.get("type") != "message" or payload.get("role") != "user":
            continue
        imgs = [it for it in payload.get("content", []) if it.get("type") == "input_image"]
        if len(imgs) < len(UPLOAD_LABELS):
            continue
        for i, label in enumerate(UPLOAD_LABELS):
            url = imgs[i]["image_url"]
            data = base64.b64decode(url.split(",", 1)[1])
            (CHAT / f"{i:02d}_{label}.png").write_bytes(data)
        return
    raise SystemExit("Could not find the uploaded screenshot images in the Codex session log.")


def _segments(vals: np.ndarray, threshold: int, min_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, value in enumerate(vals):
        if value > threshold and not in_seg:
            start = i
            in_seg = True
        elif (value <= threshold or i == len(vals) - 1) and in_seg:
            end = i if value <= threshold else i + 1
            if end - start >= min_len:
                out.append((start, end))
            in_seg = False
    return out


def detect_tile_grid(image: Image.Image) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    arr = np.asarray(image.convert("RGB"))
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    mask = ((mx - mn) > 20) & (mx < 250)
    xs = _segments(mask.sum(axis=0), threshold=30, min_len=30)
    ys = _segments(mask.sum(axis=1), threshold=30, min_len=30)
    if len(xs) < 15 or len(ys) < 5:
        raise ValueError(f"Could not detect a 15 x 5 tile grid, got {len(xs)} x {len(ys)}.")
    return xs[:15], ys[:5]


def crop_uploaded_tiles() -> None:
    CROPS.mkdir(parents=True, exist_ok=True)
    # Column groups in uploaded screenshots: pert[0:5], orig[5:10], diff[10:15].
    jobs = {
        "pert_t0": (0, 0),
        "orig_t0": (5, 0),
        "diff_t0": (10, 0),
        # The last row contains t=20, which is sometimes constant/blank in the
        # uploaded grids. Use the penultimate meaningful frame, t=19.
        "pert_t19": (4, 3),
        "orig_t19": (9, 3),
        "diff_t19": (14, 3),
    }
    for row, filename in ROW_TO_UPLOAD.items():
        image = Image.open(CHAT / filename).convert("RGB")
        xs, ys = detect_tile_grid(image)
        for name, (col, row_index) in jobs.items():
            x0, x1 = xs[col]
            y0, y1 = ys[row_index]
            pad = 1
            crop = image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))
            crop.save(CROPS / f"{row}_{name}.png")


def crop_external_forcing_tiles() -> None:
    forcing_grid = Image.open(CHAT / "00_forcing_grid.png").convert("RGB")
    grid_xs = [(57, 403), (487, 834), (917, 1264)]
    grid_ys = [(30, 375), (444, 791)]
    mapping = {
        "ringsCos": (0, 0),
        "sBands": (1, 0),
        "isoCircles": (2, 0),
        "petals": (0, 1),
        "ringsL1": (1, 1),
        "ringsLinf": (2, 1),
    }
    for name, (col, row) in mapping.items():
        x0, x1 = grid_xs[col]
        y0, y1 = grid_ys[row]
        forcing_grid.crop((x0, y0, x1, y1)).save(CROPS / f"{name}_external_forcing.png")

    # Main heatmap area of the uploaded 45-degree Field f figure.
    field_f = Image.open(CHAT / "08_field_f_45deg.png").convert("RGB")
    # The standalone Field f image uses the opposite vertical convention.
    ImageOps.flip(field_f.crop((20, 65, 792, 837))).save(CROPS / "45 deg default_external_forcing.png")


def load_tile(row: str, name: str, size: int) -> Image.Image | None:
    path = CROPS / f"{row}_{name}.png"
    if not path.exists():
        return None
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def load_external_forcing(row: str, size: int) -> Image.Image | None:
    if row == "none":
        return None
    path = CROPS / f"{row}_external_forcing.png"
    if not path.exists():
        return None
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont, fill=(30, 30, 30)) -> None:
    lines = text.split("\n")
    bboxes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    heights = [b[3] - b[1] for b in bboxes]
    widths = [b[2] - b[0] for b in bboxes]
    y = box[1] + (box[3] - box[1] - sum(heights) - 4 * (len(lines) - 1)) // 2
    for line, width, height in zip(lines, widths, heights):
        x = box[0] + (box[2] - box[0] - width) // 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += height + 4


def paste_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image | None,
    x: int,
    y: int,
    size: int,
    border=(110, 110, 110),
    width: int = 2,
    missing: str | None = None,
) -> None:
    if image is None:
        draw.rectangle([x, y, x + size, y + size], fill=(248, 248, 248), outline=(232, 232, 232), width=1)
        if missing:
            draw_centered(draw, (x, y, x + size, y + size), missing, FONT_SMALL, fill=(130, 130, 130))
        return
    canvas.paste(image, (x, y))
    draw.rectangle([x, y, x + size, y + size], outline=border, width=width)


def build(rows: list[str], output: Path, note: str) -> None:
    tile = 118
    left = 160
    top = 190
    col_gap = 22
    row_gap = 36
    columns = ["External\nforcing", "orig\nt=0", "pert\nt=0", "delta\nt=0", "orig\nt=19", "pert\nt=19", "delta\nt=19"]
    width = left + len(columns) * tile + (len(columns) - 1) * col_gap + 40
    height = top + len(rows) * (tile + row_gap) + 78
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw_centered(draw, (0, 18, width, 62), "2D NS: mean attack perturbations echo forcing", FONT_TITLE, fill=(10, 10, 10))
    draw_centered(
        draw,
        (40, 70, width - 40, 126),
        "Random-field initial conditions are attacked; averaging by forcing tag/time makes the perturbation pattern visible.",
        FONT_SUBTITLE,
        fill=(45, 45, 45),
    )

    xs = [left + i * (tile + col_gap) for i in range(len(columns))]
    for x, header in zip(xs, columns):
        draw_centered(draw, (x - 5, top - 58, x + tile + 5, top - 8), header, FONT_HEADER, fill=(20, 20, 20))

    tile_names = ["orig_t0", "pert_t0", "diff_t0", "orig_t19", "pert_t19", "diff_t19"]
    for r, row_name in enumerate(rows):
        y = top + r * (tile + row_gap)
        draw_centered(draw, (8, y, left - 14, y + tile), row_name, FONT_LABEL, fill=(18, 18, 18))
        forcing = load_external_forcing(row_name, tile)
        force_border = (232, 98, 0) if row_name == "45 deg default" else (55, 55, 55)
        force_width = 5 if row_name == "45 deg default" else 3
        paste_cell(canvas, draw, forcing, xs[0], y, tile, border=force_border, width=force_width, missing="none\n(no external\nforcing)")
        for c, tile_name in enumerate(tile_names, start=1):
            image = load_tile(row_name, tile_name, tile)
            is_first_delta = tile_name == "diff_t0"
            paste_cell(
                canvas,
                draw,
                image,
                xs[c],
                y,
                tile,
                border=(232, 98, 0) if is_first_delta else (110, 110, 110),
                width=5 if is_first_delta else 2,
                missing="not in\nuploaded\nscreenshots",
            )

    draw_centered(draw, (35, height - 58, width - 35, height - 12), note, FONT_NOTE, fill=(80, 80, 80))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    extract_uploaded_images()
    crop_uploaded_tiles()
    crop_external_forcing_tiles()
    build(
        ["45 deg default", "ringsCos", "sBands", "isoCircles", "petals", "ringsL1", "ringsLinf", "none"],
        EIGHT_ROW_OUT,
        "Orange marks: 45-degree forcing and first-frame delta. ringsL1/ringsLinf avg screenshots were not in the upload.",
    )
    build(
        ["45 deg default", "ringsCos", "sBands", "isoCircles", "petals", "none"],
        AVAILABLE_OUT,
        "Only rows present in the uploaded screenshots are shown here; orange boxes mark the emphasized first-frame delta.",
    )
    print(EIGHT_ROW_OUT)
    print(AVAILABLE_OUT)


if __name__ == "__main__":
    main()
