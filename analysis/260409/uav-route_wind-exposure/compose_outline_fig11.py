#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compose outline Fig. 11 from the 2D plan and 3D perspective route maps.

Panels keep their source pixels. The shorter panel is scaled up with Lanczos
so both rows share one height; neither image is downscaled. Output is PNG.

用法:
  python analysis/260409/uav-route_wind-exposure/compose_outline_fig11.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = REPO_ROOT / "results" / "uav_route_wind_shear" / "route_overview_map.png"
DEFAULT_PERSPECTIVE = (
    REPO_ROOT / "results" / "uav_route_wind_shear" / "route_overview_map_3d_z120.png"
)
DEFAULT_OUT = REPO_ROOT / "docs" / "scs-wrf-of-manuscript" / "images" / "outline-fig11.png"

PANEL_TITLES = ("a) 2D Plan View", "b) 3D Perspective View")
FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
)


def _dejavu_serif_bold() -> Path | None:
    """Matplotlib ships DejaVu Serif; use it when the Linux font paths are absent."""
    try:
        import matplotlib
    except ImportError:
        return None
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif-Bold.ttf"
    return path if path.is_file() else None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = list(FONT_CANDIDATES)
    bundled = _dejavu_serif_bold()
    if bundled is not None:
        candidates.append(bundled)
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(
        "DejaVu Serif Bold was not found. On Linux install fonts-dejavu-core; "
        "locally it is shipped with matplotlib as DejaVuSerif-Bold.ttf."
    )


def _match_height(image: Image.Image, height: int) -> Image.Image:
    """Return ``image`` at ``height`` without downscaling."""
    if image.height == height:
        return image
    if image.height > height:
        raise ValueError(f"refusing to downscale {image.height}px to {height}px")
    width = int(round(image.width * (height / image.height)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _axes_box(plan: Image.Image) -> tuple[int, int, int, int]:
    """Left, top, right, bottom of the 2D axes frame, in plan pixels."""
    import numpy as np

    gray = np.asarray(plan.convert("L"))
    dark_rows = (gray < 40).sum(axis=1)
    spine_rows = np.where(dark_rows > gray.shape[1] * 0.45)[0]
    if spine_rows.size < 2:
        raise RuntimeError("could not find the 2D axes frame")
    top = int(spine_rows[0])
    bottom = int(spine_rows[-1])
    band = gray[top]
    dark_cols = np.where(band < 40)[0]
    return int(dark_cols[0]), top, int(dark_cols[-1]), bottom


def compose(
    plan_path: Path,
    perspective_path: Path,
    out_path: Path,
    *,
    gap_px: int | None = None,
    margin_px: int | None = None,
) -> Path:
    plan = Image.open(plan_path).convert("RGBA")
    perspective = Image.open(perspective_path).convert("RGBA")
    ax_left, ax_top, _ax_right, ax_bottom = _axes_box(plan)
    axes_h = ax_bottom - ax_top
    # Match the 3D frame to the 2D axes box so the two panels share top and bottom.
    perspective = _match_height(perspective, axes_h)

    gap = gap_px if gap_px is not None else max(48, axes_h // 28)
    margin = margin_px if margin_px is not None else max(72, axes_h // 18)
    font_size = max(48, axes_h // 26)
    font = _font(font_size)
    title_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_heights = []
    for title in PANEL_TITLES:
        box = title_probe.textbbox((0, 0), title, font=font)
        title_heights.append(box[3] - box[1])
    title_h = max(title_heights)
    title_gap = max(16, font_size // 3)
    plan_x = margin
    plan_y = margin + title_h + title_gap
    # 3D top/bottom line up with the 2D axes frame, not the PNG padding.
    persp_x = plan_x + plan.width + gap
    persp_y = plan_y + ax_top
    width = persp_x + perspective.width + margin
    height = max(plan_y + plan.height, persp_y + perspective.height) + margin
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Panel a title starts at the axes left spine, as in the hand layout.
    draw.text((plan_x + ax_left, plan_y - title_gap - title_h), PANEL_TITLES[0], fill=(0, 0, 0, 255), font=font)
    draw.text((persp_x, plan_y - title_gap - title_h), PANEL_TITLES[1], fill=(0, 0, 0, 255), font=font)
    canvas.alpha_composite(plan, (plan_x, plan_y))
    canvas.alpha_composite(perspective, (persp_x, persp_y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, format="PNG", compress_level=3)
    print(f"[Fig11] {out_path}  {canvas.size[0]}×{canvas.size[1]}", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="2D route overview PNG")
    p.add_argument("--perspective", type=Path, default=DEFAULT_PERSPECTIVE, help="3D route overview PNG")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Composed PNG")
    p.add_argument("--gap", type=int, default=None, help="Pixels between panels")
    p.add_argument("--margin", type=int, default=None, help="Outer margin in pixels")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    compose(
        Path(args.plan),
        Path(args.perspective),
        Path(args.out),
        gap_px=args.gap,
        margin_px=args.margin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
