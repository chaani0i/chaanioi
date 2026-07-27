#!/usr/bin/env python3
"""Generate assets/profile-particles.gif from profile.png (assemble → hold → scatter)."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "profile.png"
OUT = ROOT / "assets" / "profile-particles.gif"

SIZE = 280
PARTICLES = 1600
FRAMES_IN = 24
HOLD = 12
FRAMES_OUT = 14


def ease_out_cubic(t: np.ndarray | float):
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    return 3 * t * t - 2 * t * t * t


def main() -> None:
    np.random.seed(42)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(SRC).convert("RGBA")
    w, h = img.size
    side = min(w, int(h * 0.72))
    left = (w - side) // 2
    top = int(h * 0.02)
    face = img.crop((left, top, left + side, top + side)).resize(
        (SIZE, SIZE), Image.Resampling.LANCZOS
    )

    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).ellipse((5, 5, SIZE - 6, SIZE - 6), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    face_arr = np.array(face)

    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    cy = cx = SIZE / 2.0
    inside = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) < (SIZE / 2 - 6)
    lum = (
        face_arr[..., 0].astype(np.float32) * 0.299
        + face_arr[..., 1] * 0.587
        + face_arr[..., 2] * 0.114
    )
    weight = np.where(inside, (255 - lum) ** 1.35 + 8, 0).astype(np.float64)
    weight /= weight.sum()

    flat_idx = np.random.choice(SIZE * SIZE, size=PARTICLES, replace=False, p=weight.ravel())
    py = (flat_idx // SIZE).astype(np.float64)
    px = (flat_idx % SIZE).astype(np.float64)
    colors = face_arr[flat_idx // SIZE, flat_idx % SIZE]

    angles = np.random.uniform(0, 2 * math.pi, PARTICLES)
    radii = np.random.uniform(SIZE * 0.55, SIZE * 0.98, PARTICLES)
    far = np.random.rand(PARTICLES) < 0.35
    radii[far] = np.random.uniform(SIZE * 0.95, SIZE * 1.3, int(far.sum()))
    sx = cx + radii * np.cos(angles)
    sy = cy + radii * np.sin(angles)
    psizes = np.random.choice([1, 1, 2, 2, 2, 3], size=PARTICLES).astype(int)
    delays = np.random.uniform(0.0, 0.32, PARTICLES)
    accent_set = set(np.random.choice(PARTICLES, 60, replace=False).tolist())
    rng = (np.arange(PARTICLES) * 7919 % 1000) / 1000.0
    sin_i = np.sin(np.arange(PARTICLES))
    cos_i = np.cos(np.arange(PARTICLES))

    def render_frame(progress_in: float, scatter: float = 0.0) -> Image.Image:
        canvas = Image.new("RGBA", (SIZE, SIZE), (10, 18, 28, 255))
        ring = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        ImageDraw.Draw(ring).ellipse(
            (3, 3, SIZE - 4, SIZE - 4), outline=(0, 198, 255, 50), width=2
        )
        canvas = Image.alpha_composite(canvas, ring)
        draw = ImageDraw.Draw(canvas)

        t = np.clip((progress_in - delays) / (1 - float(delays.max()) + 1e-6), 0, 1)
        t = ease_out_cubic(t)

        if scatter > 0:
            dx = px - cx
            dy = py - cy
            mag = np.sqrt(dx * dx + dy * dy) + 1e-6
            explode_r = scatter * (45 + rng * 130)
            ex = px + (dx / mag) * explode_r + sin_i * scatter * 18
            ey = py + (dy / mag) * explode_r + cos_i * scatter * 18
            curx = px * (1 - scatter) + ex * scatter
            cury = py * (1 - scatter) + ey * scatter
            alpha_mul = np.full(PARTICLES, 1.0 - scatter * 0.85)
        else:
            curx = sx + (px - sx) * t
            cury = sy + (py - sy) * t
            alpha_mul = 0.35 + 0.65 * t

        for ii in np.argsort(psizes):
            ii = int(ii)
            x, y = float(curx[ii]), float(cury[ii])
            if x < -4 or y < -4 or x > SIZE + 4 or y > SIZE + 4:
                continue
            r, g, b = int(colors[ii, 0]), int(colors[ii, 1]), int(colors[ii, 2])
            a = int(min(255, float(colors[ii, 3]) * float(alpha_mul[ii])))
            if ii in accent_set and progress_in < 0.85 and scatter == 0:
                r, g, b = 0, 198, 255
                a = min(255, int(a * 0.9))
            s = int(psizes[ii])
            if scatter > 0.4:
                s = max(1, s - 1)
            draw.rectangle([x, y, x + s, y + s], fill=(r, g, b, a))

        if progress_in > 0.72 and scatter == 0:
            face_circ = face.copy()
            face_circ.putalpha(mask)
            fade = ease_in_out(min(1.0, (progress_in - 0.72) / 0.28))
            alpha = face_circ.split()[-1].point(lambda p, f=fade: int(p * f * 0.92))
            face_circ.putalpha(alpha)
            canvas = Image.alpha_composite(canvas, face_circ)

        if progress_in >= 0.98 and scatter < 0.05:
            portrait = face.copy()
            portrait.putalpha(mask)
            canvas = Image.alpha_composite(canvas, portrait)
            border = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
            bd = ImageDraw.Draw(border)
            bd.ellipse((2, 2, SIZE - 3, SIZE - 3), outline=(0, 198, 255, 220), width=3)
            bd.ellipse((7, 7, SIZE - 8, SIZE - 8), outline=(0, 198, 255, 70), width=1)
            canvas = Image.alpha_composite(canvas, border)

        return canvas.convert("RGB")

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        idx = 0
        for f in range(FRAMES_IN):
            render_frame(f / (FRAMES_IN - 1), 0).save(frames_dir / f"frame_{idx:03d}.png")
            idx += 1
        for _ in range(HOLD):
            render_frame(1.0, 0).save(frames_dir / f"frame_{idx:03d}.png")
            idx += 1
        for f in range(FRAMES_OUT):
            render_frame(1.0, ease_in_out(f / max(1, FRAMES_OUT - 1))).save(
                frames_dir / f"frame_{idx:03d}.png"
            )
            idx += 1
        for _ in range(2):
            render_frame(0.0, 0).save(frames_dir / f"frame_{idx:03d}.png")
            idx += 1

        palette = frames_dir / "palette.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "18",
                "-i",
                str(frames_dir / "frame_%03d.png"),
                "-vf",
                "palettegen=max_colors=128:stats_mode=diff",
                str(palette),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "18",
                "-i",
                str(frames_dir / "frame_%03d.png"),
                "-i",
                str(palette),
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                str(OUT),
            ],
            check=True,
            capture_output=True,
        )

    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
