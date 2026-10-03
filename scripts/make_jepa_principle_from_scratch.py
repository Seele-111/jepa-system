from __future__ import annotations

from pathlib import Path
import math
import shutil

from PIL import Image, ImageDraw, ImageFont


FIG_DIR = (
    Path("C:/Users/admin/Desktop")
    / "\u8001\u5e08\u6539\u7684\u8bba\u6587"
    / "\u4fee\u65392"
    / "figures"
    / "ai_generated"
)
OUT_MAIN = FIG_DIR / "jepa_principle_2.png"
OUT_ALIAS = FIG_DIR / "jepa_principle.png"
OUT_VERSIONED = FIG_DIR / "jepa_principle_from_scratch.png"
OUT_PREVIEW = FIG_DIR / "jepa_principle_2_print_preview.png"


def backup(path: Path) -> None:
    if path.exists():
        dst = path.with_name(path.stem + "_before_cvf_style" + path.suffix)
        if not dst.exists():
            shutil.copy2(path, dst)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(23, True)
F_BOX = font(18, True)
F_LABEL = font(17, True)
F_TEXT = font(15, False)
F_SMALL = font(13, False)
F_FORMULA = font(16, True)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = "#111827",
    leading: int = 3,
) -> None:
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + leading * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, fnt)
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=fnt, fill=fill)
        y += h + leading


def round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    radius: int = 10,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#111827",
    width: int = 3,
    head: int = 10,
) -> None:
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    a = math.atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * math.cos(a - math.pi / 6), y2 - head * math.sin(a - math.pi / 6)),
        (x2 - head * math.cos(a + math.pi / 6), y2 - head * math.sin(a + math.pi / 6)),
    ]
    draw.polygon(pts, fill=color)


def elbow_arrow(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[int, int]],
    color: str = "#111827",
    width: int = 3,
    head: int = 10,
) -> None:
    for a, b in zip(pts[:-2], pts[1:-1]):
        draw.line([a, b], fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], color=color, width=width, head=head)


def dashed(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#D33F49",
    width: int = 2,
    dash: int = 8,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    t = 0.0
    while t < length:
        e = min(t + dash, length)
        draw.line([(x1 + dx * t, y1 + dy * t), (x1 + dx * e, y1 + dy * e)], fill=color, width=width)
        t += dash * 2


def dashed_elbow(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: str = "#D33F49") -> None:
    for a, b in zip(pts[:-1], pts[1:]):
        dashed(draw, a, b, color=color)


def label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    color: str = "#111827",
    fnt: ImageFont.ImageFont = F_TEXT,
    bg: str | None = None,
) -> None:
    if bg:
        w, h = text_size(draw, text, fnt)
        draw.rounded_rectangle((x - 5, y - 3, x + w + 5, y + h + 5), radius=5, fill=bg)
    draw.text((x, y), text, font=fnt, fill=color)


def draw_video(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    round_rect(draw, (x, y, x + 138, y + 82), "#FFFFFF", "#94A3B8", radius=7, width=2)
    for i in range(6):
        draw.rectangle((x + 10 + i * 20, y + 9, x + 19 + i * 20, y + 15), fill="#111827")
        draw.rectangle((x + 10 + i * 20, y + 67, x + 19 + i * 20, y + 73), fill="#111827")
    for i in range(3):
        fx = x + 25 + i * 32
        draw.rectangle((fx, y + 30, fx + 38, y + 50), fill="#DBEAFE", outline="#60A5FA", width=1)
        draw.rectangle((fx, y + 50, fx + 38, y + 63), fill="#B7D7A8")
        draw.ellipse((fx + 17, y + 38, fx + 27, y + 48), fill="#EF4444")
    center_text(draw, (x - 5, y + 92, x + 143, y + 124), "video window x", F_TEXT)


def draw_grid(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 22, gap: int = 6) -> tuple[int, int, int, int]:
    target = {(1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (3, 4)}
    for r in range(5):
        for c in range(5):
            x0 = x + c * (size + gap)
            y0 = y + r * (size + gap)
            if (r, c) in target:
                fill, outline = "#FFFFFF", "#94A3B8"
            else:
                fill, outline = "#4F7FD2", "#315CA8"
            draw.rounded_rectangle((x0, y0, x0 + size, y0 + size), radius=4, fill=fill, outline=outline, width=1)
    bx = x + 3 * (size + gap) - 4
    by = y + 1 * (size + gap) - 4
    bw = 2 * size + gap + 8
    bh = 3 * size + 2 * gap + 8
    draw.rounded_rectangle((bx, by, bx + bw, by + bh), radius=5, outline="#D33F49", width=3)
    return bx, by, bx + bw, by + bh


def draw_bars(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    color: str,
    outline: str,
    height: int = 48,
    n: int = 5,
    step: int = 12,
) -> tuple[int, int, int, int]:
    for i in range(n):
        draw.rounded_rectangle((x + i * step, y, x + i * step + 7, y + height), radius=3, fill=color, outline=outline, width=1)
    return x, y, x + (n - 1) * step + 7, y + height


def draw_curve(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    draw.line((x, y + h, x + w, y + h), fill="#111827", width=2)
    draw.line((x, y, x, y + h), fill="#111827", width=2)
    vals = [0.05, 0.10, 0.18, 0.30, 0.54, 0.78, 0.52, 0.74, 0.92, 0.60, 0.36, 0.22]
    pts = [(x + i * w / (len(vals) - 1), y + h - v * h) for i, v in enumerate(vals)]
    draw.line(pts, fill="#2563EB", width=3)
    draw.line(pts[4:9], fill="#D33F49", width=4)
    label(draw, x + 26, y - 24, "surprise score over time", fnt=F_SMALL)
    label(draw, x + w - 8, y + h + 3, "t", fnt=F_SMALL)


def draw_stop_grad(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rounded_rectangle((x, y + 11, x + 22, y + 31), radius=4, fill="#FFFFFF", outline="#111827", width=2)
    draw.arc((x + 5, y, x + 17, y + 22), 180, 360, fill="#111827", width=2)
    draw.line((x + 5, y + 11, x + 5, y + 16), fill="#111827", width=2)
    draw.line((x + 17, y + 11, x + 17, y + 16), fill="#111827", width=2)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in (OUT_MAIN, OUT_ALIAS, OUT_VERSIONED, OUT_PREVIEW):
        backup(p)

    W, H = 1120, 610
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Light column guides, not large AI-style cards.
    draw.line((245, 70, 245, 530), fill="#E5E7EB", width=1)
    draw.line((790, 70, 790, 530), fill="#E5E7EB", width=1)
    label(draw, 42, 52, "Input", fnt=F_TITLE)
    label(draw, 290, 52, "Latent masked prediction", fnt=F_TITLE)
    label(draw, 824, 52, "Predictive mismatch", fnt=F_TITLE)

    # Input video and bifurcation to two encoders.
    draw_video(draw, 58, 238)
    arrow(draw, (196, 270), (284, 178))
    arrow(draw, (196, 306), (284, 404))

    round_rect(draw, (278, 132, 426, 216), "#EAF1FF", "#315CA8", radius=8, width=2)
    center_text(draw, (278, 132, 426, 216), "Context encoder\nE_c", F_BOX)
    round_rect(draw, (278, 360, 426, 444), "#EEF2F7", "#64748B", radius=8, width=2)
    center_text(draw, (278, 360, 426, 444), "Target encoder\nE_t", F_BOX)

    # Latent tokens and mask positions.
    label(draw, 462, 108, "latent tokens", fnt=F_LABEL)
    mask_box = draw_grid(draw, 450, 140)
    bx0, by0, bx1, by1 = mask_box
    label(draw, 494, 310, "target mask m", color="#D33F49", fnt=F_TEXT)
    dashed(draw, (530, 306), ((bx0 + bx1) // 2, by1 + 6))
    arrow(draw, (426, 174), (450, 202))

    # Context goes to predictor, mask is passed as indices.
    arrow(draw, (596, 200), (824, 168))
    label(draw, 626, 158, "context z_c", color="#1E3A8A", fnt=F_SMALL, bg="#FFFFFF")
    dashed_elbow(draw, [((bx0 + bx1) // 2, by0), (684, 118), (824, 150)])
    label(draw, 692, 104, "mask indices m", color="#D33F49", fnt=F_SMALL, bg="#FFFFFF")

    round_rect(draw, (824, 118, 990, 204), "#F1ECFF", "#6D54C3", radius=8, width=2)
    center_text(draw, (824, 118, 990, 204), "Predictor\nP(z_c, m)", F_BOX)

    # Observed target branch uses same target positions with stop-gradient.
    draw_stop_grad(draw, 444, 386)
    label(draw, 436, 426, "sg", fnt=F_SMALL)
    arrow(draw, (426, 402), (548, 402))
    obs = draw_bars(draw, 548, 376, "#AAB7C8", "#64748B", height=52)
    label(draw, 510, 342, "observed target z_m", fnt=F_TEXT)
    label(draw, 523, 363, "at mask m", fnt=F_SMALL, color="#475569")

    # Predicted target representation.
    arrow(draw, (907, 204), (907, 252))
    pred = draw_bars(draw, 876, 252, "#A78BFA", "#6D54C3", height=52)
    label(draw, 850, 226, "predicted target z_hat_m", fnt=F_TEXT)

    # Residual and score.
    round_rect(draw, (840, 346, 1012, 440), "#FEE2E2", "#D33F49", radius=9, width=2)
    center_text(draw, (840, 352, 1012, 384), "latent residual", F_BOX, fill="#111827")
    center_text(draw, (840, 386, 1012, 432), "s_t = ||z_hat_m\n- sg(z_m)||_2", F_FORMULA, fill="#111827", leading=1)
    arrow(draw, ((pred[0] + pred[2]) // 2, pred[3]), (900, 346))
    elbow_arrow(draw, [(obs[2], (obs[1] + obs[3]) // 2), (760, 402), (840, 404)])
    arrow(draw, (926, 440), (926, 470), color="#D33F49")
    draw_curve(draw, 858, 496, 152, 62)

    # Minimal implementation note: small and outside the semantic core.
    draw.line((312, 560, 668, 560), fill="#CBD5E1", width=1)
    draw.ellipse((334, 578, 344, 588), fill="#D33F49")
    label(draw, 354, 573, "one target-token mask per sliding window", color="#475569", fnt=F_SMALL)

    img.save(OUT_MAIN)
    img.save(OUT_ALIAS)
    img.save(OUT_VERSIONED)
    preview_w = 760
    preview_h = round(H * preview_w / W)
    img.resize((preview_w, preview_h), Image.Resampling.LANCZOS).save(OUT_PREVIEW)
    print(f"saved {OUT_MAIN}")
    print(f"saved {OUT_ALIAS}")
    print(f"saved {OUT_VERSIONED}")
    print(f"saved {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
