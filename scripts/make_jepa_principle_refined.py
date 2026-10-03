from __future__ import annotations

from pathlib import Path
import math
import shutil

from PIL import Image, ImageDraw, ImageFont


FIG_DIR = Path(r"C:\Users\admin\Desktop\老师改的论文\修改2\figures\ai_generated")
OUT_MAIN = FIG_DIR / "jepa_principle_2.png"
OUT_ALIAS = FIG_DIR / "jepa_principle.png"
OUT_VERSIONED = FIG_DIR / "jepa_principle_refined.png"


def backup(path: Path) -> None:
    if path.exists():
        dst = path.with_name(path.stem + "_teacher_backup" + path.suffix)
        if not dst.exists():
            shutil.copy2(path, dst)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    choices = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for p in choices:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


F_HEADER = font(30, True)
F_LABEL = font(27, True)
F_MED = font(23, True)
F_SMALL = font(20, False)
F_SMALL_B = font(20, True)
F_TINY = font(17, False)
F_FORMULA = font(18, True)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill="#111827", leading=6):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    hs = [text_size(draw, ln, fnt)[1] for ln in lines]
    total = sum(hs) + leading * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for ln, h in zip(lines, hs):
        w, _ = text_size(draw, ln, fnt)
        draw.text((x0 + (x1 - x0 - w) / 2, y), ln, font=fnt, fill=fill)
        y += h + leading


def round_rect(draw: ImageDraw.ImageDraw, box, r=18, fill="#FFFFFF", outline="#CBD5E1", width=2, shadow=True):
    x0, y0, x1, y1 = box
    if shadow:
        draw.rounded_rectangle((x0 + 5, y0 + 6, x1 + 5, y1 + 6), radius=r, fill="#E5E7EB")
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start, end, color="#1F2937", width=5, head=18):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    ang = math.atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - math.pi / 6), y2 - head * math.sin(ang - math.pi / 6)),
        (x2 - head * math.cos(ang + math.pi / 6), y2 - head * math.sin(ang + math.pi / 6)),
    ]
    draw.polygon(pts, fill=color)


def dash_line(draw, p, q, color="#EF4444", width=3, dash=12):
    x1, y1 = p
    x2, y2 = q
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    d = 0
    while d < length:
        a = d
        b = min(d + dash, length)
        draw.line([(x1 + dx * a, y1 + dy * a), (x1 + dx * b, y1 + dy * b)], fill=color, width=width)
        d += dash * 2


def token_grid(draw, x, y, rows, cols, size=34, gap=9, mask=None, context="#5B83D7", target="#E5E7EB"):
    mask = set(mask or [])
    for r in range(rows):
        for c in range(cols):
            fill = target if (r, c) in mask else context
            outline = "#94A3B8" if (r, c) in mask else "#4267B2"
            draw.rounded_rectangle(
                (x + c * (size + gap), y + r * (size + gap), x + c * (size + gap) + size, y + r * (size + gap) + size),
                radius=4,
                fill=fill,
                outline=outline,
                width=2,
            )


def latent_bars(draw, x, y, n=5, h=74, color="#8B5CF6", outline="#5B3BA3"):
    for i in range(n):
        draw.rounded_rectangle((x + i * 18, y, x + i * 18 + 11, y + h), radius=3, fill=color, outline=outline, width=1)


def video_strip(draw, x, y, w=214, h=150):
    round_rect(draw, (x, y, x + w, y + h), r=16, fill="#FFFFFF", outline="#CBD5E1", width=2, shadow=True)
    for i in range(5):
        fx = x + 17 + i * 40
        fy = y + 32 + (i % 2) * 10
        draw.rounded_rectangle((fx, fy, fx + 52, fy + 76), radius=5, fill="#DDEBFF", outline="#93B4E8", width=2)
        draw.rectangle((fx, fy + 46, fx + 52, fy + 76), fill="#B7D7A8")
        draw.ellipse((fx + 26, fy + 22, fx + 40, fy + 36), fill="#EF4444")
    # Film perforations.
    for i in range(8):
        draw.rectangle((x + 14 + i * 26, y + 12, x + 26 + i * 26, y + 22), fill="#111827")
        draw.rectangle((x + 14 + i * 26, y + h - 22, x + 26 + i * 26, y + h - 12), fill="#111827")
    center_text(draw, (x, y + h + 10, x + w, y + h + 48), "generated video\nwindow", F_SMALL_B)


def lock_icon(draw, x, y):
    draw.rounded_rectangle((x, y + 14, x + 36, y + 44), radius=7, fill="#FFFFFF", outline="#111827", width=3)
    draw.arc((x + 8, y, x + 28, y + 28), 180, 360, fill="#111827", width=4)
    draw.line((x + 8, y + 14, x + 8, y + 20), fill="#111827", width=4)
    draw.line((x + 28, y + 14, x + 28, y + 20), fill="#111827", width=4)


def draw_curve(draw, x, y, w, h):
    draw.line((x, y + h, x + w, y + h), fill="#334155", width=3)
    draw.line((x, y, x, y + h), fill="#334155", width=3)
    vals = [0.12, 0.14, 0.18, 0.22, 0.31, 0.56, 0.46, 0.82, 0.92, 0.65, 0.49, 0.36, 0.25, 0.20]
    pts = []
    for i, v in enumerate(vals):
        pts.append((x + i * w / (len(vals) - 1), y + h - v * h))
    draw.line(pts, fill="#2563EB", width=5, joint="curve")
    # Highlight high-surprise region.
    draw.line(pts[5:10], fill="#EF4444", width=6, joint="curve")
    draw.text((x + w - 22, y + h + 8), "t", font=F_SMALL_B, fill="#111827")


def draw_mask_insets(draw):
    # Compact bottom policy strip, kept intentionally small for a single-column figure.
    round_rect(draw, (408, 616, 1128, 724), r=22, fill="#F8FAFC", outline="#D6DEE9", width=2, shadow=False)
    draw.text((440, 640), "mask policy used in this paper", font=F_SMALL_B, fill="#111827")
    draw.text((440, 675), "I-JEPA: spatial target", font=F_TINY, fill="#334155")
    token_grid(draw, 602, 664, 2, 4, size=17, gap=5, mask={(0, 1), (0, 2), (1, 1), (1, 2)})
    draw.text((738, 675), "V-JEPA: later temporal target", font=F_TINY, fill="#334155")
    token_grid(draw, 955, 664, 2, 4, size=17, gap=5, mask={(0, 2), (0, 3), (1, 2), (1, 3)})
    draw.text((440, 702), "single 50% target-token mask per prediction window", font=F_TINY, fill="#64748B")


def main():
    W, H = 1536, 768
    for p in (OUT_MAIN, OUT_ALIAS):
        backup(p)

    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Background groups.
    draw.rounded_rectangle((24, 62, 375, 590), radius=34, fill="#F8FAFC", outline="#E2E8F0", width=2)
    draw.rounded_rectangle((405, 62, 1062, 590), radius=34, fill="#F3F7FF", outline="#D7E4FF", width=2)
    draw.rounded_rectangle((1092, 62, 1510, 590), radius=34, fill="#FFF7ED", outline="#FED7AA", width=2)
    draw.text((55, 86), "input window", font=F_HEADER, fill="#111827")
    draw.text((438, 86), "masked latent prediction", font=F_HEADER, fill="#111827")
    draw.text((1132, 86), "predictor surprise", font=F_HEADER, fill="#111827")

    # Input video.
    video_strip(draw, 82, 218)
    arrow(draw, (303, 294), (426, 294), width=5, head=16)

    # Encoder.
    round_rect(draw, (426, 220, 568, 368), r=18, fill="#2F65C8", outline="#1E4FA6", width=2)
    center_text(draw, (426, 220, 568, 368), "Frozen\nJEPA\nencoder", F_LABEL, fill="#FFFFFF")
    arrow(draw, (568, 294), (642, 294), width=5, head=16)

    # Latent grid.
    draw.text((646, 170), "latent token map", font=F_SMALL_B, fill="#111827")
    mask_cells = {(1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (3, 4)}
    token_grid(draw, 648, 205, 5, 6, size=34, gap=8, mask=mask_cells)
    mx, my = 648 + 3 * 42 - 5, 205 + 1 * 42 - 5
    draw.rounded_rectangle((mx, my, mx + 2 * 42 + 4, my + 3 * 42 + 4), radius=8, outline="#EF4444", width=4)
    draw.text((672, 430), "target mask m", font=F_SMALL_B, fill="#EF4444")
    dash_line(draw, (752, 418), (mx + 38, my + 118), color="#EF4444", width=3, dash=10)

    # Context branch.
    arrow(draw, (868, 248), (940, 190), width=5, head=16)
    draw.text((918, 148), "visible context", font=F_SMALL_B, fill="#1E3A8A")
    draw.text((965, 174), "z_c", font=F_SMALL_B, fill="#1E3A8A")
    latent_bars(draw, 980, 142, n=5, h=80, color="#6B8AE8", outline="#3557B7")
    arrow(draw, (1078, 182), (1142, 182), width=5, head=16)

    # Predictor.
    round_rect(draw, (1142, 124, 1292, 240), r=18, fill="#7C5CC4", outline="#593AA0", width=2)
    center_text(draw, (1142, 124, 1292, 240), "Predictor\nP(z_c, m)", F_LABEL, fill="#FFFFFF")
    arrow(draw, (1292, 182), (1340, 182), width=5, head=16)
    draw.text((1336, 128), "predicted", font=F_SMALL_B, fill="#111827")
    draw.text((1336, 154), "z_hat_m", font=F_SMALL_B, fill="#111827")
    latent_bars(draw, 1346, 204, n=5, h=66, color="#A78BFA", outline="#6D54C3")

    # Mask positions to predictor.
    dash_line(draw, (750, 428), (750, 504), color="#EF4444", width=3, dash=10)
    dash_line(draw, (750, 504), (1118, 504), color="#EF4444", width=3, dash=10)
    dash_line(draw, (1118, 504), (1218, 240), color="#EF4444", width=3, dash=10)

    # Stop-gradient target branch.
    round_rect(draw, (464, 444, 614, 548), r=18, fill="#64748B", outline="#475569", width=2)
    center_text(draw, (464, 444, 614, 548), "Target\nencoder", F_LABEL, fill="#FFFFFF")
    lock_icon(draw, 626, 474)
    draw.text((617, 526), "stop-grad", font=F_TINY, fill="#111827")
    arrow(draw, (614, 496), (888, 496), width=5, head=16)
    draw.text((736, 452), "observed target", font=F_SMALL_B, fill="#111827")
    draw.text((902, 499), "z_m", font=F_SMALL_B, fill="#111827")
    latent_bars(draw, 914, 462, n=5, h=70, color="#94A3B8", outline="#64748B")

    # Mismatch and curve.
    round_rect(draw, (1120, 344, 1350, 470), r=22, fill="#EF4B37", outline="#C93628", width=2)
    center_text(draw, (1120, 350, 1350, 408), "Representation\nmismatch", F_LABEL, fill="#FFFFFF")
    center_text(draw, (1120, 410, 1350, 456), "s_t = || z_hat_m - sg(z_m) ||_2", F_FORMULA, fill="#FFFFFF")
    arrow(draw, (1392, 260), (1288, 344), width=5, head=16)
    arrow(draw, (1014, 500), (1120, 438), width=5, head=16)
    arrow(draw, (1350, 407), (1382, 407), color="#EF4444", width=5, head=16)

    draw.text((1382, 282), "surprise curve", font=F_TINY, fill="#111827")
    draw_curve(draw, 1380, 320, 104, 105)
    draw.text((1128, 492), "high residual: artifact cue", font=F_SMALL_B, fill="#EF4444")
    draw.text((1128, 522), "low residual: predictable", font=F_SMALL, fill="#475569")

    # Bottom mask insets.
    draw_mask_insets(draw)

    img.save(OUT_MAIN)
    img.save(OUT_ALIAS)
    img.save(OUT_VERSIONED)
    print(f"saved {OUT_MAIN}")
    print(f"saved {OUT_ALIAS}")
    print(f"saved {OUT_VERSIONED}")


if __name__ == "__main__":
    main()
