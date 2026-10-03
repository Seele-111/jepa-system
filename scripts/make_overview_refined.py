from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


FIG_DIR = Path(r"C:\Users\admin\Desktop\老师改的论文\修改2\figures")
SRC = FIG_DIR / "overview.png"
BACKUP = FIG_DIR / "overview_teacher_backup.png"
OUT = FIG_DIR / "overview.png"
VERSIONED = FIG_DIR / "overview_refined.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(24, True)
F_LABEL = font(16, True)
F_SMALL = font(12, False)
F_SMALL_B = font(12, True)
F_TINY = font(11, False)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill="#111111", leading=2):
    x, y, w, h = xy
    lines = text.split("\n")
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + leading * (len(lines) - 1)
    cy = y + (h - total_h) / 2
    for line, lh in zip(lines, heights):
        lw, _ = text_size(draw, line, fnt)
        draw.text((x + (w - lw) / 2, cy), line, font=fnt, fill=fill)
        cy += lh + leading


def round_rect(draw, box, radius=16, fill="#FFFFFF", outline="#999999", width=2, shadow=True):
    if shadow:
        sx0, sy0, sx1, sy1 = box
        draw.rounded_rectangle((sx0 + 3, sy0 + 3, sx1 + 3, sy1 + 3), radius=radius, fill="#DDE9D9")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill="#1F2937", width=3, head=11):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    ang = math.atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - math.pi / 6), y2 - head * math.sin(ang - math.pi / 6)),
        (x2 - head * math.cos(ang + math.pi / 6), y2 - head * math.sin(ang + math.pi / 6)),
    ]
    draw.polygon(pts, fill=fill)


def draw_curve(draw, origin, size, color="#2F5E9E", width=3, threshold=None):
    x, y = origin
    w, h = size
    pts = []
    values = [0.10, 0.20, 0.27, 0.32, 0.50, 0.46, 0.62, 0.86, 0.72, 0.58, 0.42, 0.30, 0.22, 0.16]
    for i, v in enumerate(values):
        px = x + i * w / (len(values) - 1)
        py = y + h - v * h
        pts.append((px, py))
    if threshold is not None:
        ty = y + h - threshold * h
        draw.line([(x, ty), (x + w, ty)], fill="#D7191C", width=2)
    draw.line(pts, fill=color, width=width, joint="curve")


def accent(draw, box, color):
    x0, y0, x1, _ = box
    draw.rounded_rectangle((x0 + 12, y0 + 8, x1 - 12, y0 + 13), radius=3, fill=color)


def draw_timeline(draw, x, y, w, segments, base="#1F2937", label=None):
    draw.line([(x, y), (x + w, y)], fill=base, width=2)
    for a, b, color, alpha_like in segments:
        xa = x + a * w
        xb = x + b * w
        draw.rounded_rectangle((xa, y - 6, xb, y + 6), radius=4, fill=color)
        if alpha_like:
            draw.rectangle((xa, y - 2, xb, y + 2), fill=color)
    if label:
        draw.text((x, y + 12), label, font=F_TINY, fill="#374151")


def draw_graph(draw, cx, cy):
    nodes = {
        "wide": (cx - 43, cy - 18),
        "child1": (cx - 10, cy - 42),
        "child2": (cx + 36, cy - 12),
        "gap": (cx + 8, cy + 34),
        "rej": (cx - 38, cy + 30),
    }
    edges = [
        ("wide", "child1", "#6B7280", False),
        ("wide", "child2", "#6B7280", False),
        ("child1", "child2", "#9CA3AF", True),
        ("wide", "gap", "#6B7280", False),
        ("rej", "gap", "#9CA3AF", True),
        ("gap", "child2", "#6B7280", False),
    ]
    for a, b, col, dash in edges:
        p, q = nodes[a], nodes[b]
        if dash:
            dash_line(draw, p, q, col, width=2)
        else:
            draw.line([p, q], fill=col, width=2)
    for name, p in nodes.items():
        fill = "#9CC6F2"
        outline = "#2F5E9E"
        if name == "gap":
            fill = "#FDE68A"
            outline = "#B45309"
        if name == "rej":
            fill = "#F3F4F6"
            outline = "#6B7280"
        draw.ellipse((p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9), fill=fill, outline=outline, width=2)
    draw.text((cx - 46, cy + 54), "IoU + sim + gap", font=F_TINY, fill="#374151")


def dash_line(draw, p, q, fill, width=2, dash=6):
    x1, y1 = p
    x2, y2 = q
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    d = 0
    while d < length:
        a = d
        b = min(d + dash, length)
        draw.line([(x1 + dx * a, y1 + dy * a), (x1 + dx * b, y1 + dy * b)], fill=fill, width=width)
        d += dash * 2


def draw_module(draw):
    # Replace only the teacher's green detection module.
    gx0, gy0, gx1, gy1 = 800, 7, 1239, 523
    draw.rounded_rectangle((gx0, gy0, gx1, gy1), radius=36, fill="#EEF8EA", outline=None)
    draw.rounded_rectangle((gx0 + 4, gy0 + 4, gx1 - 4, gy1 - 4), radius=32, outline="#DCEFD6", width=1)
    tw, th = text_size(draw, "Framewise Artifact Detection", F_TITLE)
    draw.text((gx0 + (gx1 - gx0 - tw) / 2, 16), "Framewise Artifact Detection", font=F_TITLE, fill="#000000")

    # Incoming arrow from composite surprise.
    arrow(draw, (758, 282), (822, 282), width=4, head=12)

    # Module boxes.
    b1 = (824, 196, 932, 367)
    b2 = (956, 70, 1090, 208)
    b3 = (956, 276, 1090, 434)
    b4 = (1114, 142, 1222, 392)

    # Temporal locator wedge.
    wedge = [(b1[0], b1[1]), (b1[2], b1[1] + 38), (b1[2], b1[3] - 38), (b1[0], b1[3])]
    draw.polygon(wedge, fill="#8EA9DB", outline="#2F5597")
    draw.line([wedge[0], wedge[1], wedge[2], wedge[3], wedge[0]], fill="#2F5597", width=2)
    center_text(draw, (b1[0] + 8, b1[1] + 28, b1[2] - b1[0] - 16, 62), "Temporal\nLocator", F_LABEL)
    center_text(draw, (b1[0] + 8, b1[1] + 92, b1[2] - b1[0] - 16, 26), "(Dilated TCN)", F_SMALL)
    draw_curve(draw, (b1[0] + 22, b1[1] + 128), (64, 28), width=2)
    draw.text((b1[0] + 42, b1[1] + 153), "p_t", font=F_SMALL_B, fill="#111111")

    # Fusion box.
    round_rect(draw, b2, radius=15, fill="#FFFFFF", outline="#87B980", width=2)
    accent(draw, b2, "#7C5CC4")
    center_text(draw, (b2[0] + 8, b2[1] + 14, b2[2] - b2[0] - 16, 30), "Rank Fusion", F_LABEL)
    draw_curve(draw, (b2[0] + 18, b2[1] + 56), (40, 23), color="#2F5E9E", width=2)
    draw_curve(draw, (b2[0] + 18, b2[1] + 88), (40, 23), color="#0E9AC1", width=2)
    draw.text((b2[0] + 63, b2[1] + 58), "TCN prob.", font=F_SMALL_B, fill="#111111")
    draw.text((b2[0] + 63, b2[1] + 90), "JEPA rank", font=F_SMALL_B, fill="#111111")
    draw.line([(b2[0] + 20, b2[1] + 122), (b2[2] - 20, b2[1] + 122)], fill="#374151", width=2)
    draw.text((b2[0] + 54, b2[1] + 124), "p_final", font=F_TINY, fill="#374151")

    # Proposal box.
    round_rect(draw, b3, radius=15, fill="#FFFFFF", outline="#87B980", width=2)
    accent(draw, b3, "#F6A03A")
    center_text(draw, (b3[0] + 8, b3[1] + 16, b3[2] - b3[0] - 16, 36), "Proposal\nWindows", F_LABEL)
    draw_timeline(
        draw,
        b3[0] + 17,
        b3[1] + 80,
        b3[2] - b3[0] - 34,
        [(0.22, 0.74, "#F6A03A", True)],
        label="broad candidate",
    )
    draw_timeline(
        draw,
        b3[0] + 17,
        b3[1] + 128,
        b3[2] - b3[0] - 34,
        [(0.20, 0.38, "#E31A1C", False), (0.58, 0.74, "#E31A1C", False)],
        label="split candidates",
    )

    # Topology refinement box.
    round_rect(draw, b4, radius=15, fill="#FFFFFF", outline="#87B980", width=2)
    accent(draw, b4, "#2F5E9E")
    center_text(draw, (b4[0] + 8, b4[1] + 17, b4[2] - b4[0] - 16, 42), "Topology\nRefinement", F_LABEL)
    draw_graph(draw, b4[0] + 55, b4[1] + 117)
    draw_timeline(
        draw,
        b4[0] + 16,
        b4[3] - 34,
        b4[2] - b4[0] - 32,
        [(0.16, 0.36, "#E31A1C", False), (0.61, 0.80, "#E31A1C", False)],
    )
    draw.text((b4[0] + 17, b4[3] - 22), "split / trim / reject", font=F_TINY, fill="#374151")

    # Arrows inside the green block.
    arrow(draw, (932, 282), (956, 140), width=3, head=10)
    draw.text((934, 194), "p_t", font=F_TINY, fill="#374151")
    # JEPA rank comes directly from predictor-surprise evidence, not from TCN alone.
    dash_line(draw, (766, 252), (956, 126), "#0E9AC1", width=2, dash=7)
    arrow(draw, (946, 133), (956, 126), fill="#0E9AC1", width=2, head=8)
    draw.text((868, 170), "JEPA rank", font=F_TINY, fill="#0E758F")
    arrow(draw, (1023, 208), (1023, 276), width=3, head=10)
    draw.text((1030, 234), "threshold", font=F_TINY, fill="#374151")
    arrow(draw, (1090, 354), (1114, 280), width=3, head=10)
    draw.text((1083, 306), "proposals", font=F_TINY, fill="#374151")

    # Outgoing arrow to the final error probability panel.
    arrow(draw, (1222, 282), (1301, 282), width=4, head=12)


def main():
    if not BACKUP.exists():
        Image.open(SRC).save(BACKUP)
    img = Image.open(BACKUP).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw_module(draw)
    img.save(VERSIONED)
    img.save(OUT)
    print(f"saved {OUT}")
    print(f"versioned copy {VERSIONED}")
    print(f"backup {BACKUP}")


if __name__ == "__main__":
    main()
