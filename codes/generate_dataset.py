# ============================================================
# 📦 Dataset Generator — LineCNN_OCR
# ============================================================
# Run modes:
#   python generate_dataset.py           → 60k training samples
#   python generate_dataset.py --test    → test_line.png (192×64)
#   python generate_dataset.py --para    → test_para.png + splitter check
# ============================================================

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
import os
import cv2

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
OUT_DIR     = "data_line"
NUM_SAMPLES = 60_000
LINE_W      = 240
LINE_H      = 64
MAX_CHARS   = 60

CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-()/@#%&"
)
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────
# FONTS
# ──────────────────────────────────────────
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]
AVAILABLE_FONTS = [p for p in FONT_PATHS if os.path.exists(p)]
if not AVAILABLE_FONTS:
    AVAILABLE_FONTS = [None]
print(f"✅ Found {len(AVAILABLE_FONTS)} font(s)")


def get_font(size=None):
    path = random.choice(AVAILABLE_FONTS)
    size = size or random.randint(14, 32)
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


# ──────────────────────────────────────────
# WORDS
# ──────────────────────────────────────────
with open("words_alpha.txt") as f:
    WORDS = [w.strip() for w in f if 2 <= len(w.strip()) <= 9]

PUNCT_END  = [".", ",", "!", "?", ";", ":"]
PUNCT_WRAP = [("(", ")"), ('"', '"')]


def random_word():
    w = random.choice(WORDS)
    if random.random() < 0.2:
        w = w.capitalize()
    if random.random() < 0.12:
        w += random.choice(PUNCT_END)
    if random.random() < 0.04:
        l, r = random.choice(PUNCT_WRAP)
        w = l + w + r
    return w


def random_number_token():
    return random.choice([
        str(random.randint(0, 9999)),
        f"{random.randint(1,99)}.{random.randint(0,99)}",
        f"#{random.randint(1,100)}",
        f"{random.randint(10,99)}%",
    ])


def generate_line_text():
    words, total = [], 0
    for _ in range(random.randint(4, 10)):
        w = random_number_token() if random.random() < 0.08 else random_word()
        w_clean = "".join(c for c in w if c in char_to_idx)
        if not w_clean:
            continue
        sep = " " if words else ""
        if total + len(sep) + len(w_clean) > MAX_CHARS:
            break
        words.append(w_clean)
        total += len(sep) + len(w_clean)
    if not words:
        words = [random.choice(WORDS)[:8]]
    return " ".join(words)


# ──────────────────────────────────────────
# RENDER single line → (LINE_H × LINE_W)
# ──────────────────────────────────────────
def render_line(text, font, line_w=LINE_W, line_h=LINE_H):
    RENDER_H = 80
    pad_x    = random.randint(6, 16)
    pad_y    = random.randint(4, 12)

    dummy = Image.new("L", (8192, RENDER_H), 255)
    draw  = ImageDraw.Draw(dummy)
    bb    = draw.textbbox((pad_x, pad_y), text, font=font)
    w     = max(bb[2] + pad_x, line_w)

    bg  = random.randint(220, 255)
    img = Image.new("L", (w, RENDER_H), bg)

    if random.random() < 0.2:
        d = ImageDraw.Draw(img)
        d.line([(0, RENDER_H//2), (w, RENDER_H//2)],
               fill=random.randint(200, 215))

    draw = ImageDraw.Draw(img)
    draw.text((pad_x, pad_y), text, font=font, fill=random.randint(0, 50))

    if random.random() < 0.25:
        img = img.rotate(random.uniform(-1.5, 1.5),
                         fillcolor=bg, expand=False)

    arr = np.array(img)
    return cv2.resize(arr, (line_w, line_h), interpolation=cv2.INTER_AREA)


def augment(arr):
    if random.random() < 0.5:
        arr = np.clip(arr.astype(np.float32)
                      * random.uniform(0.75, 1.25)
                      + random.randint(-20, 20), 0, 255)
    if random.random() < 0.3:
        pil = Image.fromarray(arr.astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(
              radius=random.uniform(0.3, 0.9)))
        arr = np.array(pil, dtype=np.float32)
    if random.random() < 0.2:
        mask      = np.random.rand(*arr.shape) < 0.008
        arr[mask] = np.random.randint(0, 256, mask.sum())
    return arr.astype(np.uint8)


# ──────────────────────────────────────────
# HORIZONTAL PROJECTION LINE SPLITTER
# ──────────────────────────────────────────
def split_lines_projection(gray_np, debug=False):
    """
    Split a paragraph image into line strips using projection profile.
    Expects landscape orientation (wider than tall).
    dil_h is fixed at 3px — only bridges intra-character gaps (dots on i/j).
    """
    H, W = gray_np.shape

    # binarise: ink = 255
    _, binary = cv2.threshold(
        gray_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # tiny dilation — fills dot-above-i gaps only, never inter-line gaps
    dil_h  = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dil_h))
    filled = cv2.dilate(binary, kernel)

    # row projection
    proj    = filled.sum(axis=1).astype(np.float32)
    thresh  = W * 0.005
    is_text = proj > thresh

    # find bands
    lines, inside, y0 = [], False, 0
    for y in range(H):
        if is_text[y] and not inside:
            inside, y0 = True, y
        elif not is_text[y] and inside:
            inside = False
            band_h = y - y0
            if band_h >= 4:
                pad = max(2, band_h // 8)
                lines.append([max(0, y0 - pad), min(H, y + pad)])

    if inside and (H - y0) >= 4:
        pad = max(2, (H - y0) // 8)
        lines.append([max(0, y0 - pad), H])

    # merge bands closer than 20% of median band height
    if lines:
        med_h   = sorted(b - a for a, b in lines)[len(lines) // 2]
        min_gap = max(2, int(med_h * 0.2))
        merged  = []
        for seg in lines:
            if merged and (seg[0] - merged[-1][1]) < min_gap:
                merged[-1][1] = seg[1]
            else:
                merged.append(list(seg))
        lines = merged

    if debug:
        print(f"  [splitter] image={W}×{H}  dil_h={dil_h}"
              f"  thresh={thresh:.1f}  found {len(lines)} line(s): {lines}")

    return [(a, b) for a, b in lines]


# ──────────────────────────────────────────
# GENERATE TRAINING DATA
# ──────────────────────────────────────────
def generate():
    for i in range(NUM_SAMPLES):
        font = get_font()
        text = generate_line_text()
        arr  = render_line(text, font)
        arr  = augment(arr)

        Image.fromarray(arr).save(f"{OUT_DIR}/img_{i:05d}.png")
        with open(f"{OUT_DIR}/img_{i:05d}.txt", "w") as f:
            f.write(text)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1:>6} / {NUM_SAMPLES}")

    print(f"✅ Dataset ready in '{OUT_DIR}/'  ({NUM_SAMPLES} samples)")


# ──────────────────────────────────────────
# TEST IMAGE HELPERS
# ──────────────────────────────────────────
def make_test_line(filename="test_line.png", font_size=24):
    """Single line at 192×64 — landscape."""
    font = get_font(size=font_size)
    text = generate_line_text()
    arr  = render_line(text, font)
    Image.fromarray(arr).save(filename)
    print(f"✅ {filename}  ({LINE_W}×{LINE_H}px)  label: '{text}'")


def make_test_para(lines, filename="test_para.png",
                   font_size=24, line_gap=20):
    """
    Multi-line paragraph image — rendered in landscape (lines are horizontal).
    The splitter operates on this directly — NO rotation needed.
    line_gap >= 15px ensures the splitter can detect boundaries.
    """
    font_obj = get_font(size=font_size)
    pad      = 16

    dummy = Image.new("L", (4096, 512), 255)
    draw  = ImageDraw.Draw(dummy)
    bbs      = [draw.textbbox((0, 0), l, font=font_obj) for l in lines]
    line_hs  = [bb[3] - bb[1] for bb in bbs]
    max_w    = max(bb[2] - bb[0] for bb in bbs)

    # landscape: width = text width + padding, height = stacked lines
    total_w = max_w + pad * 2
    total_h = pad * 2 + sum(line_hs) + line_gap * (len(lines) - 1)

    img  = Image.new("L", (total_w, total_h), 255)
    draw = ImageDraw.Draw(img)
    y    = pad

    for i, line in enumerate(lines):
        draw.text((pad, y), line, font=font_obj, fill=0)
        y += line_hs[i] + line_gap

    img.save(filename)
    w, h = img.size
    print(f"✅ {filename}  ({w}×{h}px)  — landscape, no rotation needed")
    print(f"   {len(lines)} lines, gap={line_gap}px")
    print(f"   Labels: {' / '.join(lines)}")

    # self-check splitter
    gray   = np.array(img)
    splits = split_lines_projection(gray, debug=True)
    status = "✅" if len(splits) == len(lines) else "⚠️ "
    print(f"   {status} Splitter: detected {len(splits)}, expected {len(lines)}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        make_test_line()
    elif "--para" in sys.argv:
        make_test_para([
            "Hello world if the saved image itself is not padded ",
            "This is a test then we can rule out decode ",
            "Line three here pinpoint exactly where the ",
            "And a fourth line The issue is happening ",
        ])
    else:
        generate()
