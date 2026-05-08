# ============================================================
# 📦 Dataset Generator — LineCNN_OCR
# ============================================================
# Run modes:
#   python generate_dataset.py           → 60k training samples
#   python generate_dataset.py --test    → test_line.png (480×64)
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
LINE_W      = 480
LINE_H      = 64
MAX_CHARS   = 80    # conservative: 120 steps / 55 chars = 2.18× CTC ratio

CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-()/@#%&"
)
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────
# FONTS — include serif for real-doc coverage
# ──────────────────────────────────────────
FONT_PATHS = [
    # Sans-serif
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # Monospace
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    # Serif — critical for real PDF/book text
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
]
AVAILABLE_FONTS = [p for p in FONT_PATHS if os.path.exists(p)]
if not AVAILABLE_FONTS:
    AVAILABLE_FONTS = [None]
print(f"✅ Found {len(AVAILABLE_FONTS)} font(s): {[os.path.basename(f) for f in AVAILABLE_FONTS if f]}")


def get_font(size=None):
    path = random.choice(AVAILABLE_FONTS)
    size = size or random.randint(14, 35)  # was 20, 35
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


# ──────────────────────────────────────────
# WORDS
# ──────────────────────────────────────────
with open("words_alpha.txt") as f:
    WORDS = [w.strip() for w in f if 2 <= len(w.strip()) <= 9]

PUNCT_END  = [".", ",", "!", "?", ";", ":", "--"]
PUNCT_WRAP = [("(", ")"), ('"', '"')]


def random_word():
    w = random.choice(WORDS)
    if random.random() < 0.25:
        w = w.capitalize()
    if random.random() < 0.15:
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
    for _ in range(random.randint(15, 20)):
        w = random_number_token() if random.random() < 0.06 else random_word()
        w_clean = "".join(c for c in w if c in char_to_idx)
        if not w_clean:
            continue
        sep = " " if words else ""
        if total + len(sep) + len(w_clean) > MAX_CHARS:
            break
        words.append(w_clean)
        total += len(sep) + len(w_clean)
    if not words:
        print("hi")
        words = [random.choice(WORDS)[:8]]
    return " ".join(words)


# ──────────────────────────────────────────
# RENDER single line → (LINE_H × LINE_W)
# Preserves aspect ratio, pads with white
# ──────────────────────────────────────────
def render_line(text, font, line_w=LINE_W, line_h=LINE_H):
    # Vary render height to simulate different camera distances/font sizes
    RENDER_H = random.choice([40, 50, 60, 70, 80, 90])  # was fixed 80
    pad_x    = random.randint(4, 12)
    pad_y    = random.randint(2, 6)

    dummy = Image.new("L", (8192, RENDER_H), 255)
    draw  = ImageDraw.Draw(dummy)
    bb    = draw.textbbox((pad_x, pad_y), text, font=font)
    render_w = max(bb[2] + pad_x, line_w)

    bg  = random.randint(235, 255)
    img = Image.new("L", (render_w, RENDER_H), bg)

    # occasional faint ruled line (simulates lined paper)
    if random.random() < 0.15:
        d = ImageDraw.Draw(img)
        d.line([(0, RENDER_H*2//3), (render_w, RENDER_H*2//3)],
               fill=random.randint(200, 220))

    draw = ImageDraw.Draw(img)
    # ink darkness: real documents are usually near-black (0–20)
    ink = random.randint(0, 25) if random.random() < 0.8 else random.randint(25, 60)
    draw.text((pad_x, pad_y), text, font=font, fill=ink)

    # slight rotation (camera/scan tilt)
    if random.random() < 0.2:
        img = img.rotate(random.uniform(-1.0, 1.0),
                         fillcolor=bg, expand=False)

    arr = np.array(img)

    # ── aspect-ratio-preserving resize ────────────────────
    # Scale so height = line_h, then pad width to line_w
    h_orig, w_orig = arr.shape
    scale    = line_h / h_orig
    new_w    = int(w_orig * scale)
    resized  = cv2.resize(arr, (new_w, line_h), interpolation=cv2.INTER_AREA)

    if new_w >= line_w:
        scale_w = line_w / w_orig
        new_h   = max(1, int(RENDER_H * scale_w))
        out     = cv2.resize(arr, (line_w, new_h), interpolation=cv2.INTER_AREA)
        padded  = np.full((line_h, line_w), int(bg), dtype=np.uint8)
        y_off   = (line_h - new_h) // 2
        padded[y_off:y_off + new_h, :] = out
        out     = padded
    else:
        out = np.full((line_h, line_w), int(bg), dtype=np.uint8)
        out[:, :new_w] = resized

    return out


def augment(arr):
    if random.random() < 0.3:          # was 0.2
        factor = random.uniform(1.2, 2.0)   # was 1.2, 1.6 — extend upper end
        small_h = max(16, int(arr.shape[0] / factor))
        small_w = max(60, int(arr.shape[1] / factor))
        small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        arr = cv2.resize(small, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)
    # rest unchanged...

    # existing augmentations below...
    if random.random() < 0.5:
        arr = np.clip(arr.astype(np.float32)
                      * random.uniform(0.80, 1.20)
                      + random.randint(-15, 15), 0, 255)
    # blur (simulate camera defocus / print quality)
    if random.random() < 0.3:
        pil = Image.fromarray(arr.astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(
              radius=random.uniform(0.2, 0.7)))
        arr = np.array(pil, dtype=np.float32)
    # salt & pepper noise
    if random.random() < 0.15:
        mask      = np.random.rand(*arr.shape) < 0.005
        arr[mask] = np.random.randint(0, 256, mask.sum())
    return arr.astype(np.uint8)


# ──────────────────────────────────────────
# HORIZONTAL PROJECTION LINE SPLITTER
# ──────────────────────────────────────────
def split_lines_projection(gray_np, debug=False):
    """
    Two-pass splitter:
      Pass 1: NO dilation — handles real documents with clear gaps
      Pass 2: 3px dilation — fallback for fragmented/synthetic text

    Pass 2 only triggers if Pass 1 gives suspiciously large bands.
    """
    H, W = gray_np.shape

    _, binary = cv2.threshold(
        gray_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thresh = W * 0.005

    def find_bands(img_bin, dil_h=0):
        if dil_h > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dil_h))
            img_bin = cv2.dilate(img_bin, kernel)
        proj    = img_bin.sum(axis=1).astype(np.float32)
        is_text = proj > thresh
        bands, inside, y0 = [], False, 0
        for y in range(H):
            if is_text[y] and not inside:
                inside, y0 = True, y
            elif not is_text[y] and inside:
                inside = False
                if (y - y0) >= 4:
                    bands.append([y0, y])
        if inside and (H - y0) >= 4:
            bands.append([y0, H])
        return bands

    # Pass 1: no dilation
    bands = find_bands(binary, dil_h=0)

    # Pass 2: retry with dilation only if bands look merged
    if bands:
        med_h = sorted(b - a for a, b in bands)[len(bands) // 2]
        if med_h > H / 4 and len(bands) <= 2:
            bands = find_bands(binary, dil_h=3)

    # add padding
    padded = []
    for y0, y1 in bands:
        pad = max(2, (y1 - y0) // 8)
        padded.append([max(0, y0 - pad), min(H, y1 + pad)])

    # merge close bands
    if padded:
        med_h   = sorted(b - a for a, b in padded)[len(padded) // 2]
        min_gap = max(2, int(med_h * 0.2))
        merged  = []
        for seg in padded:
            if merged and (seg[0] - merged[-1][1]) < min_gap:
                merged[-1][1] = seg[1]
            else:
                merged.append(list(seg))
        padded = merged

    if debug:
        print(f"  [splitter] image={W}×{H}  thresh={thresh:.1f}"
              f"  found {len(padded)} line(s): {padded}")

    return [(a, b) for a, b in padded]


# ──────────────────────────────────────────
# GENERATE
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
# TEST HELPERS
# ──────────────────────────────────────────
def make_test_line(filename="test_line.png", font_size=32):
    font = get_font(size=font_size)
    text = generate_line_text()
    arr  = render_line(text, font)
    Image.fromarray(arr).save(filename)
    print(f"✅ {filename}  ({LINE_W}×{LINE_H}px)  label: '{text}'")


def make_test_para(lines, filename="test_para.png",
                   font_size=40, line_gap=25):
    font_obj = get_font(size=font_size)
    pad      = 16

    dummy = Image.new("L", (4096, 512), 255)
    draw  = ImageDraw.Draw(dummy)
    bbs      = [draw.textbbox((0, 0), l, font=font_obj) for l in lines]
    line_hs  = [bb[3] - bb[1] for bb in bbs]
    max_w    = max(bb[2] - bb[0] for bb in bbs)

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
    print(f"✅ {filename}  ({w}×{h}px)")
    print(f"   {len(lines)} lines, gap={line_gap}px")

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
            "Hello world if the saved image itself is not padded",
            "This is a test then we can rule out decode",
            "Line three here pinpoint exactly where the",
            "And a fourth line The issue is happening",
        ])
    else:
        generate()
