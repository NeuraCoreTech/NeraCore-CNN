# ============================================================
# 🏋️ Training + Inference — LineCNN_OCR
# ============================================================
# Usage:
#   python train.py                      → train on data_line/
#   python train.py image.png            → OCR single-line image (192×64)
#   python train.py image.png --para     → OCR paragraph image (any landscape)
#   python train.py image.png --beam=N   → beam search width N
#   python train.py image.png --para --beam=5 --debug
#
# Image orientation contract:
#   Single line : landscape 192×64  (wider than tall)  — never rotate
#   Paragraph   : landscape any×any (wider than tall)  — never rotate
#   The splitter operates on the image as-is.
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
import os, sys
import cv2

from model import LineCNN_OCR, ctc_greedy_decode, NUM_CLASSES
from model import char_to_idx, idx_to_char
from generate_dataset import split_lines_projection

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
DATA_DIR    = "data_line"
SAVE_PATH   = "model_line.pth"
LINE_W      = 240
LINE_H      = 64
T_STEPS     = 120
MAX_LABEL   = T_STEPS - 4

BATCH_SIZE  = 32
ACCUM_STEPS = 2
EPOCHS      = 100
LR          = 5e-4
VAL_SPLIT   = 0.05
NUM_WORKERS = 4
PATIENCE    = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device : {device}")


# ──────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────
class LineDataset(Dataset):
    def __init__(self, root):
        self.root  = root
        self.items = []
        skipped    = 0

        for f in sorted(os.listdir(root)):
            if not f.endswith(".png"):
                continue
            stem     = f.replace(".png", "")
            txt_path = os.path.join(root, stem + ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as fh:
                text = fh.read().strip()

            encoded = [char_to_idx[c] for c in text if c in char_to_idx]
            if len(encoded) == 0 or len(encoded) > MAX_LABEL:
                skipped += 1
                continue

            self.items.append((stem, encoded))

        print(f"   Loaded {len(self.items)} samples  (skipped {skipped})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, encoded = self.items[idx]
        img = Image.open(os.path.join(self.root, stem + ".png")).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)    # (1, 64, 192)
        return img, torch.tensor(encoded, dtype=torch.long), len(encoded)


def ctc_collate(batch):
    imgs, labels, lengths = zip(*batch)
    imgs          = torch.stack(imgs)
    label_lengths = torch.tensor(lengths, dtype=torch.long)
    labels_cat    = torch.cat(labels)
    input_lengths = torch.full((len(imgs),), T_STEPS, dtype=torch.long)
    return imgs, labels_cat, input_lengths, label_lengths


# ──────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────
def cer(pred, target):
    if not target:
        return 0.0 if not pred else 1.0
    m, n = len(pred), len(target)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = min(prev[j]+1, dp[j-1]+1,
                        prev[j-1] + (0 if pred[i-1]==target[j-1] else 1))
    return dp[n] / n


def decode_label(flat, lengths):
    texts, off = [], 0
    for l in lengths.tolist():
        texts.append("".join(idx_to_char.get(i, "")
                             for i in flat[off:off+l].tolist()))
        off += l
    return texts


def evaluate(model, loader, n_batches=20):
    model.eval()
    cers = []
    with torch.no_grad():
        for b, (imgs, labels_cat, input_lengths, label_lengths) in enumerate(loader):
            if b >= n_batches:
                break
            lp      = model(imgs.to(device))
            targets = decode_label(labels_cat, label_lengths)
            for i in range(lp.size(1)):
                cers.append(cer(ctc_greedy_decode(lp[:, i, :].cpu()),
                                targets[i]))
    model.train()
    return float(np.mean(cers)) if cers else 1.0


# ──────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────
def train():
    full_ds  = LineDataset(DATA_DIR)
    val_size = int(len(full_ds) * VAL_SPLIT)
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(full_ds, [trn_size, val_size],
                                   generator=torch.Generator().manual_seed(42))

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=ctc_collate, num_workers=NUM_WORKERS,
                            pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=ctc_collate, num_workers=NUM_WORKERS,
                            pin_memory=True)

    print(f"📊 Train: {trn_size}  |  Val: {val_size}")
    print(f"📦 Effective batch: {BATCH_SIZE}×{ACCUM_STEPS} = {BATCH_SIZE*ACCUM_STEPS}")

    model   = LineCNN_OCR(NUM_CLASSES).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"🧠 Params: {total_p:,}  ({total_p/1e6:.3f} M)")

    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    best_cer_val, patience_ctr = float("inf"), 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for bi, (imgs, labels_cat, input_lengths, label_lengths) in enumerate(trn_loader):
            imgs          = imgs.to(device)
            labels_cat    = labels_cat.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            lp   = model(imgs)
            loss = criterion(lp, labels_cat, input_lengths, label_lengths)
            (loss / ACCUM_STEPS).backward()
            epoch_loss += loss.item()
            steps      += 1

            if (bi + 1) % ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        val_cer  = evaluate(model, val_loader)
        improved = val_cer < best_cer_val

        if improved:
            best_cer_val = val_cer
            patience_ctr = 0
            torch.save(model.state_dict(), SAVE_PATH)
            marker = "  ✅ best"
        else:
            patience_ctr += 1
            marker = f"  (no improve {patience_ctr}/{PATIENCE})"

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={epoch_loss/steps:.4f}  "
              f"val_CER={val_cer:.4f}{marker}")

        if patience_ctr >= PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch}.")
            break

    print(f"\n✅ Done. Best val CER: {best_cer_val:.4f}  →  {SAVE_PATH}")


# ──────────────────────────────────────────
# PREPROCESSING — no rotation anywhere
# ──────────────────────────────────────────
def load_image(image_path):
    """Load any image as grayscale numpy array. Never rotate."""
    return np.array(Image.open(image_path).convert("L"))


def line_to_tensor(gray_np):
    """
    Resize a line crop to (64 × 192) and return (1,1,64,192) tensor.
    Input should be landscape (wider than tall).
    """
    resized = cv2.resize(gray_np, (LINE_W, LINE_H), interpolation=cv2.INTER_AREA)
    return torch.tensor(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


# ──────────────────────────────────────────
# DECODE ONE STRIP
# ──────────────────────────────────────────
def decode_strip(model, gray_crop, beam_width=1):
    x  = line_to_tensor(gray_crop).to(device)
    with torch.no_grad():
        lp = model(x)[:, 0, :].cpu()    # (96, 80)
    return ctc_greedy_decode(lp) if beam_width <= 1 else ctc_beam_decode(lp, beam_width)

def add_white_padding(gray_np, pad=40):
    return cv2.copyMakeBorder(
        gray_np,
        top=pad,
        bottom=pad,
        left=pad,
        right=pad,
        borderType=cv2.BORDER_CONSTANT,
        value=255  # white for grayscale
    )
# ──────────────────────────────────────────
# INFER
# ──────────────────────────────────────────
def infer(image_path, paragraph_mode=False, beam_width=1, debug=False):
    model = LineCNN_OCR(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()

    gray = load_image(image_path)    # no rotation — ever

    if debug:
        h, w = gray.shape
        print(f"  [debug] loaded image: {w}×{h}px")

    if paragraph_mode:
        strips = split_lines_projection(gray, debug=debug)

        if not strips:
            print("⚠️  No lines detected — using whole image as one line.")
            strips = [(0, gray.shape[0])]

        print(f"🔍 Detected {len(strips)} line(s)\n")
        results = []

        for i, (y0, y1) in enumerate(strips):
            crop = gray[y0:y1, :]
            #My edit START
            crop = add_white_padding(crop, pad=20)
            #My edit END

            if debug:
                Image.fromarray(crop).save(f"debug_line_{i}.png")
                h_c, w_c = crop.shape
                print(f"  [debug] line {i+1}: rows {y0}–{y1}  ({w_c}×{h_c}px)")
            text = decode_strip(model, crop, beam_width)
            print(f"  Line {i+1}: {text}")
            results.append(text)

        print(f"\n📝 Full text:\n" + "\n".join(results))

    else:
        text = decode_strip(model, gray, beam_width)
        print(f"📝 Decoded: '{text}'")


# ──────────────────────────────────────────
# BEAM SEARCH
# ──────────────────────────────────────────
def ctc_beam_decode(log_probs, beam_width=5):
    T, C  = log_probs.shape
    probs = log_probs.exp().numpy()
    beams = [([], 0.0)]
    for t in range(T):
        nb = {}
        for seq, sc in beams:
            for c in range(C):
                p = float(probs[t, c])
                if p < 1e-10:
                    continue
                key = tuple(seq + [c])
                s   = sc + np.log(p + 1e-10)
                if key not in nb or nb[key] < s:
                    nb[key] = s
        beams = sorted(nb.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(list(k), v) for k, v in beams]
    if not beams:
        return ""
    best = beams[0][0]
    col  = [best[0]] if best else []
    for idx in best[1:]:
        if idx != col[-1]:
            col.append(idx)
    return "".join(idx_to_char.get(i, "") for i in col if i != 0)


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        beam  = 1
        para  = "--para"  in sys.argv
        debug = "--debug" in sys.argv
        for arg in sys.argv:
            if arg.startswith("--beam"):
                parts = arg.split("=") if "=" in arg else [arg, "5"]
                beam  = int(parts[1]) if len(parts) > 1 else 5
        infer(sys.argv[1], paragraph_mode=para,
              beam_width=beam, debug=debug)
    else:
        train()
