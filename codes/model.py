# ============================================================
# 🧠 LineCNN_OCR — CTC OCR for a single text line
# ============================================================
# Input  : (B, 1, 64, 192)   — single line, portrait width
# Output : (T=96, B, 80)     — CTC log-softmax
#
# Pipeline:
#   Full image → horizontal projection → cut line strips
#   → each strip → this model → greedy/beam decode
#   → join lines → paragraph text
#
# 96 time steps / ~20 chars per line = 4.8× CTC ratio  ✅
# ~1.1M params
# ============================================================

import torch
import torch.nn as nn

# ──────────────────────────────────────────
# CHARSET  (index 0 = CTC blank)
# ──────────────────────────────────────────
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-()/@#%&"
)
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
idx_to_char[0] = ""
NUM_CLASSES = len(CHARS) + 1   # 80


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LineCNN_OCR(nn.Module):
    """
    Spatial flow (H × W):
        Input         :  64 × 192
        After Block 1 :  32 ×  96   MaxPool(2,2)
        After Block 2 :  16 ×  96   MaxPool(2,1)  ← H only
        After Block 3 :   8 ×  96   MaxPool(2,1)
        After Block 4 :   4 ×  96   MaxPool(2,1)
        AvgPool(H)    :   1 ×  96   → (B, 128, 96)
        Conv1d        :  80 ×  96   → (96, B, 80) CTC

    Single line input → 96 time steps → ~20 chars/line comfortably.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = double_conv(1,   32)
        self.pool1  = nn.MaxPool2d((2, 2))   # 64×192 → 32×96

        self.block2 = double_conv(32,  64)
        self.pool2  = nn.MaxPool2d((2, 1))   # 32×96  → 16×96

        self.block3 = double_conv(64,  128)
        self.pool3  = nn.MaxPool2d((2, 1))   # 16×96  →  8×96

        self.block4 = double_conv(128, 128)
        self.pool4  = nn.MaxPool2d((2, 1))   #  8×96  →  4×96

        self.height_pool = nn.AdaptiveAvgPool2d((1, None))  # 4×96 → 1×96

        self.dropout    = nn.Dropout(p=0.2)
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, 64, 192)
        x = self.pool1(self.block1(x))    # (B,  32, 32, 96)
        x = self.pool2(self.block2(x))    # (B,  64, 16, 96)
        x = self.pool3(self.block3(x))    # (B, 128,  8, 96)
        x = self.pool4(self.block4(x))    # (B, 128,  4, 96)
        x = self.height_pool(x)           # (B, 128,  1, 96)
        x = x.squeeze(2)                  # (B, 128, 96)
        x = self.dropout(x)
        x = self.classifier(x)            # (B,  80, 96)
        x = x.permute(2, 0, 1)           # (96, B, 80)
        x = x.log_softmax(2)
        return x


# ──────────────────────────────────────────
# CTC GREEDY DECODER
# ──────────────────────────────────────────
def ctc_greedy_decode(log_probs):
    """log_probs: (T, C) → string"""
    indices = log_probs.argmax(dim=1).tolist()
    collapsed = [indices[0]]
    for idx in indices[1:]:
        if idx != collapsed[-1]:
            collapsed.append(idx)
    return "".join(idx_to_char[i] for i in collapsed if i != 0)


if __name__ == "__main__":
    model = LineCNN_OCR()
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LineCNN_OCR  |  params: {total:,}  ({total/1e6:.3f} M)")
    dummy = torch.randn(4, 1, 64, 192)
    out   = model(dummy)
    print(f"✅ Output: {tuple(out.shape)}  (expected (96, 4, {NUM_CLASSES}))")
