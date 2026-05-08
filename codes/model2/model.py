# ============================================================
# 🧠 LineCNN_OCR — CTC OCR for a single text line
# ============================================================
# Input  : (B, 1, 64, 480)   — single line, landscape
# Output : (T=240, B, 80)    — CTC log-softmax
#
# 240 time steps / 80 max chars = 3.0× CTC ratio  ✅
# Spatial flow:
#   64×480 → MaxPool(2,2) → 32×240 → H-only pools → 1×240
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

# Expected time steps — used for validation
EXPECTED_T = 240


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
        Input         :  64 × 480
        After Block 1 :  32 × 240   MaxPool(2,2)  ← shrink both once
        After Block 2 :  16 × 240   MaxPool(2,1)  ← H only
        After Block 3 :   8 × 240   MaxPool(2,1)
        After Block 4 :   4 × 240   MaxPool(2,1)
        AvgPool(H)    :   1 × 240   → (B, 128, 240) sequence
        Conv1d        :  80 × 240   → (240, B, 80) CTC output

    240 time steps / 80 max chars = 3.0× CTC ratio  ✅
    ~700K params
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = double_conv(1,   32)
        self.pool1  = nn.MaxPool2d((2, 2))   # 64×480 → 32×240

        self.block2 = double_conv(32,  64)
        self.pool2  = nn.MaxPool2d((2, 1))   # 32×240 → 16×240

        self.block3 = double_conv(64,  128)
        self.pool3  = nn.MaxPool2d((2, 1))   # 16×240 →  8×240

        self.block4 = double_conv(128, 128)
        self.pool4  = nn.MaxPool2d((2, 1))   #  8×240 →  4×240

        self.height_pool = nn.AdaptiveAvgPool2d((1, None))  # 4×240 → 1×240

        self.dropout    = nn.Dropout(p=0.2)
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.pool1(self.block1(x))    # (B,  32, 32, 240)
        x = self.pool2(self.block2(x))    # (B,  64, 16, 240)
        x = self.pool3(self.block3(x))    # (B, 128,  8, 240)
        x = self.pool4(self.block4(x))    # (B, 128,  4, 240)
        x = self.height_pool(x)           # (B, 128,  1, 240)
        x = x.squeeze(2)                  # (B, 128, 240)
        x = self.dropout(x)
        x = self.classifier(x)            # (B,  80, 240)
        x = x.permute(2, 0, 1)           # (240, B, 80)
        x = x.log_softmax(2)
        return x


def ctc_greedy_decode(log_probs):
    """log_probs: (T, C) → string"""
    indices = log_probs.argmax(dim=1).tolist()
    collapsed = [indices[0]]
    for idx in indices[1:]:
        if idx != collapsed[-1]:
            collapsed.append(idx)
    return "".join(idx_to_char[i] for i in collapsed if i != 0)


def verify_checkpoint(path, device="cpu"):
    """
    Load checkpoint and verify its output time steps match EXPECTED_T.
    Raises a clear error if there's a mismatch instead of silently loading
    wrong weights.
    """
    ckpt  = torch.load(path, map_location=device)
    model = LineCNN_OCR(NUM_CLASSES).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    # run a dummy forward to check actual T
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 64, 480).to(device)
        out   = model(dummy)
        T_actual = out.shape[0]

    if T_actual != EXPECTED_T:
        raise RuntimeError(
            f"Checkpoint mismatch: model outputs T={T_actual} steps "
            f"but EXPECTED_T={EXPECTED_T}.\n"
            f"Delete '{path}' and retrain with the current architecture."
        )

    return model


if __name__ == "__main__":
    model = LineCNN_OCR()
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LineCNN_OCR  |  params: {total:,}  ({total/1e6:.3f} M)")
    dummy = torch.randn(4, 1, 64, 480)
    out   = model(dummy)
    print(f"✅ Output: {tuple(out.shape)}  (expected (240, 4, {NUM_CLASSES}))")
    assert out.shape == (240, 4, NUM_CLASSES), "❌ Shape mismatch!"
    print("✅ All dimensions correct")
