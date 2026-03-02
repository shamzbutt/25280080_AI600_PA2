"""
AI600 PA2 – Quick, Draw! MLP Classifier
Final inference script with Test-Time Augmentation.

Usage:
    python inference.py
    python inference.py --model ./models/champion_best.pth --n_aug 7

Portal values:
    Model Parameters : 2,519,055
    Epochs           : 35
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

INPUT_SIZE  = 784
NUM_CLASSES = 15

# ─────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────

class PancakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 1024), nn.BatchNorm1d(1024), nn.ReLU(),  nn.Dropout(0.3),
            nn.Linear(1024, 1024),       nn.BatchNorm1d(1024), nn.ReLU(),  nn.Dropout(0.3),
            nn.Linear(1024, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x.view(-1, INPUT_SIZE))


class TowerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x.view(-1, INPUT_SIZE))


class ChampionMLP(nn.Module):
    """Maximum accuracy: 784->1024->1024->512->256->15 (~2.52M params)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(1024, 1024),       nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(1024, 512),        nn.BatchNorm1d(512),  nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 256),         nn.BatchNorm1d(256),  nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x.view(-1, INPUT_SIZE))


MODEL_MAP = {'pancake': PancakeMLP, 'tower': TowerMLP, 'champion': ChampionMLP}

# ─────────────────────────────────────────────────────────────────
# Augmentation (must match training exactly)
# ─────────────────────────────────────────────────────────────────

_affine = T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))

def augment_batch(x):
    x = x.view(-1, 1, 28, 28)
    x = _affine(x)
    flip_mask = torch.rand(x.size(0), device=x.device) > 0.5
    x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    x = x + 0.02 * torch.randn_like(x)
    if torch.rand(1).item() > 0.5:
        i = torch.randint(0, 18, (1,)).item()
        j = torch.randint(0, 18, (1,)).item()
        x[:, :, i:i+10, j:j+10] = 0
    return x.clamp(0, 1).view(-1, 784)

# ─────────────────────────────────────────────────────────────────
# Consistency guard
# ─────────────────────────────────────────────────────────────────

def check_consistency(model_path, model_type):
    basename = os.path.basename(model_path).lower()
    for name in MODEL_MAP:
        if name in basename and name != model_type:
            print(f"\n  WARNING: '{basename}' looks like '{name}' "
                  f"but --model_type is '{model_type}'. Check this!\n")
            return
    print(f"  model_type='{model_type}' matches '{basename}'. OK.")

# ─────────────────────────────────────────────────────────────────
# TTA Inference
# ─────────────────────────────────────────────────────────────────

def predict_tta(model, loader, device, n_aug=7):
    """Average predictions over n_aug augmented passes per batch."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            probs = torch.softmax(model(X_batch), dim=1)
            for _ in range(n_aug - 1):
                probs += torch.softmax(model(augment_batch(X_batch.clone())), dim=1)
            probs /= n_aug
            all_preds.extend(probs.argmax(1).cpu().numpy())
    return all_preds

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run_inference(model_path, test_data_path, model_type, batch_size, n_aug):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    check_consistency(model_path, model_type)

    # Load model
    model = MODEL_MAP[model_type]()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : {model_type}")
    print(f"  Parameters   : {total_params:,}  "
          f"{'OK' if total_params <= 3_000_000 else 'OVER 3M LIMIT!'}")
    print(f"  TTA passes   : {n_aug}")

    # Load test data — shuffle=False preserves submission order
    data   = np.load(test_data_path)
    X_test = torch.tensor(data['test_images'].astype(np.float32) / 255.0).view(-1, 784)
    loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)
    print(f"  Test samples : {len(X_test):,}")

    print(f"\nRunning TTA inference ({n_aug} passes)...")
    preds = predict_tta(model, loader, device, n_aug)

    csv_string = ','.join(map(str, preds))
    with open('submission.txt', 'w') as f:
        f.write(csv_string)

    print(f"  Total predictions : {len(preds)}")
    print(f"  Sample (first 10) : {preds[:10]}")
    print(f"\n  Saved to 'submission.txt'")
    print(f"\n{'─'*50}")
    print(f"  PASTE INTO LEADERBOARD PORTAL:")
    print(f"  Model Parameters : {total_params:,}")
    print(f"  Epochs           : 25")
    print(f"  Portal           : https://ai-600-leaderboard-tau.vercel.app/")
    print(f"{'─'*50}\n")
    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QuickDraw TTA Inference – AI600 PA2')
    parser.add_argument('--model',      default='./models/champion_best.pth')
    parser.add_argument('--test_data',  default='./processed_data/quickdraw_test.npz')
    parser.add_argument('--model_type', default='champion', choices=list(MODEL_MAP.keys()))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_aug',      type=int, default=7,
                        help='TTA passes: 1=off, 7=default, 10=best accuracy')
    args = parser.parse_args()
    run_inference(args.model, args.test_data, args.model_type,
                  args.batch_size, args.n_aug)
