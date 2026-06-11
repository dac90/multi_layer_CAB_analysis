"""
Neural Network Trainer with Architecture String DSL
====================================================

Architecture string format:
  - Numbers specify the width of the next layer
  - Letters specify the layer type:
      R = ReLU activation
      T = Tanh activation
      S = Sigmoid activation
      G = GELU activation
      I = Identity (no activation)
      B = BatchNorm1d
      L = LayerNorm
      D = Dropout(0.5)

Examples:
  "64R128R64R"     → Linear(in,64)->ReLU->Linear(64,128)->ReLU->Linear(128,64)->ReLU->Linear(64,out)
  "32T64TB"        → Linear(in,32)->Tanh->Linear(32,64)->Tanh->BatchNorm->Linear(64,out)
  "128G256GD64G"   → Linear(in,128)->GELU->Linear(128,256)->GELU->Dropout->Linear(256,64)->GELU->Linear(64,out)

Usage:
  python train_nn.py --arch "64R128R64R" [OPTIONS]

Options:
  --arch        Architecture string (required)
  --epochs      Number of training epochs (default: 200)
  --lr          Learning rate (default: 1e-3)
  --batch-size  Mini-batch size (default: 256)
  --n-samples   Dataset size (default: 2000)
  --noise       Moon dataset noise (default: 0.15)
  --wandb-proj  WandB project name (default: "nn-explorer")
  --wandb-run   WandB run name (optional)
  --save-weights-every  Save weight string every N steps (default: 10)
  --plot-every  Plot every N steps (default: 10)
  --device      Force device: "cpu" | "cuda" | "mps" (auto-detected by default)
"""

import re
import sys
import json
import time
import base64
import argparse
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for speed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("[warn] wandb not installed – install with `pip install wandb`. Logging disabled.")
    WANDB_AVAILABLE = False


# ──────────────────────────────────────────────
# 1.  Architecture parser
# ──────────────────────────────────────────────

ACTIVATIONS = {
    "R": nn.ReLU(),
    "T": nn.Tanh(),
    "S": nn.Sigmoid(),
    "G": nn.GELU(),
    "I": nn.Identity(),  # pass-through / no activation
}

NON_ACTIVATION_LETTERS = {
    "B": lambda w: nn.BatchNorm1d(w),   # needs width → deferred
    "L": lambda w: nn.LayerNorm(w),     # needs width → deferred
    "D": lambda w: nn.Dropout(0.5),
}


def parse_arch(arch_str: str, in_features: int, out_features: int) -> nn.Sequential:
    """
    Parse an architecture string into a nn.Sequential model.

    Grammar (informal):
        token  := number | letter
        number := [0-9]+          (sets the next linear layer's output width)
        letter := R|T|S|G|L|B|D  (inserts a module)

    A Linear layer is emitted every time the width changes (i.e. when a new
    number appears) OR at the very end (input → first_width, …, last_width → out).
    Activation / BN / Dropout letters are inserted after the preceding Linear.
    """
    tokens = re.findall(r"\d+|[A-Za-z]", arch_str.upper())

    widths = []   # ordered list of hidden widths
    mods_after: dict[int, list] = {}  # index into widths → list of module factories

    current_idx = -1
    for tok in tokens:
        if tok.isdigit() or (len(tok) > 1 and tok.isdigit()):
            pass  # handled below
        if re.fullmatch(r"\d+", tok):
            widths.append(int(tok))
            current_idx += 1
            mods_after[current_idx] = []
        elif tok in ACTIVATIONS:
            if current_idx < 0:
                raise ValueError(f"Letter '{tok}' before any width in arch string.")
            mods_after[current_idx].append(("act", tok))
        elif tok in NON_ACTIVATION_LETTERS:
            if current_idx < 0:
                raise ValueError(f"Letter '{tok}' before any width in arch string.")
            mods_after[current_idx].append(("special", tok))
        else:
            raise ValueError(f"Unknown token '{tok}' in arch string.")

    if not widths:
        raise ValueError("Architecture string must contain at least one width (number).")

    layer_dims = [in_features] + widths + [out_features]
    layers: list[nn.Module] = []

    for i, (w_in, w_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(w_in, w_out))
        # Apply mods associated with the *output* width (index i in widths list)
        if i < len(widths):  # last linear has no mods
            for kind, tok in mods_after.get(i, []):
                if kind == "act":
                    act = ACTIVATIONS[tok]
                    # Create a fresh copy so the same letter can appear multiple times
                    layers.append(type(act)() if not isinstance(act, nn.Identity) else nn.Identity())
                else:  # special
                    factory = NON_ACTIVATION_LETTERS[tok]
                    layers.append(factory(w_out))

    return nn.Sequential(*layers)


# ──────────────────────────────────────────────
# 2.  Weight serialisation
# ──────────────────────────────────────────────

def weights_to_string(model: nn.Module) -> str:
    """
    Encode all named parameters + buffers as a compact base64 JSON string.
    Format:
      {
        "arch": "<module class names>",
        "params": { "layer.weight": "<base64 float32 bytes>", ... },
        "shapes": { "layer.weight": [rows, cols], ... }
      }
    Decoding example in docstring below.
    """
    params = {}
    shapes = {}
    for name, tensor in {**dict(model.named_parameters()),
                         **dict(model.named_buffers())}.items():
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        shapes[name] = list(arr.shape)
        params[name] = base64.b64encode(arr.tobytes()).decode("ascii")

    arch_desc = " → ".join(type(m).__name__ for m in model.modules()
                            if not isinstance(m, nn.Sequential))
    payload = json.dumps({"arch": arch_desc, "params": params, "shapes": shapes},
                         separators=(",", ":"))
    return payload


def string_to_weights(model: nn.Module, s: str) -> None:
    """Load weights back from a serialised string (in-place)."""
    payload = json.loads(s)
    state = {}
    for name, b64 in payload["params"].items():
        shape = payload["shapes"][name]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)
        state[name] = torch.from_numpy(arr.copy())
    model.load_state_dict(state, strict=False)


# ──────────────────────────────────────────────
# 3.  Visualisation
# ──────────────────────────────────────────────

_CMAP_BG = mcolors.LinearSegmentedColormap.from_list(
    "soft_rb", ["#4e9af1", "#f1754e"], N=256)
_CMAP_POINTS = {0: "#1a6fd4", 1: "#c94a1a"}


@torch.no_grad()
def make_decision_boundary_plot(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epoch: int,
    loss: float,
    acc: float,
    device: torch.device,
    resolution: int = 300,
) -> plt.Figure:
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device
    )

    # Run in chunks to avoid OOM on large grids
    chunk = 4096
    probs = []
    for i in range(0, len(grid), chunk):
        logits = model(grid[i : i + chunk])
        p = torch.softmax(logits, dim=-1)[:, 1]
        probs.append(p.cpu().numpy())
    probs = np.concatenate(probs).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=110)
    ax.contourf(xx, yy, probs, levels=50, cmap=_CMAP_BG, alpha=0.85, vmin=0, vmax=1)
    ax.contour(xx, yy, probs, levels=[0.5], colors="white", linewidths=1.5,
               linestyles="--")

    for cls in (0, 1):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=_CMAP_POINTS[cls], s=14,
                   edgecolors="white", linewidths=0.4, alpha=0.85, label=f"class {cls}")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Epoch {epoch:>4d}  |  loss {loss:.4f}  |  acc {acc:.2%}",
                 fontsize=10, pad=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_xlabel("x₁", fontsize=9)
    ax.set_ylabel("x₂", fontsize=9)
    fig.tight_layout()
    model.train()
    return fig


def fig_to_wandb_image(fig: plt.Figure) -> "wandb.Image":
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = wandb.Image(buf)
    plt.close(fig)
    return img


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ──────────────────────────────────────────────
# 4.  Training loop
# ──────────────────────────────────────────────

def build_dataset(n_samples: int, noise: float):
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np).astype(np.float32)
    y_np = y_np.astype(np.int64)
    return X_np, y_np, scaler


def train(args):
    # ── device ──
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[info] Using device: {device}")

    # ── dataset ──
    X_np, y_np, _ = build_dataset(args.n_samples, args.noise)
    X_t = torch.from_numpy(X_np).to(device)
    y_t = torch.from_numpy(y_np).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=0,
    )

    # ── model ──
    model = parse_arch(args.arch, in_features=2, out_features=2)
    model = model.to(device)
    print(f"[info] Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[info] Total parameters: {total_params:,}")

    # ── optimiser & scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    # ── torch.compile (PyTorch ≥ 2.0) ──
    try:
        model = torch.compile(model)
        print("[info] torch.compile() enabled.")
    except Exception:
        print("[info] torch.compile() not available, skipping.")

    # ── WandB init ──
    if WANDB_AVAILABLE:
        run = wandb.init(
            project=args.wandb_proj,
            name=args.wandb_run or f"arch_{args.arch}",
            config={
                "arch": args.arch,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "n_samples": args.n_samples,
                "noise": args.noise,
                "device": str(device),
                "total_params": total_params,
            },
        )
        wandb.watch(model, log="gradients", log_freq=50)
    else:
        run = None

    # ── training ──
    weight_strings = []   # collected weight snapshots
    scaler_amp = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        t0 = time.perf_counter()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)

            if scaler_amp is not None:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        scheduler.step()

        avg_loss = epoch_loss / total
        acc = correct / total
        elapsed = time.perf_counter() - t0

        # ── per-epoch console log ──
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"acc={acc:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed*1000:.0f}ms")

        do_plot = (epoch % args.plot_every == 0) or epoch == 1 or epoch == args.epochs
        do_weights = (epoch % args.save_weights_every == 0) or epoch == 1 or epoch == args.epochs

        # ── weight snapshot ──
        if do_weights:
            ws = weights_to_string(model)
            weight_strings.append((epoch, ws))
            print(f"  [weights] snapshot at epoch {epoch} "
                  f"({len(ws)} chars / {len(ws)//1024} KB)")

        # ── plot + wandb log ──
        log_dict: dict = {
            "loss": avg_loss,
            "accuracy": acc,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        }

        if do_plot:
            fig = make_decision_boundary_plot(
                model, X_np, y_np, epoch, avg_loss, acc, device
            )
            if WANDB_AVAILABLE:
                log_dict["decision_boundary"] = fig_to_wandb_image(fig)
            else:
                # Save locally if WandB unavailable
                out_dir = Path("plots")
                out_dir.mkdir(exist_ok=True)
                png = fig_to_png_bytes(fig)
                out_path = out_dir / f"epoch_{epoch:04d}.png"
                out_path.write_bytes(png)
                print(f"  [plot] saved to {out_path}")

        if WANDB_AVAILABLE:
            wandb.log(log_dict, step=epoch)

    # ── final summary ──
    print("\n" + "="*60)
    print(f"Training complete.  Final accuracy: {acc:.3%}")
    print(f"Collected {len(weight_strings)} weight snapshots.")

    # Write last weight string to stdout for downstream use
    if weight_strings:
        last_epoch, last_ws = weight_strings[-1]
        summary_path = Path("weight_snapshots.jsonl")
        with summary_path.open("w") as f:
            for ep, ws in weight_strings:
                f.write(json.dumps({"epoch": ep, "weights": ws}) + "\n")
        print(f"\nWeight snapshots written to: {summary_path}")
        print(f"\nLast weight string (epoch {last_epoch}):")
        print(last_ws[:200] + "…" if len(last_ws) > 200 else last_ws)

    if WANDB_AVAILABLE and run is not None:
        # Log final model weights as artifact
        artifact = wandb.Artifact("model_weights", type="model")
        artifact.add_file(str(summary_path))
        run.log_artifact(artifact)
        wandb.finish()

    return weight_strings


# ──────────────────────────────────────────────
# 5.  CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a configurable neural net on the moons dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arch", required=True,
                   help='Architecture string, e.g. "64R128R64R"')
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--wandb-proj", default="nn-explorer")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--save-weights-every", type=int, default=10)
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument("--device", default=None,
                   choices=["cpu", "cuda", "mps"],
                   help="Force a specific device (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)