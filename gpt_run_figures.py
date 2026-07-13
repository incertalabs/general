"""
gpt_run_figures.py
------------------
Slide figures from ONE completed `gpt_train_instrumented.py` run — no retraining
(checkpoint + logs only, per the CLAUDE.md scope amendment).

Reads (defaults match gpt_train_instrumented.py's output paths):
    gpt_run_outputs/<tag>_losses.csv
    gpt_run_outputs/<tag>_samples.txt
    gpt_run_outputs/<tag>_ckpt.pt        (model + vocab + config)

Writes back-row-legible figures (Okabe-Ito, large fonts) to ../figures/:
    gpt_loss_annotated_<tag>.png   loss curve + the ln(V) random-guessing baseline
    gpt_samples_panel_<tag>.png    the three saved samples as one slide panel
    gpt_nextchar_bars_<tag>.png    P(next char | prompt) — what a base model IS
    gpt_embedding_map_<tag>.png    the learned char embeddings in 2D (PCA):
                                   digits/letters/punctuation cluster — geometry = meaning
    gpt_temperature_<tag>.png      same logits at T = 0.6 / 1.0 / 1.3

Usage (on the machine that holds the run outputs):
    python gpt_run_figures.py --tag cpsc
    python gpt_run_figures.py --tag cpsc --prompt "THE CONSUMER REPORTED THAT THE "

Only main() needs torch; the figure functions are pure numpy/matplotlib.
"""

import argparse
import csv
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Okabe–Ito (colorblind-safe), matching the wb_* figures
C_BLUE, C_ORANGE, C_SKY, C_GREEN, C_PINK, C_YELLOW = (
    "#0072B2", "#D55E00", "#56B4E9", "#009E73", "#CC79A7", "#E69F00")

plt.rcParams.update({"font.size": 15, "axes.titlesize": 19,
                     "axes.labelsize": 17, "figure.dpi": 110})


def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)


def show_char(ch: str) -> str:
    """Printable stand-in for whitespace/control characters."""
    return {" ": "␣", "\n": "\\n", "\t": "\\t", "\r": "\\r"}.get(ch, ch)


# ---------------------------------------------------------------------------
# 1 · loss curve, annotated with the random-guessing baseline
# ---------------------------------------------------------------------------
def load_losses(csv_path: Path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    va = [float(r["val_loss"]) for r in rows]
    return steps, tr, va


def fig_loss_annotated(steps, tr, va, vocab_size, fig_path: Path):
    baseline = math.log(vocab_size)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(baseline, color="0.45", lw=2.5, ls="--",
               label=f"random guessing: ln({vocab_size}) = {baseline:.2f}")
    ax.plot(steps, tr, color=C_BLUE, lw=3, marker="o", ms=7, label="train loss")
    ax.plot(steps, va, color=C_ORANGE, lw=3, marker="s", ms=7, label="val loss")
    ax.annotate(f"final val: {va[-1]:.2f}", xy=(steps[-1], va[-1]),
                xytext=(-25, 35), textcoords="offset points", fontsize=15,
                color=C_ORANGE, ha="right",
                arrowprops=dict(arrowstyle="->", color=C_ORANGE))
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats/char)")
    ax.set_title("What the 45 minutes bought — vs. knowing nothing")
    ax.set_xlim(left=0)
    ax.legend(fontsize=15, frameon=False, loc="upper right")
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2 · the three saved samples as one slide panel
# ---------------------------------------------------------------------------
def parse_samples(text: str):
    """[(label, sample_text), ...] from <tag>_samples.txt."""
    return [(m.group(1), m.group(2).strip())
            for m in re.finditer(r"^--- (.*?) ---\n(.*?)(?=^--- |\Z)",
                                 text, re.S | re.M)]


def fig_samples_panel(samples, fig_path: Path, max_chars=260, wrap=88):
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n))
    axes = np.atleast_1d(axes)
    for ax, (label, body) in zip(axes, samples):
        snippet = body[:max_chars] + ("…" if len(body) > max_chars else "")
        wrapped = "\n".join(textwrap.wrap(snippet.replace("\n", " ↵ "), wrap))
        ax.set_axis_off()
        ax.set_title(label, fontsize=15, loc="left", color=C_BLUE)
        ax.text(0, 0.95, wrapped, family="monospace", fontsize=11.5,
                va="top", ha="left", transform=ax.transAxes)
    fig.suptitle("The base model writes: fake incident reports, one char at a time",
                 fontsize=18, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3 · P(next char | prompt) — what a base model actually outputs
# ---------------------------------------------------------------------------
def fig_nextchar_bars(probs: np.ndarray, itos: dict, prompt: str,
                      fig_path: Path, top_n=15):
    top = np.argsort(probs)[::-1][:top_n]
    labels = [show_char(itos[int(i)]) for i in top]
    y = np.arange(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [C_ORANGE] + [C_BLUE] * (top_n - 1)
    ax.barh(y, probs[top], color=colors)
    for yi, p in zip(y, probs[top]):
        ax.text(p, yi, f" {p:.2f}", va="center", fontsize=13)
    ax.set_yticks(y, labels, family="monospace", fontsize=16)
    ax.set_xlabel("P(next character)")
    ax.set_title(f'The entire job of a base model:\n'
                 f'what comes after “…{prompt[-22:]}”?', fontsize=17)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4 · the character embedding table in 2D — geometry = meaning, at char level
# ---------------------------------------------------------------------------
def pca_2d(W: np.ndarray):
    X = W - W.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    xy = X @ Vt[:2].T
    evr = (S**2 / (S**2).sum())[:2]
    return xy, evr


CHAR_CLASSES = [  # (name, predicate, color)
    ("uppercase", str.isupper, C_BLUE),
    ("lowercase", str.islower, C_SKY),
    ("digit", str.isdigit, C_ORANGE),
    ("whitespace", str.isspace, C_PINK),
    ("punct/other", lambda c: True, C_GREEN),  # fallback
]


def char_color(ch: str) -> str:
    for _, pred, color in CHAR_CLASSES:
        if pred(ch):
            return color
    return C_GREEN


def fig_embedding_map(W: np.ndarray, itos: dict, fig_path: Path):
    xy, evr = pca_2d(W)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(W.shape[0]):
        ch = itos[int(i)]
        ax.text(xy[i, 0], xy[i, 1], show_char(ch), fontsize=14,
                family="monospace", fontweight="bold",
                color=char_color(ch), ha="center", va="center")
    pad = 0.08 * (xy.max(0) - xy.min(0))
    ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
    ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
    ax.set_xlabel(f"PC 1 ({evr[0]:.0%} of variance)")
    ax.set_ylabel(f"PC 2 ({evr[1]:.0%})")
    ax.set_title(f"The learned embedding table, in 2D — "
                 f"{W.shape[0]} chars × {W.shape[1]} dims\n"
                 "nobody told it letters ≠ digits: geometry = meaning",
                 fontsize=17)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n, ms=12)
               for n, _, c in CHAR_CLASSES]
    ax.legend(handles=handles, fontsize=13, frameon=False, loc="best")
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5 · same logits, three temperatures — the sampling knob
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def fig_temperature_panel(logits: np.ndarray, itos: dict, prompt: str,
                          fig_path: Path, temps=(0.6, 1.0, 1.3), top_n=12):
    order = np.argsort(softmax(logits))[::-1][:top_n]  # fixed order: T=1 ranking
    labels = [show_char(itos[int(i)]) for i in order]
    y = np.arange(top_n)[::-1]
    fig, axes = plt.subplots(1, len(temps), figsize=(13, 5.5), sharey=True)
    for ax, T in zip(axes, temps):
        p = softmax(logits / T)
        ax.barh(y, p[order], color=C_BLUE if T == 1.0 else C_GREEN)
        ax.set_title(f"T = {T}", fontsize=17,
                     color=C_BLUE if T == 1.0 else "black")
        ax.set_xlabel("P(next char)")
        _tidy(ax)
    axes[0].set_yticks(y, labels, family="monospace", fontsize=15)
    fig.suptitle(f"Same logits after “…{prompt[-18:]}” — "
                 "temperature reshapes the distribution", fontsize=17)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main — the only torch-dependent part
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu")
    except Exception:  # older/newer torch weights_only default differences
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="cpsc")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "gpt_run_outputs")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--prompt", default="THE CONSUMER REPORTED THAT THE ")
    p.add_argument("--top-n", type=int, default=15)
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from gpt_train_instrumented import GPTLanguageModel

    ckpt = load_checkpoint(args.out_dir / f"{args.tag}_ckpt.pt")
    cfg, stoi, itos = ckpt["config"], ckpt["stoi"], ckpt["itos"]

    # 1 · loss curve (+ ln V baseline)
    steps, tr, va = load_losses(args.out_dir / f"{args.tag}_losses.csv")
    fig_loss_annotated(steps, tr, va, cfg["vocab_size"],
                       args.fig_dir / f"gpt_loss_annotated_{args.tag}.png")

    # 2 · samples panel
    samples = parse_samples(
        (args.out_dir / f"{args.tag}_samples.txt").read_text(encoding="utf-8"))
    fig_samples_panel(samples, args.fig_dir / f"gpt_samples_panel_{args.tag}.png")

    # 3+5 · model forward on the prompt -> next-char distribution + temperature
    model = GPTLanguageModel(cfg["vocab_size"], cfg["n_embd"], cfg["n_head"],
                             cfg["n_layer"], cfg["block_size"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids = [stoi[c] for c in args.prompt if c in stoi] or [0]
    dropped = [c for c in args.prompt if c not in stoi]
    if dropped:
        print(f"note: dropped chars not in vocab: {dropped}")
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids[-cfg["block_size"]:]]))
    logits = logits[0, -1].numpy()

    fig_nextchar_bars(softmax(logits), itos, args.prompt,
                      args.fig_dir / f"gpt_nextchar_bars_{args.tag}.png",
                      top_n=args.top_n)
    fig_temperature_panel(logits, itos, args.prompt,
                          args.fig_dir / f"gpt_temperature_{args.tag}.png")

    # 4 · embedding map (needs only the weight matrix, not the model)
    W = ckpt["model_state"]["token_embedding.weight"].numpy()
    fig_embedding_map(W, itos, args.fig_dir / f"gpt_embedding_map_{args.tag}.png")

    print(f"5 figures written to {args.fig_dir} (tag: {args.tag})")


if __name__ == "__main__":
    main()
