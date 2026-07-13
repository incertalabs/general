"""
gpt_train_instrumented.py
-------------------------
The class's `gpt_train.py`, UNCHANGED in model, hyperparameters, and seed
(faithfulness matters: "this is what *you* ran"), instrumented to capture
slide artifacts from ONE offline run (per the CLAUDE.md scope amendment,
2026-07-10 — never live on stage, never BERT, never cloud):

  - loss history        -> <out-dir>/<tag>_losses.csv
  - loss-curve figure   -> <fig-dir>/gpt_loss_curve_<tag>.png  (back-row legible)
  - generated samples   -> <out-dir>/<tag>_samples.txt   (3 temperature/top-k settings)
  - checkpoint          -> <out-dir>/<tag>_ckpt.pt        (model + vocab + config)
  - run metadata        -> <out-dir>/<tag>_meta.txt       (param count, timing, env)

Usage:
    # faithful re-run on the class corpus (~45 min CPU at 3000 iters):
    python gpt_train_instrumented.py

    # alternative corpus built from CPSC incident narratives:
    python gpt_train_instrumented.py --from-cpsc --tag cpsc

    # regenerate samples from a saved checkpoint (no retraining) —
    # also take-home exercise 6:
    python gpt_train_instrumented.py --generate-only gpt_run_outputs/article_ckpt.pt \
        --temperature 0.8 --top-k 40

Smoke test (NOT for slides): --max-iters 20 --eval-interval 10 --eval-iters 2
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# -----------------------------------------------------------------------------
# Hyperparameters — IDENTICAL to the class's gpt_train.py (incl. the seed).
# CLI may override max_iters / eval cadence for smoke tests only.
# -----------------------------------------------------------------------------
torch.manual_seed(1337)

batch_size    = 32
block_size    = 128
n_embd        = 128
n_head        = 4
n_layer       = 4
dropout       = 0.1
MAX_ITERS     = 3000
EVAL_INTERVAL = 300
EVAL_ITERS    = 50
learning_rate = 3e-4
device        = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Model — verbatim from gpt_train.py
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.qkv_proj  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj  = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# -----------------------------------------------------------------------------
# Instrumentation helpers
# -----------------------------------------------------------------------------
def build_cpsc_corpus(max_chars: int) -> str:
    """Concatenate CPSC incident narratives into one training text."""
    import pandas as pd  # local import: only needed for --from-cpsc
    csv_path = REPO_ROOT / "data" / "cpsc_merged.csv"
    df = pd.read_csv(csv_path, usecols=["incident_description"])
    text = "\n".join(df["incident_description"].dropna().astype(str))
    return text[:max_chars] if max_chars else text


def save_loss_figure(history, fig_path: Path, corpus_name: str):
    """Slide-ready loss curve: large fonts, high contrast, colorblind-safe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h[0] for h in history]
    tr    = [h[1] for h in history]
    va    = [h[2] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, tr, color="#0072B2", lw=3, marker="o", ms=7, label="train loss")
    ax.plot(steps, va, color="#D55E00", lw=3, marker="s", ms=7, label="val loss")
    ax.set_xlabel("training step", fontsize=18)
    ax.set_ylabel("cross-entropy loss (nats/char)", fontsize=18)
    ax.set_title(f"The model learned: loss on '{corpus_name}'", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=17, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


SAMPLE_SETTINGS = [  # the three settings from the class script, verbatim
    dict(temperature=1.0, top_k=None, label="temperature=1.0, no top_k"),
    dict(temperature=0.6, top_k=20,  label="temperature=0.6, top_k=20 (more focused)"),
    dict(temperature=1.3, top_k=10,  label="temperature=1.3, top_k=10 (more varied)"),
]


def write_samples(model, decode, path: Path, settings=SAMPLE_SETTINGS, n_tokens=300):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    lines = []
    for s in settings:
        out = model.generate(context, max_new_tokens=n_tokens,
                             temperature=s["temperature"], top_k=s["top_k"])
        lines.append(f"--- {s['label']} ---\n{decode(out[0].tolist())}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return lines


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=SCRIPT_DIR / "article.txt",
                   help="training text file (default: the class's article.txt)")
    p.add_argument("--from-cpsc", action="store_true",
                   help="build the corpus from data/cpsc_merged.csv narratives instead")
    p.add_argument("--cpsc-max-chars", type=int, default=0,
                   help="cap CPSC corpus size in chars (0 = all)")
    p.add_argument("--tag", type=str, default=None,
                   help="label for output files (default: corpus stem)")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "gpt_run_outputs")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--max-iters", type=int, default=MAX_ITERS)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--eval-iters", type=int, default=EVAL_ITERS)
    p.add_argument("--generate-only", type=Path, default=None, metavar="CKPT",
                   help="skip training; sample from a saved checkpoint")
    p.add_argument("--temperature", type=float, default=None,
                   help="(generate-only) sampling temperature")
    p.add_argument("--top-k", type=int, default=None,
                   help="(generate-only) top-k cutoff")
    args = p.parse_args()

    # ---- generate-only path (take-home exercise 6) --------------------------
    if args.generate_only:
        ckpt = torch.load(args.generate_only, map_location=device)
        cfg, itos = ckpt["config"], ckpt["itos"]
        model = GPTLanguageModel(cfg["vocab_size"], cfg["n_embd"], cfg["n_head"],
                                 cfg["n_layer"], cfg["block_size"], cfg["dropout"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        decode = lambda ids: "".join(itos[i] for i in ids)
        settings = SAMPLE_SETTINGS
        if args.temperature is not None:
            settings = [dict(temperature=args.temperature, top_k=args.top_k,
                             label=f"temperature={args.temperature}, top_k={args.top_k}")]
        for line in write_samples(model, decode,
                                  args.generate_only.with_suffix(".regen.txt"),
                                  settings):
            print(line)
        return

    # ---- corpus --------------------------------------------------------------
    if args.from_cpsc:
        text, corpus_name = build_cpsc_corpus(args.cpsc_max_chars), "CPSC narratives"
    else:
        text, corpus_name = args.corpus.read_text(encoding="utf-8"), args.corpus.name
    tag = args.tag or ("cpsc" if args.from_cpsc else args.corpus.stem)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {corpus_name} — {len(text):,} chars, vocab size {vocab_size}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- train (identical loop, plus history capture) ------------------------
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer,
                             block_size, dropout).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # NOTE for the talk: AdamW's default weight_decay is active here — the class
    # was regularizing without knowing it. Verify the value from the meta file.
    wd = optimizer.defaults.get("weight_decay")

    history, t0 = [], time.time()
    for it in range(args.max_iters):
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model)
            history.append((it, losses["train"], losses["val"]))
            print(f"step {it}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}  [{time.time() - t0:,.0f}s]")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t0

    # ---- artifacts ------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_dir / f"{tag}_losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        w.writerows(history)

    save_loss_figure(history, args.fig_dir / f"gpt_loss_curve_{tag}.png", corpus_name)

    model.eval()
    write_samples(model, decode, args.out_dir / f"{tag}_samples.txt")

    torch.save({
        "model_state": model.state_dict(),
        "itos": itos, "stoi": stoi,
        "config": dict(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head,
                       n_layer=n_layer, block_size=block_size, dropout=dropout),
    }, args.out_dir / f"{tag}_ckpt.pt")

    (args.out_dir / f"{tag}_meta.txt").write_text(
        f"corpus: {corpus_name} ({len(text):,} chars, vocab {vocab_size})\n"
        f"parameters: {n_params:,} ({n_params / 1e6:.2f}M)\n"
        f"iters: {args.max_iters}  batch: {batch_size}  block: {block_size}\n"
        f"optimizer: AdamW lr={learning_rate} weight_decay={wd}\n"
        f"dropout: {dropout}  seed: 1337  device: {device}\n"
        f"final train/val loss: {history[-1][1]:.4f} / {history[-1][2]:.4f}\n"
        f"wall time: {elapsed:,.0f}s\n"
        f"python {sys.version.split()[0]}  torch {torch.__version__}\n",
        encoding="utf-8")

    print(f"\nDone in {elapsed:,.0f}s. Artifacts in {args.out_dir} "
          f"+ figure in {args.fig_dir}")


if __name__ == "__main__":
    main()
