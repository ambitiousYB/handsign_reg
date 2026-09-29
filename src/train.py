"""
Huấn luyện.

Thiết kế riêng cho Kaggle: lưu checkpoint sau MỖI epoch vào /kaggle/working và
tự động chạy tiếp nếu phiên bị ngắt. Phiên Kaggle có thể bị cắt bất ngờ, và mất
6 tiếng huấn luyện vì không lưu checkpoint là chuyện rất hay xảy ra.

Ví dụ:
  python train.py --manifest data/landmarks/manifest.csv \\
      --split-mode group --model bilstm --out runs/bilstm_group
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURE_DIM, NOSIGN_LABEL, TrainConfig
from dataset import SignDataset, build_label_map, class_weights
from splits import check_leakage, make_splits
from models import build_model, count_params


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logit = model(x)
        p = logit.softmax(-1)
        probs.append(p.cpu())
        ps.append(p.argmax(-1).cpu())
        ys.append(y)
    y = torch.cat(ys).numpy()
    pred = torch.cat(ps).numpy()
    prob = torch.cat(probs).numpy()

    acc = float((y == pred).mean())
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    k = min(5, n_classes)
    top5 = float(np.mean([y[i] in np.argsort(-prob[i])[:k] for i in range(len(y))]))
    return dict(acc=acc, macro_f1=macro_f1, top5=top5), y, pred, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--split-mode", default="group",
                    choices=["random", "group", "signer", "provided"])
    ap.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--nosign-ratio", type=float, default=0.12,
                    help="tỉ lệ mẫu 'không phải ký hiệu' sinh thêm khi huấn luyện")
    ap.add_argument("--min-per-class", type=int, default=8,
                    help="loại nhãn có quá ít mẫu")
    ap.add_argument("--top-k-classes", type=int, default=0,
                    help="chỉ giữ K nhãn nhiều mẫu nhất (0 = giữ tất cả). "
                         "472 gloss là quá nhiều cho demo real-time; 100-150 vừa phải.")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(model=args.model, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, hidden=args.hidden, layers=args.layers,
                      dropout=args.dropout, seed=args.seed)
    set_seed(cfg.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Thiết bị: {device}")
    if device.type == "cuda":
        print(f"    {torch.cuda.get_device_name(0)}")

    # ---------------- dữ liệu ----------------
    df = pd.read_csv(args.manifest)
    df["signer_id"] = df.get("signer_id", "").fillna("").astype(str)
    df["group_id"] = df["group_id"].astype(str)

    vc = df["label"].value_counts()
    keep = vc[vc >= args.min_per_class].index
    if args.top_k_classes > 0:
        keep = vc.loc[keep].nlargest(args.top_k_classes).index
        print(f"[i] Giữ {len(keep)} nhãn nhiều mẫu nhất")
    dropped = sorted(set(vc.index) - set(keep))
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    if dropped:
        print(f"[i] Bỏ {len(dropped)} nhãn có dưới {args.min_per_class} mẫu")

    splits = make_splits(df, mode=args.split_mode, seed=cfg.seed)
    key = {"random": "stem", "group": "group_id",
           "signer": "signer_id", "provided": "group_id"}[args.split_mode]
    if args.split_mode not in ("random",):
        check_leakage(df, splits, key)

    use_nosign = args.nosign_ratio > 0
    label_to_idx = build_label_map(df, with_nosign=use_nosign)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    n_classes = len(label_to_idx)
    (out / "labels.json").write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[i] Số lớp: {n_classes} (gồm {NOSIGN_LABEL})" if use_nosign
          else f"[i] Số lớp: {n_classes}")
    for k, v in splits.items():
        print(f"    {k:5s}: {len(v):5d} clip, {df.iloc[v]['label'].nunique()} nhãn")

    cache = not args.no_cache
    ds_tr = SignDataset(df, splits["train"], label_to_idx, cfg, train=True,
                        cache=False, nosign_ratio=args.nosign_ratio)
    ds_va = SignDataset(df, splits["val"], label_to_idx, cfg, train=False, cache=cache)
    ds_te = SignDataset(df, splits["test"], label_to_idx, cfg, train=False, cache=cache)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=(device.type == "cuda"),
                       drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size * 2, shuffle=False,
                       num_workers=args.workers)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size * 2, shuffle=False,
                       num_workers=args.workers)

    # ---------------- mô hình ----------------
    model = build_model(cfg.model, FEATURE_DIM, n_classes, cfg.hidden,
                        cfg.layers, cfg.dropout).to(device)
    print(f"[i] {cfg.model}: {count_params(model)/1e6:.2f}M tham số")

    w = class_weights(df, splits["train"], label_to_idx).to(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    ckpt_path = out / "last.pt"
    best_path = out / "best.pt"
    start_epoch, best_f1, bad = 0, -1.0, 0
    history = []

    if args.resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"] + 1
        best_f1 = st["best_f1"]
        history = st.get("history", [])
        print(f"[i] Chạy tiếp từ epoch {start_epoch} (best macro-F1 = {best_f1:.4f})")

    steps_per_epoch = max(len(dl_tr), 1)
    total_steps = steps_per_epoch * cfg.epochs
    gstep = start_epoch * steps_per_epoch

    # ---------------- vòng lặp ----------------
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0, tot, seen, correct = time.time(), 0.0, 0, 0
        lr = cfg.lr
        for x, y in dl_tr:
            lr = cosine_lr(gstep, total_steps,
                           cfg.warmup_epochs * steps_per_epoch, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logit = model(x)
                loss = crit(logit, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            tot += loss.item() * y.size(0)
            correct += (logit.argmax(-1) == y).sum().item()
            seen += y.size(0)
            gstep += 1

        m_va, *_ = evaluate(model, dl_va, device, n_classes)
        rec = dict(epoch=epoch, loss=tot / max(seen, 1), train_acc=correct / max(seen, 1),
                   lr=lr, **{f"val_{k}": v for k, v in m_va.items()},
                   sec=time.time() - t0)
        history.append(rec)
        print(f"ep {epoch:3d} | loss {rec['loss']:.3f} | train {rec['train_acc']:.3f} "
              f"| val acc {m_va['acc']:.3f} f1 {m_va['macro_f1']:.3f} "
              f"top5 {m_va['top5']:.3f} | {rec['sec']:.0f}s", flush=True)

        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch,
                        best_f1=best_f1, history=history, cfg=vars(args),
                        label_to_idx=label_to_idx), ckpt_path)

        if m_va["macro_f1"] > best_f1:
            best_f1, bad = m_va["macro_f1"], 0
            torch.save(dict(model=model.state_dict(), epoch=epoch,
                            label_to_idx=label_to_idx, cfg=vars(args),
                            feature_dim=FEATURE_DIM, seq_len=cfg.seq_len,
                            arch=cfg.model, hidden=cfg.hidden, layers=cfg.layers,
                            dropout=cfg.dropout), best_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[i] Dừng sớm ở epoch {epoch} (không cải thiện {bad} epoch)")
                break

        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # ---------------- kiểm tra cuối ----------------
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    m_te, y, pred, prob = evaluate(model, dl_te, device, n_classes)
    print("\n" + "=" * 58)
    print(f"TEST ({args.split_mode} split)  acc {m_te['acc']:.4f}  "
          f"macro-F1 {m_te['macro_f1']:.4f}  top5 {m_te['top5']:.4f}")
    print("=" * 58)

    np.savez(out / "test_preds.npz", y=y, pred=pred, prob=prob)
    (out / "test_metrics.json").write_text(
        json.dumps(dict(split_mode=args.split_mode, **m_te), indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    cfg.save(out / "train_config.json")


if __name__ == "__main__":
    main()
