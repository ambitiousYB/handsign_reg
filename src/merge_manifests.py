"""
Gộp nhiều manifest thành một.

Dùng để ghép landmark của bộ dữ liệu công khai với clip bạn tự quay (lớp nghỉ
`__NOSIGN__`, và các ký hiệu bổ sung nếu có).

    python merge_manifests.py \
        --inputs data/landmarks/manifest.csv data/mine/manifest.csv \
        --out data/all_manifest.csv

Script kiểm tra luôn hai thứ dễ sai khi gộp:
  * trùng đường dẫn npz giữa các manifest
  * trùng khoá nhóm giữa các nguồn khác nhau (sẽ gây rò rỉ khi chia tập)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLS = ["npz", "video", "label", "stem", "group_id", "signer_id", "split",
        "n_frames", "fps", "det_rate", "lhand_rate", "rhand_rate"]
TEXT_COLS = ("video", "signer_id", "split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-groups", action="store_true", default=True,
                    help="thêm tiền tố nguồn vào group_id để hai nguồn không đụng khoá")
    args = ap.parse_args()

    frames = []
    for i, path in enumerate(args.inputs):
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Không thấy {p}")
        d = pd.read_csv(p)
        for c in COLS:
            if c not in d.columns:
                d[c] = "" if c in TEXT_COLS else 0
        d["source"] = p.parent.name or f"src{i}"
        if args.prefix_groups:
            d["group_id"] = d["source"].astype(str) + ":" + d["group_id"].astype(str)
        frames.append(d[COLS + ["source"]])
        print(f"  {p}  ->  {len(d)} clip, {d.label.nunique()} nhãn")

    out = pd.concat(frames, ignore_index=True)

    dup = out["npz"].duplicated().sum()
    if dup:
        print(f"[!] {dup} đường dẫn npz bị trùng, giữ bản đầu tiên")
        out = out.drop_duplicates(subset="npz").reset_index(drop=True)

    out["signer_id"] = out["signer_id"].fillna("").astype(str)
    out["split"] = out["split"].fillna("").astype(str)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 54)
    print(f"Đã gộp   : {args.out}")
    print(f"Tổng clip: {len(out)}")
    print(f"Tổng nhãn: {out.label.nunique()}")
    print(f"Tổng nhóm: {out.group_id.nunique()}")
    n_signer = out.loc[out.signer_id.str.len() > 0, "signer_id"].nunique()
    print(f"Có ID người: {n_signer} người trên "
          f"{int((out.signer_id.str.len() > 0).sum())} clip")
    if out["split"].str.len().gt(0).any():
        print("\nSố clip theo split gốc:")
        print(out["split"].replace("", "(không gắn -> vào train)").value_counts().to_string())
    print("\nSố clip theo nguồn:")
    print(out.groupby("source").size().to_string())
    print("\nSố clip theo nhãn (10 ít nhất):")
    print(out.label.value_counts().tail(10).to_string())
    print("=" * 54)


if __name__ == "__main__":
    main()
