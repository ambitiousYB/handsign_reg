"""
Chia train/val/test chống rò rỉ dữ liệu.

Tách riêng khỏi dataset.py để không phụ thuộc PyTorch — nhờ vậy bạn kiểm tra được
cách chia tập trên laptop không GPU, không cần cài gì nặng.

Đây là module quan trọng nhất về mặt phương pháp trong cả dự án. Một phép chia sai
sẽ cho ra con số 97% đẹp đẽ và hoàn toàn vô nghĩa, và không có thông báo lỗi nào
cảnh báo bạn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_splits(
    df: pd.DataFrame,
    mode: str = "group",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Trả về chỉ số hàng cho train/val/test.

    mode:
      'random' — chia ngẫu nhiên theo từng clip. CHỈ dùng để chứng minh nó thổi
                 phồng kết quả ra sao. Không bao giờ dùng làm con số chính.
      'group'  — không cho hai biến thể của cùng một clip gốc nằm ở hai phía.
                 Bắt buộc với bộ HF namwu, nơi 127824.mp4 và 127824_1.mp4 là
                 augmentation của cùng một video.
      'signer' — không cho một người biểu diễn xuất hiện ở cả train và test.
                 Thước đo duy nhất phản ánh hệ thống có dùng được cho người lạ
                 hay không. Cần dataset có ID người.
      'provided' — dùng đúng phép chia mà tác giả dataset đã làm sẵn (cột `split`).
                 VSL400 đã chia test theo signer ID, người ký trong test chưa từng
                 xuất hiện khi huấn luyện. Tự chia lại là VỨT BỎ công sức đó và
                 gần như chắc chắn tạo ra rò rỉ. Tập val được tách khỏi train theo
                 nhóm, không đụng vào test.
    """
    rng = np.random.RandomState(seed)
    n = len(df)

    if mode == "provided":
        if "split" not in df.columns or not df["split"].astype(str).str.len().gt(0).any():
            raise ValueError(
                "Manifest không có cột 'split'. Chạy lại extract_dataset.py với "
                "--split-tag train và --split-tag test cho hai thư mục tương ứng, "
                "rồi gộp bằng merge_manifests.py."
            )
        sp = df["split"].fillna("").astype(str).str.strip().str.lower()
        test_idx = np.where(sp == "test")[0]
        tv_idx = np.where(sp != "test")[0]
        if len(test_idx) == 0:
            raise ValueError(
                "Cột 'split' không có dòng nào là 'test' — tập test sẽ rỗng. "
                "Kiểm tra lại lệnh trích xuất thư mục test có cờ --split-tag test chưa."
            )

        key = "group_id" if "group_id" in df.columns else "stem"
        tv_groups = df.iloc[tv_idx][key].astype(str).values
        uniq = np.array(sorted(set(tv_groups)))
        rng.shuffle(uniq)
        n_val_g = max(1, int(len(uniq) * val_frac))
        val_g = set(uniq[:n_val_g])

        val, train = [], []
        for pos, g in zip(tv_idx, tv_groups):
            (val if g in val_g else train).append(pos)
        return dict(train=np.array(train, dtype=np.int64),
                    val=np.array(val, dtype=np.int64),
                    test=test_idx.astype(np.int64))

    if mode == "random":
        idx = rng.permutation(n)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        return dict(test=idx[:n_test],
                    val=idx[n_test:n_test + n_val],
                    train=idx[n_test + n_val:])

    key = "group_id" if mode == "group" else "signer_id"
    if key not in df.columns or not df[key].astype(str).str.len().gt(0).any():
        raise ValueError(
            f"Manifest không có cột '{key}' dùng được. "
            f"mode='signer' cần dataset có ID người biểu diễn "
            f"(VSL400 hoặc Multi-VSL); bộ HF namwu không có."
        )

    groups = df[key].astype(str).values
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)

    n_test_g = max(1, int(len(uniq) * test_frac))
    n_val_g = max(1, int(len(uniq) * val_frac))
    test_g = set(uniq[:n_test_g])
    val_g = set(uniq[n_test_g:n_test_g + n_val_g])

    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, g in enumerate(groups):
        if g in test_g:
            out["test"].append(i)
        elif g in val_g:
            out["val"].append(i)
        else:
            out["train"].append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def check_leakage(df: pd.DataFrame, splits: dict[str, np.ndarray], key: str) -> None:
    """Khẳng định không khoá nào nằm ở hai tập. Gọi sau MỖI lần chia."""
    sets = {k: set(df.iloc[v][key].astype(str)) for k, v in splits.items() if len(v)}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            if inter:
                raise AssertionError(
                    f"RÒ RỈ DỮ LIỆU: {len(inter)} khoá '{key}' có ở cả "
                    f"{names[i]} và {names[j]}. Ví dụ: {list(inter)[:5]}"
                )
    print(f"[ok] Không rò rỉ theo '{key}' giữa các tập.")


def split_report(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, idx in splits.items():
        sub = df.iloc[idx]
        rows.append(dict(
            tap=name, so_clip=len(sub), so_nhan=sub["label"].nunique(),
            so_nhom=sub["group_id"].nunique() if "group_id" in sub else 0,
            so_nguoi=(sub["signer_id"].astype(str).replace("", np.nan).nunique()
                      if "signer_id" in sub else 0),
        ))
    return pd.DataFrame(rows)
