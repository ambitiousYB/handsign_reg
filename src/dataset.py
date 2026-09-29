"""
Dataset PyTorch và augmentation.

Logic chia tập nằm ở splits.py (không phụ thuộc torch). Module này re-export
make_splits/check_leakage cho tiện import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEQ_LEN, FEATURE_DIM, DIM_GLOBAL, DIM_LOCAL, NOSIGN_LABEL, TrainConfig
from landmarks import RawSequence, featurize, resample, add_velocity
from splits import make_splits, check_leakage, split_report  # noqa: F401  (re-export)


# ==========================================================================
# 2. Augmentation
# ==========================================================================

def _affine_xy(block: np.ndarray, n_pts: int, rot: float, scale: float,
               shift: np.ndarray | None) -> np.ndarray:
    """Áp phép xoay/co giãn/tịnh tiến lên khối toạ độ phẳng (T, n_pts*2)."""
    T = block.shape[0]
    pts = block.reshape(T, n_pts, 2)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]], np.float32)
    pts = pts @ R.T * scale
    if shift is not None:
        pts = pts + shift[None, None, :]
    return pts.reshape(T, n_pts * 2)


class Augment:
    """Augmentation áp dụng SAU chuẩn hoá.

    Lưu ý quan trọng: vì ta đã chuẩn hoá theo tâm vai, tỉ lệ vai và góc vai, nên
    xoay/co giãn ảnh gốc là vô nghĩa — phép chuẩn hoá sẽ triệt tiêu chúng. Do đó
    augment hình học phải tác động lên toạ độ ĐÃ chuẩn hoá; ý nghĩa của nó là mô
    phỏng sai số của chính bước chuẩn hoá.
    """

    def __init__(self, cfg: TrainConfig, mirror: bool = False):
        self.cfg = cfg
        self.mirror = mirror

    def temporal(self, seq: RawSequence, rng: np.random.Generator) -> RawSequence:
        T = len(seq)
        if T < 8:
            return seq
        w = self.cfg.aug_time_warp
        lo = int(T * (1.0 - w) * rng.uniform(0.0, 1.0) * 0.5)
        hi = T - int(T * w * rng.uniform(0.0, 1.0) * 0.5)
        hi = max(lo + 6, min(hi, T))
        return seq.slice(lo, hi)

    def hand_dropout(self, seq: RawSequence, rng: np.random.Generator) -> RawSequence:
        """Mô phỏng việc MediaPipe mất dấu một bàn tay — chuyện xảy ra liên tục
        khi tay bị che hoặc ra khỏi khung hình."""
        p = self.cfg.aug_dropout_hand_p
        if rng.random() < p:
            seq.lhand_ok = seq.lhand_ok * 0.0
        if rng.random() < p:
            seq.rhand_ok = seq.rhand_ok * 0.0
        # rơi rớt lẻ tẻ theo frame
        T = len(seq)
        if T and rng.random() < 0.3:
            m = rng.random(T) > 0.05
            seq.lhand_ok = seq.lhand_ok * m
            seq.rhand_ok = seq.rhand_ok * (rng.random(T) > 0.05)
        return seq

    def spatial(self, static: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        rot = np.deg2rad(rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg))
        sc = 1.0 + rng.uniform(-cfg.aug_scale, cfg.aug_scale)
        sh = rng.uniform(-cfg.aug_shift, cfg.aug_shift, size=2).astype(np.float32)

        g = static[:, :DIM_GLOBAL]
        l = static[:, DIM_GLOBAL:DIM_GLOBAL + DIM_LOCAL]
        rest = static[:, DIM_GLOBAL + DIM_LOCAL:]

        g = _affine_xy(g, DIM_GLOBAL // 2, rot, sc, sh)
        l = _affine_xy(l, DIM_LOCAL // 2, rot, sc, None)  # bàn tay không tịnh tiến

        out = np.concatenate([g, l, rest], axis=1)
        out += rng.normal(0, cfg.aug_noise_std, out.shape).astype(np.float32)
        return out.astype(np.float32)


# ==========================================================================
# 3. Dataset
# ==========================================================================

class SignDataset(Dataset):
    """Đọc manifest -> tensor (T, FEATURE_DIM).

    cache=True nạp trước toàn bộ đặc trưng tĩnh vào RAM dạng float16. Với ~4.000
    clip thì tốn khoảng 4.000 × 48 × 228 × 2 byte ≈ 87 MB — thoải mái trên Kaggle
    và làm mỗi epoch nhanh hơn nhiều lần so với đọc npz liên tục.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: Iterable[int],
        label_to_idx: dict[str, int],
        cfg: TrainConfig,
        train: bool = False,
        cache: bool = True,
        mirror: bool = False,
        nosign_ratio: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.label_to_idx = label_to_idx
        self.cfg = cfg
        self.train = train
        self.aug = Augment(cfg, mirror=mirror) if train else None
        self.nosign_ratio = nosign_ratio if train else 0.0
        self.nosign_idx = label_to_idx.get(NOSIGN_LABEL, -1)

        self.cache: dict[int, np.ndarray] | None = {} if cache else None
        if cache:
            self._warm()

    def _warm(self) -> None:
        for k, i in enumerate(self.indices):
            seq = RawSequence.from_npz(self.df.iloc[i]["npz"])
            self.cache[int(i)] = featurize(seq).astype(np.float16)
            if (k + 1) % 500 == 0:
                print(f"  nạp cache {k+1}/{len(self.indices)}", flush=True)

    def __len__(self) -> int:
        extra = int(len(self.indices) * self.nosign_ratio)
        return len(self.indices) + extra

    def _load_static(self, row_i: int, rng: np.random.Generator) -> np.ndarray:
        if self.cache is not None and not self.train:
            return self.cache[row_i].astype(np.float32)

        seq = RawSequence.from_npz(self.df.iloc[row_i]["npz"])
        if self.aug is not None:
            seq = self.aug.temporal(seq, rng)
            seq = self.aug.hand_dropout(seq, rng)
        return featurize(seq)

    def _make_nosign(self, rng: np.random.Generator) -> np.ndarray:
        """Sinh mẫu 'không phải ký hiệu' từ các frame đầu/cuối của clip bất kỳ.

        Ở đầu và cuối mỗi clip ký hiệu, người biểu diễn thường đang ở tư thế nghỉ
        (tay buông hoặc chưa vào vị trí). Lấy vài frame đó rồi kéo dài ra chính là
        mô phỏng trạng thái rảnh mà webcam sẽ thấy phần lớn thời gian.

        Không có lớp này, hệ thống streaming sẽ phun ra gloss liên tục ngay cả khi
        người dùng chỉ đang ngồi yên.
        """
        i = int(rng.choice(self.indices))
        seq = RawSequence.from_npz(self.df.iloc[i]["npz"])
        T = len(seq)
        if T < 4:
            return np.zeros((SEQ_LEN, FEATURE_DIM - DIM_GLOBAL), np.float32)
        k = max(2, T // 8)
        seq = seq.slice(0, k) if rng.random() < 0.5 else seq.slice(T - k, T)
        st = featurize(seq)
        st = resample(st, self.cfg.seq_len)
        # làm cho gần như đứng yên: lấy frame đầu, thêm rung nhẹ
        st = np.repeat(st[:1], self.cfg.seq_len, axis=0)
        st = st + rng.normal(0, 0.004, st.shape).astype(np.float32)
        return st.astype(np.float32)

    def __getitem__(self, i: int):
        rng = np.random.default_rng(
            None if self.train else (12345 + i)
        )
        n_real = len(self.indices)

        if i >= n_real:
            static = self._make_nosign(rng)
            y = self.nosign_idx
        else:
            row_i = int(self.indices[i])
            static = self._load_static(row_i, rng)
            static = resample(static, self.cfg.seq_len)
            if self.aug is not None:
                static = self.aug.spatial(static, rng)
            y = self.label_to_idx[self.df.iloc[row_i]["label"]]

        x = add_velocity(static)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)


def build_label_map(df: pd.DataFrame, with_nosign: bool = True) -> dict[str, int]:
    labels = sorted(df["label"].unique().tolist())
    if with_nosign and NOSIGN_LABEL not in labels:
        labels.append(NOSIGN_LABEL)
    return {l: i for i, l in enumerate(labels)}


def class_weights(df: pd.DataFrame, indices: np.ndarray,
                  label_to_idx: dict[str, int]) -> torch.Tensor:
    counts = np.zeros(len(label_to_idx), np.float64)
    for i in indices:
        counts[label_to_idx[df.iloc[int(i)]["label"]]] += 1
    counts[counts == 0] = counts[counts > 0].mean() if (counts > 0).any() else 1.0
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(np.clip(w, 0.2, 5.0), dtype=torch.float32)
