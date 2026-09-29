"""
Cấu hình trung tâm cho pipeline VSL streaming.

Mọi hằng số dùng chung giữa khâu trích xuất offline và khâu inference real-time
phải nằm ở đây. Nếu train và stream dùng hai bộ chỉ số keypoint khác nhau thì mô
hình sẽ chạy nhưng cho kết quả rác — đây là lỗi im lặng nguy hiểm nhất của pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# --------------------------------------------------------------------------
# 1. Chỉ số keypoint của MediaPipe
# --------------------------------------------------------------------------

# BlazePose 33 điểm -> giữ lại phần thân trên liên quan tới ký hiệu.
POSE_IDX = [
    0,   # mũi
    11, 12,  # vai trái, vai phải
    13, 14,  # khuỷu trái, khuỷu phải
    15, 16,  # cổ tay trái, cổ tay phải
    23, 24,  # hông trái, hông phải
]
N_POSE = len(POSE_IDX)          # 9

# Vị trí của vai/hông TRONG mảng đã cắt POSE_IDX (không phải chỉ số gốc).
L_SHOULDER, R_SHOULDER = 1, 2
L_WRIST, R_WRIST = 5, 6
L_HIP, R_HIP = 7, 8

# FaceMesh 468 điểm -> 20 điểm mang thông tin ngữ pháp phi thủ công
# (miệng, mắt, lông mày). Nét mặt trong ngôn ngữ ký hiệu mang nghĩa ngữ pháp
# (nghi vấn, phủ định, mức độ) nên không được bỏ hoàn toàn.
FACE_IDX = [
    61, 291, 0, 17, 13, 14, 78, 308, 40, 270,   # môi
    33, 133, 362, 263, 159, 145, 386, 374,      # mắt
    70, 300,                                     # lông mày
]
N_FACE = len(FACE_IDX)          # 20

N_HAND = 21                     # MediaPipe Hands: chuẩn 21 điểm mỗi bàn tay

# --------------------------------------------------------------------------
# 2. Kích thước vector đặc trưng mỗi frame
# --------------------------------------------------------------------------
# body(9) + lhand(21) + rhand(21) + face(20) = 71 điểm, toạ độ (x, y)
N_BODY_FRAME_PTS = N_POSE + 2 * N_HAND + N_FACE     # 71
DIM_GLOBAL = N_BODY_FRAME_PTS * 2                   # 142  toạ độ chuẩn hoá theo vai
DIM_LOCAL = 2 * N_HAND * 2                          # 84   bàn tay chuẩn hoá cục bộ
DIM_VEL = DIM_GLOBAL                                # 142  vận tốc của phần global
DIM_FLAGS = 2                                       # 2    cờ có/không thấy mỗi bàn tay
FEATURE_DIM = DIM_GLOBAL + DIM_LOCAL + DIM_VEL + DIM_FLAGS   # 370

# Độ dài chuỗi cố định sau khi resample. PHẢI giống nhau giữa train và stream.
SEQ_LEN = 48

# Nhãn dành cho "không phải ký hiệu nào" (frame nghỉ, tay buông, chuyển tiếp).
NOSIGN_LABEL = "__NOSIGN__"

# --------------------------------------------------------------------------
# 3. Tham số streaming
# --------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """Ngưỡng cho máy trạng thái phát hiện ranh giới ký hiệu."""

    # Ngưỡng năng lượng chuyển động, đơn vị = (độ rộng vai) / giây.
    motion_hi: float = 0.55      # vượt ngưỡng này -> bắt đầu một ký hiệu
    motion_lo: float = 0.22      # dưới ngưỡng này -> có thể đã kết thúc
    quiet_frames: int = 6        # số frame yên tĩnh liên tiếp để chốt kết thúc
    min_sign_frames: int = 10    # đoạn ngắn hơn -> coi là nhiễu, bỏ
    max_sign_frames: int = 90    # đoạn dài hơn -> cắt cưỡng bức
    cooldown_frames: int = 8     # thời gian nghỉ sau khi phát ra một gloss

    # Ngưỡng chấp nhận kết quả phân loại.
    conf_threshold: float = 0.60
    margin_threshold: float = 0.15   # p(top1) - p(top2)
    agree_windows: int = 2           # số cửa sổ liên tiếp phải đồng thuận

    # Làm mượt logits theo thời gian.
    ema_alpha: float = 0.6

    target_fps: float = 20.0


@dataclass
class TrainConfig:
    model: str = "bilstm"            # bilstm | transformer
    seq_len: int = SEQ_LEN
    batch_size: int = 64
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    warmup_epochs: int = 3
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.3
    patience: int = 15               # early stopping theo macro-F1 trên val
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 2
    seed: int = 42

    # Augmentation
    aug_rotate_deg: float = 10.0
    aug_scale: float = 0.12
    aug_shift: float = 0.06
    aug_time_warp: float = 0.15
    aug_dropout_hand_p: float = 0.08   # xác suất "mất" một bàn tay cả đoạn
    aug_noise_std: float = 0.008

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class Paths:
    """Đường dẫn mặc định. Trên Kaggle thì override bằng cờ dòng lệnh."""

    raw_videos: Path = Path("data/raw")
    landmarks: Path = Path("data/landmarks")
    manifest: Path = Path("data/manifest.csv")
    runs: Path = Path("runs")
    splits: Path = Path("data/splits")
    extra: dict = field(default_factory=dict)

    def mkdirs(self) -> None:
        for p in (self.raw_videos, self.landmarks, self.runs, self.splits):
            Path(p).mkdir(parents=True, exist_ok=True)


DEFAULT_PATHS = Paths()
