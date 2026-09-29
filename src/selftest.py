"""
Kiểm thử nhanh phần lõi bằng dữ liệu giả.

Chạy TRƯỚC khi đụng tới dataset thật. Bắt được lỗi hình dạng tensor và lỗi chuẩn
hoá trong 2 giây, thay vì sau 3 tiếng trích xuất landmark.

    python selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (FEATURE_DIM, SEQ_LEN, N_POSE, N_HAND, N_FACE,
                    DIM_GLOBAL, DIM_LOCAL, StreamConfig)
from landmarks import (RawSequence, featurize, resample, add_velocity,
                       build_window, motion_energy)
from splits import make_splits, check_leakage
from gloss2vi import GlossTranslator
from eval_streaming import edit_ops
from stream import SegmentDetector

PASS, FAIL = "  ok  ", " FAIL "
n_fail = 0


def check(name, cond, extra=""):
    global n_fail
    print(f"[{PASS if cond else FAIL}] {name} {extra}")
    if not cond:
        n_fail += 1


HAND_TEMPLATE = np.array([
    [0.00, 0.00], [-0.02, -0.02], [-0.04, -0.04], [-0.05, -0.06], [-0.06, -0.08],
    [-0.01, -0.06], [-0.015, -0.09], [-0.02, -0.11], [-0.02, -0.13],
    [0.005, -0.06], [0.005, -0.10], [0.005, -0.12], [0.005, -0.14],
    [0.02, -0.06], [0.025, -0.09], [0.03, -0.11], [0.03, -0.13],
    [0.035, -0.05], [0.045, -0.07], [0.05, -0.09], [0.055, -0.10],
], np.float32)


def fake_sequence(T=40, fps=25.0, moving=True, seed=0, noise=0.004) -> RawSequence:
    """Người giả đang cử động tay. Bàn tay dùng khuôn 21 điểm có hình dạng thật,
    vì nếu bàn tay chỉ là nhiễu thì phép chuẩn hoá cục bộ sẽ khuếch đại nhiễu và
    bài kiểm thử trở nên vô nghĩa."""
    rng = np.random.default_rng(seed)
    seq = RawSequence.empty(T, fps=fps)
    t = np.linspace(0, 1, T)
    for i in range(T):
        sx = 0.05 * np.sin(2 * np.pi * t[i]) if moving else 0.0
        seq.pose[i] = np.array([
            [0.50, 0.20],
            [0.62, 0.35], [0.38, 0.35],
            [0.68, 0.50], [0.32, 0.50],
            [0.66 + sx, 0.62], [0.34 - sx, 0.62],
            [0.58, 0.72], [0.42, 0.72],
        ], np.float32)
        seq.pose_vis[i] = 0.9
        seq.pose_ok[i] = 1.0
        grip = 0.8 + 0.2 * np.sin(2 * np.pi * t[i])     # nắm/mở bàn tay
        tmpl = HAND_TEMPLATE * grip
        seq.lhand[i] = np.array([0.66 + sx, 0.62]) + tmpl + rng.normal(0, noise, (N_HAND, 2))
        seq.rhand[i] = np.array([0.34 - sx, 0.62]) + tmpl * [-1, 1] + rng.normal(0, noise, (N_HAND, 2))
        seq.face[i] = np.array([0.50, 0.20]) + rng.normal(0, noise, (N_FACE, 2))
        seq.lhand_ok[i] = 1.0
        seq.rhand_ok[i] = 1.0
    return seq


print("=" * 62)
print("KIỂM THỬ PIPELINE VSL")
print("=" * 62)

# 1. hằng số kích thước
check("FEATURE_DIM khớp với cấu hình",
      FEATURE_DIM == DIM_GLOBAL + DIM_LOCAL + DIM_GLOBAL + 2,
      f"= {FEATURE_DIM}")
check("số điểm keypoint", N_POSE + 2 * N_HAND + N_FACE == DIM_GLOBAL // 2,
      f"{N_POSE}+2x{N_HAND}+{N_FACE} = {DIM_GLOBAL//2}")

# 2. featurize
seq = fake_sequence(noise=0.0)
static = featurize(seq)
check("featurize ra đúng shape", static.shape == (40, DIM_GLOBAL + DIM_LOCAL + 2),
      str(static.shape))
check("featurize không có NaN/Inf", np.isfinite(static).all())

# 3. resample + velocity
r = resample(static, SEQ_LEN)
check("resample về đúng SEQ_LEN", r.shape == (SEQ_LEN, static.shape[1]), str(r.shape))
w = add_velocity(r)
check("vector cuối đúng FEATURE_DIM", w.shape == (SEQ_LEN, FEATURE_DIM), str(w.shape))
check("frame đầu có vận tốc = 0", np.allclose(w[0, -DIM_GLOBAL:], 0.0))

full = build_window(seq, SEQ_LEN)
check("build_window khớp đường đi thủ công", np.allclose(full, w))

# 4. bất biến với vị trí / tỉ lệ  <-- kiểm tra quan trọng nhất
seq2 = fake_sequence(noise=0.0)
seq2.pose = seq2.pose * 0.6 + 0.2          # người nhỏ hơn, đứng lệch
seq2.lhand = seq2.lhand * 0.6 + 0.2
seq2.rhand = seq2.rhand * 0.6 + 0.2
seq2.face = seq2.face * 0.6 + 0.2
f2 = build_window(seq2, SEQ_LEN)
d = float(np.abs(full - f2).max())
check("bất biến với co giãn + tịnh tiến", d < 5e-3, f"sai lệch tối đa {d:.2e}")

# 5. bất biến với fps nguồn
seq3 = fake_sequence(T=80, fps=50.0, noise=0.0)
f3 = build_window(seq3, SEQ_LEN)
d3 = float(np.abs(full - f3).max())
check("bất biến với fps nguồn (40@25 vs 80@50)", d3 < 0.05, f"sai lệch {d3:.3f}")

# 6. mất pose -> forward fill
seq4 = fake_sequence(noise=0.0)
seq4.pose_ok[10:20] = 0.0
seq4.pose[10:20] = 0.0
f4 = build_window(seq4, SEQ_LEN)
check("chịu được mất pose giữa chừng", np.isfinite(f4).all())

# 7. mất bàn tay
seq5 = fake_sequence(noise=0.0)
seq5.lhand_ok[:] = 0.0
f5 = build_window(seq5, SEQ_LEN)
check("mất tay trái -> cờ = 0", np.allclose(f5[:, DIM_GLOBAL + DIM_LOCAL], 0.0))
check("mất tay trái -> toạ độ tay = 0",
      np.allclose(f5[:, 18:18 + 42], 0.0))

# 7b. bàn tay suy biến (hướng thẳng vào camera) phải bị loại, không sinh số rác
seq5b = fake_sequence(noise=0.0)
seq5b.lhand[:] = np.array([0.66, 0.62], np.float32)     # 21 điểm trùng nhau
f5b = build_window(seq5b, SEQ_LEN)
check("bàn tay suy biến -> cờ bị hạ",
      np.allclose(f5b[:, DIM_GLOBAL + DIM_LOCAL], 0.0))
check("bàn tay suy biến -> không có giá trị nổ",
      np.abs(f5b).max() < 8.0, f"max={np.abs(f5b).max():.2f}")

# 8. năng lượng chuyển động
e_move = motion_energy(fake_sequence(moving=True)).mean()
e_still = motion_energy(fake_sequence(moving=False)).mean()
check("chuyển động > đứng yên", e_move > e_still * 3,
      f"{e_move:.3f} vs {e_still:.3f}")
cfg = StreamConfig()
check("ngưỡng motion_hi nằm trong dải hợp lý",
      e_still < cfg.motion_hi, f"nghỉ={e_still:.3f} < hi={cfg.motion_hi}")

# 9. chia tập
df = pd.DataFrame(dict(
    npz=[f"x{i}.npz" for i in range(60)],
    label=[f"L{i%6}" for i in range(60)],
    stem=[f"s{i}" for i in range(60)],
    group_id=[f"g{i//3}" for i in range(60)],
    signer_id=[f"S{i%5}" for i in range(60)],
))
sp = make_splits(df, mode="group", seed=1)
check("chia theo group đủ 3 tập", all(len(sp[k]) > 0 for k in sp),
      {k: len(v) for k, v in sp.items()})
try:
    check_leakage(df, sp, "group_id")
    ok = True
except AssertionError:
    ok = False
check("chia theo group không rò rỉ", ok)

sp2 = make_splits(df, mode="signer", seed=1)
try:
    check_leakage(df, sp2, "signer_id")
    ok2 = True
except AssertionError:
    ok2 = False
check("chia theo signer không rò rỉ", ok2)

sp3 = make_splits(df, mode="random", seed=1)
leaked = len(set(df.iloc[sp3["train"]]["group_id"]) &
             set(df.iloc[sp3["test"]]["group_id"]))
check("chia ngẫu nhiên CÓ rò rỉ (đúng như dự đoán)", leaked > 0,
      f"{leaked} nhóm trùng")

# 10. máy trạng thái
det = SegmentDetector(StreamConfig())
events = []
for e in ([0.05] * 5 + [0.9] * 20 + [0.05] * 12 + [0.05] * 10):
    events.append(det.step(e, True))
check("máy trạng thái phát hiện đúng 1 ranh giới",
      events.count("end") == 1, f"start={events.count('start')}, end={events.count('end')}")

det2 = SegmentDetector(StreamConfig())
ev2 = [det2.step(0.05, True) for _ in range(60)]
check("đứng yên -> không phát ra gì", ev2.count("end") == 0)

# 11. WER
s, d_, i_ = edit_ops(["A", "B", "C"], ["A", "X", "C"])
check("edit_ops đếm đúng thay thế", (s, d_, i_) == (1, 0, 0), f"{(s, d_, i_)}")
s, d_, i_ = edit_ops(["A", "B", "C"], ["A", "C"])
check("edit_ops đếm đúng xoá", (s, d_, i_) == (0, 1, 0), f"{(s, d_, i_)}")
s, d_, i_ = edit_ops(["A", "C"], ["A", "B", "C"])
check("edit_ops đếm đúng chèn", (s, d_, i_) == (0, 0, 1), f"{(s, d_, i_)}")

# 12. gloss -> tiếng Việt
tr = GlossTranslator()
check("thời gian lên đầu + thì quá khứ",
      tr.translate(["HÔM-QUA", "TÔI", "ĐI", "CHỢ"]) == "Hôm qua tôi đã đi chợ.")
check("phủ định cuối -> trước động từ",
      tr.translate(["TÔI", "ĐI", "KHÔNG"]) == "Tôi không đi.")
check("câu hỏi có dấu ?", tr.translate(["BẠN", "ĂN", "GÌ"]).endswith("?"))
check("chèn hệ từ 'là'", "là" in tr.translate(["TÔI", "HỌC-SINH"]))
check("gloss lạ vẫn xử lý được",
      tr.translate(["XYZ-LẠ"]) == "Xyz lạ.", tr.translate(["XYZ-LẠ"]))

print("=" * 62)
print(f"{'TẤT CẢ ĐỀU ĐẠT' if n_fail == 0 else f'{n_fail} MỤC KHÔNG ĐẠT'}")
print("=" * 62)
sys.exit(1 if n_fail else 0)
