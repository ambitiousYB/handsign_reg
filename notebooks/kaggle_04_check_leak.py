# %% [markdown]
# # Kiểm tra trùng lặp train/test (rò rỉ dữ liệu)
#
# **Vì sao cần:** kết quả 451 gloss có một điểm bất thường —
# - 150 gloss cũ (≥ 62 clip/gloss): Transformer đúng **89,3%**
# - ~300 gloss mới (~10 clip train/gloss): Transformer đúng **94,8%**
#
# Ít dữ liệu hơn 6 lần mà lại đúng hơn là dấu hiệu tập test chứa video TRÙNG hoặc
# GẦN TRÙNG tập train (cùng video cắt/nén lại, cùng người cùng buổi quay). Nếu
# đúng vậy thì 94,8% là con số ảo và không được đưa vào báo cáo như kết quả
# "người ký chưa thấy".
#
# **Cài đặt:** Accelerator = None (CPU). Input: `vsl-code`, `vsl400-landmarks`,
# và dataset landmark phần mới (vd `vsl400-landmarks-part2`). Chạy ~5–10 phút.

# %%
import os, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

def find_src(root="/kaggle/input", max_depth=5):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if "sign-language" not in d] if depth < max_depth else []
        if {"train.py", "selftest.py"} <= set(filenames):
            return Path(dirpath)

SRC = Path("/kaggle/working/src")
if not SRC.exists():
    shutil.copytree(find_src(), SRC, ignore=shutil.ignore_patterns("__pycache__"))
sys.path.insert(0, str(SRC))
from config import DIM_GLOBAL
from landmarks import RawSequence, featurize, resample

# %%
# --- Nạp landmark (giống notebook 02) -----------------------------------------
LM = Path("/kaggle/working/landmarks")
for t in sorted(Path("/kaggle/input").rglob("landmarks*.tar.gz")):
    print("Giải nén", t)
    !tar -xzf "{t}" -C /kaggle/working
for m in sorted(Path("/kaggle/input").rglob("manifest_part*.csv")) + sorted(Path("/kaggle/input").rglob("manifest.csv")):
    d = m.parent
    if (d / "runs").exists() or (d.parent / "runs").exists() or not (d / "train").exists():
        continue
    if not (LM / m.name).exists():
        print("Chép", d)
        shutil.copytree(d, LM, dirs_exist_ok=True)

parts = [m for m in [LM / "manifest.csv"] + sorted(LM.glob("manifest_part*.csv")) if m.exists()]
df = pd.concat([pd.read_csv(m) for m in parts], ignore_index=True).drop_duplicates("npz")
OLD = set(pd.read_csv(LM / "manifest.csv").label)     # 150 gloss của notebook 01
df["nhom"] = np.where(df.label.isin(OLD), "150 gloss cũ", "gloss mới")
print(df.groupby(["nhom", "split"]).size().unstack())

# %%
# --- Kiểm tra 1: cùng mã video (stem) xuất hiện ở cả train và test ------------
both = set(df[df.split == "train"].stem.astype(str)) & set(df[df.split == "test"].stem.astype(str))
print(f"Mã video có ở CẢ train lẫn test: {len(both)}", sorted(both)[:10])

# %%
# --- Kiểm tra 2: clip test gần như trùng một clip train CÙNG gloss -------------
# So đặc trưng tư thế đã chuẩn hoá (theo vai) sau khi đưa về 48 frame. Hai video
# của hai người khác nhau lệch nhau cỡ vài phần mười; cùng một video bị cắt/nén
# lại thì lệch rất nhỏ.
def vec(p):
    return resample(featurize(RawSequence.from_npz(p)), 48)[:, :DIM_GLOBAL]

rows = []
for lab, g in df.groupby("label"):
    tr, te = g[g.split == "train"], g[g.split == "test"]
    if len(tr) == 0 or len(te) == 0:
        continue
    Vtr = np.stack([vec(p) for p in tr.npz])
    for p in te.npz:
        dist = np.abs(Vtr - vec(p)[None]).mean(axis=(1, 2))
        rows.append((lab, g.nhom.iloc[0], float(dist.min())))
r = pd.DataFrame(rows, columns=["label", "nhom", "dmin"])

print("Khoảng cách tới clip train GẦN NHẤT cùng gloss (nhỏ = giống):")
print(r.groupby("nhom").dmin.describe(percentiles=[.1, .25, .5]).round(3))
for thr in (0.02, 0.05, 0.1):
    print(f"Tỉ lệ clip test có bản 'gần trùng' trong train (dmin < {thr}):")
    print("   " + r.groupby("nhom").dmin.apply(lambda x: f"{(x < thr).mean():.1%}").to_string().replace("\n", "\n   "))

# %% [markdown]
# **Đọc kết quả:**
# - Nếu "gloss mới" có tỉ lệ gần trùng cao hơn hẳn "150 gloss cũ" (vd 40% so với
#   2%), hoặc trung vị dmin nhỏ hơn hẳn → tập test của gloss mới rò rỉ, con số
#   94,8% là ẢO. Báo cáo chỉ dùng số trên 150 gloss cũ cho "người ký chưa thấy",
#   và nêu thẳng phát hiện này (đây là một phát hiện có giá trị).
# - Nếu hai nhóm tương đương → không có bằng chứng rò rỉ; gloss mới dễ hơn vì
#   lý do khác (vd ít gloss giống nhau hơn).
