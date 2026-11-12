# %% [markdown]
# # Bước 1b — Trích landmark cho các gloss CÒN LẠI (mở rộng 150 → 472 gloss)
#
# Notebook 01 đã trích 150 gloss nhiều mẫu nhất (dataset `vsl400-landmarks`).
# Notebook này trích phần còn lại, CHIA THÀNH NHIỀU PHẦN (shard) vì làm một lần
# sẽ quá giới hạn 12 giờ của một phiên Kaggle (10.800 clip đã mất ~7,3 giờ).
#
# **Cách chạy:** tạo N_SHARDS bản sao notebook này (hoặc chạy N lần), mỗi bản đặt
# `SHARD = 0, 1, 2...`. Chạy SONG SONG trên tài khoản của các thành viên cho nhanh.
# Dùng **Save Version → Save & Run All** (chạy nền, không sợ mất phiên).
#
# **Cài đặt:** Accelerator = None (CPU), Internet = On.
# **Input:** `vsl-code` (bản MỚI NHẤT), dataset video
# `nguyenanfms/vsl-vietnamese-sign-language-v2`, và `vsl400-landmarks`
# (để biết 150 gloss nào đã trích, tránh làm lại).
#
# **Đầu ra mỗi shard:** `landmarks_part{SHARD}.tar.gz` → tab Output → New Dataset
# tên `vsl-landmarks-part{SHARD}`. Notebook 02 tự gộp tất cả các phần.

# %%
# Phần còn lại nhỏ (~10 clip/gloss, ~3.000 clip train) -> MỘT shard là đủ (~3–4 giờ).
# Muốn nhanh hơn thì N_SHARDS = 2 hoặc 3 và chạy song song nhiều bản.
SHARD = 0          # <-- ĐỔI cho mỗi bản: 0 .. N_SHARDS-1
N_SHARDS = 1       # cả nhóm thống nhất một con số, không đổi giữa các bản

# %%
# Kaggle đã lên Python 3.13, mà mediapipe 0.10.14 (bản còn mp.solutions.holistic)
# chỉ có bản cài tới Python 3.12. Dựng một Python 3.12 RIÊNG bằng uv và chạy các
# script trích xuất bằng nó -> không phụ thuộc Kaggle đổi phiên bản, cũng không
# đụng tensorflow có sẵn. Notebook vẫn chạy bằng Python của Kaggle.
import os
# Jupyter của Kaggle đặt MPLBACKEND = backend inline; các lệnh `!` thừa hưởng biến
# này nhưng Python 3.12 riêng không có gói đó -> matplotlib (mediapipe import nó)
# báo lỗi. Agg = backend không giao diện, đủ cho trích xuất.
os.environ["MPLBACKEND"] = "Agg"
!pip install -q uv
!uv venv -q /tmp/py312 --python 3.12
!uv pip install -q --python /tmp/py312/bin/python mediapipe==0.10.14 pandas
PY = "/tmp/py312/bin/python"
!{PY} -c "import sys, mediapipe as mp; print('Python', sys.version.split()[0], '| mediapipe', mp.__version__, mp.solutions.holistic.Holistic)"

# %%
import os, sys, shutil, tarfile, collections
from pathlib import Path
import pandas as pd

def find_src(root="/kaggle/input", max_depth=5):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if "sign-language" not in d] if depth < max_depth else []
        if {"extract_dataset.py", "selftest.py"} <= set(filenames):
            return Path(dirpath)
    return None

SRC = Path("/kaggle/working/src")
if not SRC.exists():
    found = find_src()
    assert found is not None, "Không thấy code trong /kaggle/input — Add Input dataset vsl-code"
    shutil.copytree(found, SRC, ignore=shutil.ignore_patterns("__pycache__"))
sys.path.insert(0, str(SRC))

# %%
# --- Định vị processed/frame_splited (giống notebook 01) ----------------------
SKIP = {"raw", "processed_augmented", "keypoints_splited", "frame_splited"}

def find_frames(root="/kaggle/input", max_depth=7):
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if "frame_splited" in dirnames and p.name == "processed":
            return p / "frame_splited"
        depth = len(p.relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if d not in SKIP] if depth < max_depth else []
    return None

FRAMES = find_frames()
assert FRAMES is not None, "Không thấy processed/frame_splited — Add Input dataset video VSL"
TRAIN_DIR, TEST_DIR = FRAMES / "train", FRAMES / "test"
print("FRAMES =", FRAMES)

# %%
# --- Gloss đã trích ở notebook 01: đọc từ manifest trong vsl400-landmarks -------
def input_files(match, root="/kaggle/input", max_depth=6):
    """Như rglob nhưng KHÔNG đi vào dataset video 82 GB (duyệt nó mất rất lâu)."""
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if "sign-language" not in d] if depth < max_depth else []
        for f in sorted(filenames):
            if match(f):
                yield Path(dirpath) / f

def done_labels() -> set[str]:
    # Trường hợp Kaggle tự giải nén: có sẵn manifest.csv
    for m in input_files(lambda f: f.startswith("manifest") and f.endswith(".csv")):
        d = pd.read_csv(m, usecols=["label"])
        if len(d):
            return set(d.label.astype(str))
    # Trường hợp còn nguyên tar: chỉ rút file manifest ra, không giải nén cả kho
    for t in input_files(lambda f: f.startswith("landmarks") and f.endswith(".tar.gz")):
        with tarfile.open(t, "r:gz") as tf:
            for mem in tf:
                if mem.name.endswith("landmarks/manifest.csv"):
                    return set(pd.read_csv(tf.extractfile(mem), usecols=["label"]).label.astype(str))
    raise SystemExit("Không đọc được danh sách gloss đã trích — Add Input dataset vsl400-landmarks")

DONE = done_labels()
print(f"Đã trích ở notebook 01: {len(DONE)} gloss")

# %%
# --- Chia phần còn lại thành N_SHARDS phần, cân theo số clip -------------------
# Thuật toán tất định (sắp theo số clip rồi theo tên) -> mọi bản notebook chia
# GIỐNG HỆT nhau, không gloss nào bị trích hai lần hay bị sót.
cnt = collections.Counter(p.parent.name for p in TRAIN_DIR.rglob("*.mp4"))
cnt_te = collections.Counter(p.parent.name for p in TEST_DIR.rglob("*.mp4"))
rest = sorted((g for g in cnt if g not in DONE), key=lambda g: (-(cnt[g] + cnt_te[g]), g))

shards = [[] for _ in range(N_SHARDS)]
load = [0] * N_SHARDS
for g in rest:
    k = load.index(min(load))
    shards[k].append(g)
    load[k] += cnt[g] + cnt_te[g]

print(f"Tổng gloss trong dataset: {len(cnt)} | còn lại: {len(rest)}")
for k in range(N_SHARDS):
    print(f"  shard {k}: {len(shards[k]):3d} gloss, {load[k]:5d} clip"
          + ("   <-- shard này" if k == SHARD else ""))

KEEP = f"/kaggle/working/labels_part{SHARD}.txt"
Path(KEEP).write_text("\n".join(shards[SHARD]), encoding="utf-8")
miss_te = [g for g in shards[SHARD] if cnt_te[g] == 0]
if miss_te:
    print(f"[!] {len(miss_te)} gloss của shard này không có clip test — sẽ không được chấm: {miss_te[:5]}")

# %%
# --- Trích xuất. Cùng thư mục gốc /kaggle/working/landmarks như notebook 01 để
# đường dẫn npz trong manifest khớp khi gộp; manifest đặt tên RIÊNG theo shard.
OUT = "/kaggle/working/landmarks"
!cd /kaggle/working/src && {PY} extract_dataset.py \
    --root "{TRAIN_DIR}" --out "{OUT}/train" --manifest "{OUT}/manifest_train_part{SHARD}.csv" \
    --workers 4 --model-complexity 1 \
    --group-regex '' --split-tag train --only-labels "{KEEP}"

# %%
!cd /kaggle/working/src && {PY} extract_dataset.py \
    --root "{TEST_DIR}" --out "{OUT}/test" --manifest "{OUT}/manifest_test_part{SHARD}.csv" \
    --workers 4 --model-complexity 1 \
    --group-regex '' --split-tag test --only-labels "{KEEP}"

# %%
# `!` không dừng notebook khi lệnh lỗi -> tự kiểm tra, để lỗi hiện ĐÚNG chỗ
for s in ("train", "test"):
    assert Path(f"{OUT}/manifest_{s}_part{SHARD}.csv").exists(), \
        f"Trích xuất {s} thất bại — đọc traceback ở ô trích xuất phía trên"

!cd /kaggle/working/src && {PY} merge_manifests.py \
    --inputs "{OUT}/manifest_train_part{SHARD}.csv" "{OUT}/manifest_test_part{SHARD}.csv" \
    --out "{OUT}/manifest_part{SHARD}.csv"

# %%
df = pd.read_csv(f"{OUT}/manifest_part{SHARD}.csv")
print("Clip:", len(df), "| gloss:", df.label.nunique())
print(df.split.value_counts().to_string())
print("Tỉ lệ thấy tay: trái %.2f, phải %.2f" % (df.lhand_rate.mean(), df.rhand_rate.mean()))
print("Số frame mỗi clip: trung bình %.0f, trung vị %.0f, lớn nhất %d"
      % (df.n_frames.mean(), df.n_frames.median(), df.n_frames.max()))

# %%
!cd /kaggle/working && tar -czf landmarks_part{SHARD}.tar.gz landmarks && ls -lh landmarks_part{SHARD}.tar.gz

# %%
# Kaggle chỉ lấy TỐI ĐA 500 file của output khi tạo dataset. Để nguyên thư mục
# landmarks/ (hàng nghìn .npz) thì dataset nhận 500 file lẻ và BỎ file .tar.gz.
# Kiểm tra tar đọc được rồi mới xoá bản gốc -> output chỉ còn .tar.gz.
import tarfile
tar_path = f"/kaggle/working/landmarks_part{SHARD}.tar.gz"
with tarfile.open(tar_path, "r:gz") as tf:
    n_npz = sum(1 for m in tf if m.name.endswith(".npz"))
# tar có thể NHIỀU hơn manifest: clip pose kém bị loại khỏi manifest nhưng npz vẫn còn
assert n_npz >= len(df), f"tar chỉ có {n_npz} npz nhưng manifest có {len(df)} clip — đừng xoá"
shutil.rmtree("/kaggle/working/landmarks")
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
print(f"[ok] tar có đủ {n_npz} clip; đã xoá bản gốc. Output còn:")
!ls -lh /kaggle/working
# Xong -> tab Output -> New Dataset -> tên "vsl-landmarks-part{SHARD}"
# Kiểm tra trong hộp thoại: phải thấy landmarks_part{SHARD}.tar.gz (~vài trăm MB).
