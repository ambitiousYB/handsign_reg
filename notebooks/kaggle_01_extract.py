# %% [markdown]
# # Bước 1 — Trích xuất landmark từ VSL400 (bản Cropped trên Kaggle)
#
# **Cài đặt notebook:** Accelerator = **None (CPU)**, Internet = **On**.
# Add dataset `nguyenanfms/vsl-vietnamese-sign-language-v2` vào notebook.
#
# Dataset nặng 82 GB nhưng ta **chỉ đọc nhánh `processed/` (~10 GB)** — video
# front-view đã cắt 224×224. Nhánh `raw/` 72 GB là video thô đa góc, không dùng.
# Kaggle mount dataset ở chế độ chỉ đọc nên không tốn hạn mức `/kaggle/working`.
#
# **Điều quan trọng nhất:** dataset đã chia sẵn train/test, trong đó phần VSL400
# chia theo signer ID — người ký trong test chưa từng xuất hiện khi huấn luyện.
# Ta GIỮ NGUYÊN phép chia đó. Tự chia lại là vứt bỏ đúng thứ giá trị nhất.

# %%
!pip install -q mediapipe==0.10.14
# mediapipe 0.10.14 hạ protobuf về 4.25, làm tensorflow 2.20 có sẵn của Kaggle
# import hỏng (ImportError: runtime_version). mediapipe lại thử import tensorflow
# và chỉ bỏ qua khi KHÔNG CÓ tensorflow -> gỡ đi. Notebook này không dùng TF.
!pip uninstall -y -q tensorflow
# Kiểm tra ngay: nếu dòng này lỗi thì dòng CUỐI của traceback là nguyên nhân
!python -c "import numpy, google.protobuf as pb, mediapipe as mp; print('numpy', numpy.__version__, '| protobuf', pb.__version__, '| mediapipe', mp.__version__, mp.solutions.holistic.Holistic)"

# %%
import os, sys, shutil
from pathlib import Path

def find_src(root="/kaggle/input", max_depth=5):
    # Kaggle đặt file của dataset ngay ở gốc (/kaggle/input/<slug>/config.py),
    # không giữ thư mục src/ -> tìm theo file thay vì theo tên thư mục.
    # Bỏ qua dataset video 82 GB để khỏi duyệt hàng chục nghìn file.
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if "sign-language" not in d] if depth < max_depth else []
        if {"extract_dataset.py", "selftest.py"} <= set(filenames):
            return Path(dirpath)
    return None

SRC = Path("/kaggle/working/src")
if not SRC.exists():
    found = find_src()
    if found is None:
        for p in sorted(Path("/kaggle/input").glob("*/*"))[:50]:
            print(" ", p)
        raise SystemExit("Không tìm thấy code (extract_dataset.py) trong /kaggle/input. "
                         "Upload thư mục src/ thành Kaggle Dataset rồi Add Input vào notebook.")
    print("Dùng code từ", found)
    shutil.copytree(found, SRC, ignore=shutil.ignore_patterns("__pycache__"))
sys.path.insert(0, str(SRC))

!cd /kaggle/working/src && python selftest.py

# %%
# --- Định vị nhánh processed --------------------------------------------------
# Đường mount tùy phiên bản Kaggle (/kaggle/input/<slug>/ hoặc
# /kaggle/input/datasets/<owner>/<slug>/), nên tự dò. Dataset có HAI thư mục
# frame_splited: processed/processed/ và processed_augmented/... — chỉ lấy bản
# processed (bản augmented là biến thể của cùng clip, dùng sẽ lệch số liệu).
# Không rglob vì sẽ lục cả nhánh raw/ 72 GB.
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
assert FRAMES is not None, "Không thấy processed/frame_splited trong /kaggle/input — đã Add Input dataset VSL chưa?"
print("FRAMES =", FRAMES)

TRAIN_DIR, TEST_DIR = FRAMES / "train", FRAMES / "test"
for d in (TRAIN_DIR, TEST_DIR):
    n_gloss = len([p for p in d.iterdir() if p.is_dir()])
    n_vid = sum(1 for p in d.rglob("*.mp4"))
    print(f"{d.name:6s}: {n_gloss} gloss, {n_vid} clip")

# %%
# --- Chọn tập con gloss -------------------------------------------------------
# 472 gloss là quá nhiều cho demo real-time và cho tập test câu liên tục.
# 150 gloss nhiều mẫu nhất là điểm cân bằng tốt: đủ phong phú để bài có trọng
# lượng, đủ gọn để trích xuất trong một phiên và để viết câu test.
import collections

TOP_K = 150
cnt = collections.Counter(p.parent.name for p in TRAIN_DIR.rglob("*.mp4"))
top = [g for g, _ in cnt.most_common(TOP_K)]
Path("/kaggle/working/keep_labels.txt").write_text("\n".join(top), encoding="utf-8")

print(f"Giữ {len(top)} gloss, tổng {sum(cnt[g] for g in top)} clip train")
print("Nhiều mẫu nhất :", cnt.most_common(5))
print("Ít mẫu nhất    :", [(g, cnt[g]) for g in top[-5:]])

# Đặt TOP_K = 0 và bỏ cờ --only-labels nếu muốn dùng cả 472 gloss
# (khoảng 26.000 clip, trích xuất mất 2,5-4 giờ — vẫn vừa một phiên Kaggle)

# %%
# --- Trích xuất, tách riêng hai split -----------------------------------------
# --split-tag ghi lại clip thuộc train hay test của dataset gốc.
# --group-regex '' : mỗi file một nhóm, vì đây không phải bộ có bản sao augmentation.
OUT = "/kaggle/working/landmarks"
KEEP = "/kaggle/working/keep_labels.txt"

!cd /kaggle/working/src && python extract_dataset.py \
    --root "{TRAIN_DIR}" --out "{OUT}/train" \
    --workers 4 --model-complexity 1 \
    --group-regex '' --split-tag train --only-labels "{KEEP}"

# %%
!cd /kaggle/working/src && python extract_dataset.py \
    --root "{TEST_DIR}" --out "{OUT}/test" \
    --workers 4 --model-complexity 1 \
    --group-regex '' --split-tag test --only-labels "{KEEP}"

# %%
# --- Gộp thành một manifest ----------------------------------------------------
!cd /kaggle/working/src && python merge_manifests.py \
    --inputs "{OUT}/train/manifest.csv" "{OUT}/test/manifest.csv" \
    --out "{OUT}/manifest.csv"

# %%
import pandas as pd
df = pd.read_csv(f"{OUT}/manifest.csv")

print("Tổng clip :", len(df))
print("Số gloss  :", df.label.nunique())
print("\nPhân bố split:")
print(df.split.value_counts().to_string())
print("\nTỉ lệ thấy tay: trái %.2f, phải %.2f" % (df.lhand_rate.mean(), df.rhand_rate.mean()))

gl_tr = set(df[df.split == "train"].label)
gl_te = set(df[df.split == "test"].label)
miss = gl_tr - gl_te
if miss:
    print(f"\n[!] {len(miss)} gloss có trong train nhưng không có trong test — "
          f"chúng sẽ không được chấm điểm: {sorted(miss)[:5]}")

low = df[df.det_rate < 0.6]
print(f"\nClip phát hiện pose kém (<0.6): {len(low)} — kiểm tra nếu con số này lớn")

# %%
!rm -rf /kaggle/working/src/__pycache__
!cd /kaggle/working && tar -czf landmarks.tar.gz landmarks && ls -lh landmarks.tar.gz

# Xong -> tab Output -> "New Dataset" -> đặt tên "vsl400-landmarks"
