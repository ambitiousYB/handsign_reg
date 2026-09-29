# %% [markdown]
# # Bước 2 — Huấn luyện trên Kaggle GPU (VSL400)
#
# **Cài đặt notebook:**
# - Accelerator: **GPU T4 ×2** (code chỉ dùng 1 GPU). TRÁNH P100: PyTorch bản
#   mới trên Kaggle đã bỏ hỗ trợ P100 -> lỗi "no kernel image is available".
# - Internet: On
# - Input: `vsl-code` (thư mục src/) + `vsl400-landmarks` (output của notebook 01)
#
# Mỗi lần huấn luyện với 150 gloss mất khoảng 20-40 phút.
#
# **Thí nghiệm chính:** hai cách chia tập trên cùng dữ liệu.
# - `provided` — giữ phép chia của tác giả: người ký trong test chưa từng thấy.
#   Đây là con số TRUNG THỰC, đưa lên đầu báo cáo.
# - `random`  — xáo trộn, để cùng một người ký nằm ở cả train và test.
#   Chỉ để chứng minh phép chia sai thổi phồng kết quả đến mức nào.

# %%
import os, sys, shutil, json
from pathlib import Path
import pandas as pd

def find_src(root="/kaggle/input", max_depth=5):
    # Kaggle đặt file của dataset ngay ở gốc (/kaggle/input/<slug>/train.py),
    # không giữ thư mục src/ -> tìm theo file thay vì theo tên thư mục.
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if "sign-language" not in d] if depth < max_depth else []
        if {"train.py", "selftest.py"} <= set(filenames):
            return Path(dirpath)
    return None

SRC = Path("/kaggle/working/src")
if not SRC.exists():
    found = find_src()
    if found is None:
        for p in sorted(Path("/kaggle/input").glob("*/*"))[:50]:
            print(" ", p)
        raise SystemExit("Không tìm thấy code (train.py) trong /kaggle/input. "
                         "Add dataset vsl-code (thư mục src/) vào notebook.")
    print("Dùng code từ", found)
    shutil.copytree(found, SRC, ignore=shutil.ignore_patterns("__pycache__"))
sys.path.insert(0, str(SRC))

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ — bật GPU trong Settings")

# %%
# --- Đưa landmark về đúng chỗ ------------------------------------------------
# Manifest ghi đường dẫn tuyệt đối /kaggle/working/landmarks/..., nên landmark
# PHẢI nằm đúng thư mục đó, không phải ở /kaggle/input.
LM = Path("/kaggle/working/landmarks")

if not (LM / "manifest.csv").exists():
    tar = next(Path("/kaggle/input").rglob("landmarks.tar.gz"), None)
    if tar is not None:
        print("Giải nén", tar)
        !tar -xzf "{tar}" -C /kaggle/working
    else:
        # Kaggle đôi khi tự giải nén khi tạo dataset. Khi đó có TỚI BA file
        # manifest.csv (train/, test/, và file gộp). Phải lấy đúng file gộp:
        # file có cột split chứa CẢ train lẫn test.
        chosen = None
        for m in sorted(Path("/kaggle/input").rglob("manifest.csv"), key=lambda p: len(p.parts)):
            try:
                sp = set(pd.read_csv(m, usecols=["split"])["split"].dropna().astype(str))
            except (ValueError, KeyError):
                continue
            if {"train", "test"} <= sp:
                chosen = m
                break
        if chosen is None:
            raise SystemExit("Không tìm thấy manifest gộp có cả train lẫn test. "
                             "Kiểm tra lại bước merge_manifests ở notebook 01.")
        print("Dùng manifest gộp:", chosen)
        shutil.copytree(chosen.parent, LM, dirs_exist_ok=True)

MANIFEST = str(LM / "manifest.csv")

# %%
# --- Kiểm tra trước khi tốn giờ GPU ------------------------------------------
df = pd.read_csv(MANIFEST, keep_default_na=False)
print("Tổng clip:", len(df), "| Số gloss:", df.label.nunique())
print(df.split.value_counts().to_string())

assert "split" in df.columns, "Manifest mất cột split — xem lại merge_manifests.py"
assert (df.split == "test").sum() > 0, "Không có clip test nào"
assert (df.split == "train").sum() > 0, "Không có clip train nào"

missing = [p for p in df.npz.head(20) if not Path(p).exists()]
assert not missing, f"Đường dẫn npz không tồn tại, vd {missing[0]} — landmark chưa nằm đúng chỗ"
print("\n[ok] Manifest hợp lệ, đường dẫn landmark tồn tại.")

# %% [markdown]
# ## Thí nghiệm chính: provided so với random

# %%
!cd /kaggle/working/src && python train.py \
    --manifest "{MANIFEST}" --split-mode provided \
    --model bilstm --epochs 80 \
    --out /kaggle/working/runs/bilstm_provided

# %%
!cd /kaggle/working/src && python train.py \
    --manifest "{MANIFEST}" --split-mode random \
    --model bilstm --epochs 80 \
    --out /kaggle/working/runs/bilstm_random

# %%
!cd /kaggle/working/src && python evaluate.py --compare \
    /kaggle/working/runs/bilstm_random /kaggle/working/runs/bilstm_provided

# %% [markdown]
# ## Phân tích lỗi trên bản provided
#
# Mở `top_confusions.csv`, chọn 5 cặp nhầm nhiều nhất, rồi tìm video tương ứng
# trong `frame_splited/test/<gloss>/` của dataset gốc để xem tận mắt.

# %%
!cd /kaggle/working/src && python evaluate.py --run /kaggle/working/runs/bilstm_provided

# %% [markdown]
# ## Thí nghiệm phụ: so sánh kiến trúc
#
# Chạy trên tài khoản Kaggle khác nếu làm theo nhóm, để không cạn quota.

# %%
!cd /kaggle/working/src && python train.py \
    --manifest "{MANIFEST}" --split-mode provided \
    --model transformer --epochs 80 --layers 4 --lr 2e-4 \
    --out /kaggle/working/runs/transformer_provided

# %%
rows = []
for f in sorted(Path("/kaggle/working/runs").glob("*/test_metrics.json")):
    m = json.loads(f.read_text())
    rows.append(dict(run=f.parent.name, split=m.get("split_mode"),
                     acc=round(m["acc"], 4), macro_f1=round(m["macro_f1"], 4),
                     top5=round(m["top5"], 4)))
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## Xuất mô hình cho laptop
#
# Chạy ô này sau khi đã chốt mô hình tốt nhất. Đến giai đoạn 4, khi đã có lớp
# `__NOSIGN__` từ clip tự quay, thì huấn luyện lại rồi xuất lại.

# %%
BEST = "/kaggle/working/runs/bilstm_provided/best.pt"
!pip install -q onnx onnxruntime
!cd /kaggle/working/src && python export_onnx.py --ckpt "{BEST}" --out /kaggle/working/models/vsl.onnx
!cp /kaggle/working/runs/bilstm_provided/labels.json /kaggle/working/models/
!ls -lh /kaggle/working/models/

# Tải về laptop: vsl.onnx và vsl.labels.json trong tab Output.
