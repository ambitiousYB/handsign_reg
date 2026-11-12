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
#
# Hỗ trợ NHIỀU phần landmark: `vsl400-landmarks` (150 gloss, notebook 01) cộng
# các `vsl-landmarks-part*` (gloss còn lại, notebook 01b). Add Input tất cả các
# phần muốn dùng; chỉ add `vsl400-landmarks` thì vẫn chạy như cũ với 150 gloss.
LM = Path("/kaggle/working/landmarks")

def merged_manifests(d: Path) -> list[Path]:
    """Manifest đã gộp (có CẢ train lẫn test) trong thư mục d. Bỏ qua manifest
    riêng của train/ hoặc test/ — dùng chúng sẽ đếm trùng hoặc mất một split."""
    out = []
    # Chỉ nhận manifest của dataset landmark (notebook 01: manifest.csv, 01b:
    # manifest_part*.csv). manifest_all.csv là sản phẩm của chính notebook này —
    # nếu output cũ được add làm input, nhận nó sẽ gộp trùng dữ liệu.
    for m in sorted(d.glob("manifest.csv")) + sorted(d.glob("manifest_part*.csv")):
        try:
            sp = set(pd.read_csv(m, usecols=["split"])["split"].dropna().astype(str))
        except (ValueError, KeyError):
            continue
        if {"train", "test"} <= sp:
            out.append(m)
    return out

tars = sorted(Path("/kaggle/input").rglob("landmarks*.tar.gz"))
for t in tars:
    marker = Path(f"/kaggle/working/.done_{t.name}")
    if not marker.exists():
        print("Giải nén", t)
        !tar -xzf "{t}" -C /kaggle/working
        marker.touch()

# Phần nào Kaggle đã tự giải nén (thường gặp khi upload tar thủ công) thì nằm
# sẵn dạng thư mục -> chép về LM. Làm CẢ KHI có tar khác: các phần có thể lẫn
# hai dạng. Bỏ qua phần đã có trong LM (cùng tên manifest) để không chép trùng.
have = {p.name for p in merged_manifests(LM)} if LM.exists() else set()
for d in sorted({m.parent for m in Path("/kaggle/input").rglob("manifest*.csv")}):
    if (d / "runs").exists() or (d.parent / "runs").exists():
        continue   # output của notebook 02 cũ (add vào để dùng lại run), không phải dataset
    new = [p.name for p in merged_manifests(d) if p.name not in have]
    if new:
        print("Chép", d, "->", new)
        shutil.copytree(d, LM, dirs_exist_ok=True)
        have |= set(new)

parts = merged_manifests(LM)
if not parts:
    raise SystemExit("Không tìm thấy manifest gộp có cả train lẫn test. "
                     "Kiểm tra bước merge_manifests ở notebook 01 / 01b.")
print("Các phần landmark:", [p.name for p in parts])

if len(parts) == 1:
    MANIFEST = str(parts[0])
else:
    MANIFEST = "/kaggle/working/manifest_all.csv"
    part_args = " ".join(f'"{p}"' for p in parts)
    !cd /kaggle/working/src && python merge_manifests.py --inputs {part_args} --out "{MANIFEST}"

# %%
# --- Kiểm tra trước khi tốn giờ GPU ------------------------------------------
df = pd.read_csv(MANIFEST, keep_default_na=False)
print("Tổng clip:", len(df), "| Số gloss:", df.label.nunique())
print(df.split.value_counts().to_string())

assert "split" in df.columns, "Manifest mất cột split — xem lại merge_manifests.py"
assert (df.split == "test").sum() > 0, "Không có clip test nào"
assert (df.split == "train").sum() > 0, "Không có clip train nào"

missing = [p for p in df.npz.sample(min(300, len(df)), random_state=0) if not Path(p).exists()]
assert not missing, f"Đường dẫn npz không tồn tại, vd {missing[0]} — landmark chưa nằm đúng chỗ"
print("\n[ok] Manifest hợp lệ, đường dẫn landmark tồn tại.")

# %%
# --- Dùng lại run đã huấn luyện xong (khỏi tốn lại hàng giờ GPU) ---------------
# Nếu lần chạy trước dừng giữa chừng: Add Input = OUTPUT của version đó (tab Data
# → Add Input → Your Work → notebook này). Run nào đã có test_metrics.json sẽ được
# chép về và ô huấn luyện tương ứng tự BỎ QUA. Lưu ý: chỉ dùng lại run cùng bộ
# gloss — run 150 gloss cũ KHÔNG được trộn với dữ liệu 472 gloss.
RUNS = Path("/kaggle/working/runs")
n_gloss_now = df.label.nunique()
for f in sorted(Path("/kaggle/input").rglob("runs/*/test_metrics.json")):
    run = f.parent
    dst = RUNS / run.name
    if dst.exists():
        continue
    try:
        n_prev = len(json.loads((run / "labels.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        print(f"  bỏ qua {run.name}: thiếu labels.json")
        continue
    if abs(n_prev - n_gloss_now) > 30:   # labels.json có thêm __NOSIGN__, vài gloss bị lọc
        print(f"  bỏ qua {run.name}: {n_prev} lớp, khác bộ dữ liệu hiện tại ({n_gloss_now} gloss)")
        continue
    shutil.copytree(run, dst)
    print(f"  dùng lại {run.name} ({n_prev} lớp)")

def done(name: str) -> bool:
    ok = (RUNS / name / "test_metrics.json").exists()
    if ok:
        print(f"[bỏ qua] {name} đã có kết quả")
    return ok

# %% [markdown]
# ## Thí nghiệm chính: provided so với random

# %%
if not done("bilstm_provided"):
    !cd /kaggle/working/src && python train.py --manifest "{MANIFEST}" --split-mode provided --model bilstm --epochs 80 --out {RUNS}/bilstm_provided

# %%
if not done("bilstm_random"):
    !cd /kaggle/working/src && python train.py --manifest "{MANIFEST}" --split-mode random --model bilstm --epochs 80 --out {RUNS}/bilstm_random

# %%
!cd /kaggle/working/src && python evaluate.py --compare \
    /kaggle/working/runs/bilstm_random /kaggle/working/runs/bilstm_provided

# %%
# --- Chẩn đoán: vì sao random KHÔNG cao hơn provided? --------------------------
# Kết quả thực tế: random 84.9% < provided 87.5%, ngược với dự đoán "chia ngẫu
# nhiên thổi phồng". Giả thuyết: tập test gốc của dataset dễ hơn phần train gốc
# (vd phần train lẫn video internet). Tách accuracy của run random theo nguồn gốc
# clip. Ô này chỉ để phân tích: lỗi gì cũng IN CẢNH BÁO, không dừng notebook.
import numpy as np
from splits import make_splits
try:
    d = np.load(RUNS / "bilstm_random" / "test_preds.npz")
    if "orig_split" in d.files:
        origin = d["orig_split"]          # train.py mới lưu sẵn, không cần dựng lại
    else:
        # Run cũ: dựng lại phép chia, lặp đúng bước lọc nhãn < 8 mẫu của train.py
        df_all = pd.read_csv(MANIFEST)
        vc = df_all["label"].value_counts()
        df_all = df_all[df_all["label"].isin(vc[vc >= 8].index)].reset_index(drop=True)
        sp = make_splits(df_all, mode="random", seed=42)
        assert len(d["y"]) == len(sp["test"]), "không dựng lại được tập test random"
        origin = df_all.iloc[sp["test"]]["split"].astype(str).values
    ok = d["y"] == d["pred"]
    for o in ("train", "test"):
        m = origin == o
        print(f"Run random — clip gốc thuộc '{o:5s}': {m.sum():5d} clip, acc {ok[m].mean():.3f}")
except Exception as e:
    print(f"[!] Bỏ qua chẩn đoán: {e}")
# Nếu acc trên clip gốc 'test' cao hơn hẳn clip gốc 'train' -> tập test gốc dễ
# hơn, và so sánh random/provided phải diễn giải lại trong báo cáo.

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
if not done("transformer_provided"):
    !cd /kaggle/working/src && python train.py --manifest "{MANIFEST}" --split-mode provided --model transformer --epochs 80 --layers 4 --lr 2e-4 --out {RUNS}/transformer_provided

# %% [markdown]
# ## Nâng độ chính xác
#
# 1. **Lật gương** (`--mirror-p 0.5`): mô phỏng người ký thuận tay trái / webcam
#    lật ảnh. So với `transformer_provided` để biết có lợi thật không — giữ lại
#    run nào tốt hơn, cả hai đều là một dòng trong bảng ablation.
# 2. **Ensemble**: lấy trung bình xác suất của nhiều mô hình đã huấn luyện. Không
#    tốn thêm giờ GPU, thường tăng 1–2 điểm. Các bài báo SLR top đầu đều ensemble.

# %%
if not done("transformer_mirror"):
    !cd /kaggle/working/src && python train.py --manifest "{MANIFEST}" --split-mode provided --model transformer --epochs 80 --layers 4 --lr 2e-4 --mirror-p 0.5 --out {RUNS}/transformer_mirror

# %%
# --- Ensemble: trung bình xác suất trên tập test -------------------------------
# Hợp lệ vì mọi run 'provided' dùng cùng manifest -> cùng tập test, cùng thứ tự,
# cùng bảng nhãn (kiểm tra bên dưới). KHÔNG trộn run 'random' vào đây.
import numpy as np
from sklearn.metrics import f1_score
RUNS = Path("/kaggle/working/runs")
members = [r for r in ("bilstm_provided", "transformer_provided", "transformer_mirror")
           if (RUNS / r / "test_preds.npz").exists()]
ref_labels = json.loads((RUNS / members[0] / "labels.json").read_text(encoding="utf-8"))
P, y = 0.0, None
for r in members:
    assert json.loads((RUNS / r / "labels.json").read_text(encoding="utf-8")) == ref_labels, \
        f"{r} có bảng nhãn khác — không ensemble được"
    d = np.load(RUNS / r / "test_preds.npz")
    assert y is None or np.array_equal(d["y"], y), f"{r} có tập test khác thứ tự"
    y = d["y"]
    P = P + d["prob"]
    print(f"  {r:22s} acc {(d['prob'].argmax(1) == y).mean():.4f}")
pred = P.argmax(1)
print(f"ENSEMBLE {'+'.join(members)}")
print(f"  acc {(pred == y).mean():.4f}  macro-F1 "
      f"{f1_score(y, pred, average='macro', labels=np.unique(y)):.4f}  "
      f"top-5 {np.mean([y[i] in np.argsort(-P[i])[:5] for i in range(len(y))]):.4f}")

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
# Tự chọn run 'provided' có macro-F1 cao nhất (KHÔNG chọn run 'random')
cands = []
for f in Path("/kaggle/working/runs").glob("*/test_metrics.json"):
    m = json.loads(f.read_text())
    if m.get("split_mode") == "provided":
        cands.append((m["macro_f1"], f.parent.name))
best_f1, best_run = max(cands)
print(f"Xuất mô hình tốt nhất: {best_run} (macro-F1 {best_f1:.4f})")
BEST = f"/kaggle/working/runs/{best_run}/best.pt"
!pip install -q onnx onnxruntime
!cd /kaggle/working/src && python export_onnx.py --ckpt "{BEST}" --out /kaggle/working/models/vsl.onnx
!ls -lh /kaggle/working/models/
# Phải thấy CẢ HAI file vsl.onnx và vsl.labels.json. Thiếu vsl.onnx -> đọc lỗi ngay trên.

# %%
# --- Dọn output: chỉ giữ runs/ và models/ ---------------------------------------
# Landmark đã giải nén (hàng chục nghìn file) đã có trong các dataset landmark;
# để lại trong output thì output nặng, khó tải, và khi add output làm input cho
# lần sau thì danh sách bị landmark lấp hết.
for p in ("landmarks", "src"):
    shutil.rmtree(f"/kaggle/working/{p}", ignore_errors=True)
for p in Path("/kaggle/working").glob("*"):
    if p.is_file() and (p.name.startswith(".done_") or p.name == "manifest_all.csv"):
        p.unlink()
!ls -la /kaggle/working
# Tải về laptop: models/vsl.onnx + models/vsl.labels.json (dùng Download All dạng
# .zip để file .pt không bị trình duyệt tự giải nén thành thư mục).
