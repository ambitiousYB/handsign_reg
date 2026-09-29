"""
Trích xuất landmark hàng loạt từ thư mục video.

Cấu trúc đầu vào mong đợi (label-directory):
    <root>/<tên_nhãn>/<video>.mp4

Đầu ra:
    <out>/<tên_nhãn>/<video>.npz      landmark thô
    <out>/manifest.csv                bảng tra cứu cho DataLoader

Chạy trên Kaggle:
    python extract_dataset.py \
        --root /kaggle/input/visignlanguage-video/train \
        --out  /kaggle/working/landmarks \
        --workers 4 --group-regex '^(\\d+)'

Tính năng quan trọng: CÓ THỂ CHẠY TIẾP. Nếu phiên Kaggle bị ngắt giữa chừng,
chạy lại đúng lệnh trên, những clip đã xong sẽ được bỏ qua.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import HolisticExtractor, frames_to_raw, RawSequence  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI"}

_EXTRACTOR: HolisticExtractor | None = None
_ARGS: dict = {}


def _init_worker(model_complexity: int, opts: dict):
    """Mỗi tiến trình con giữ một instance MediaPipe riêng.

    MediaPipe không an toàn khi chia sẻ qua fork, nên KHÔNG tạo extractor ở tiến
    trình cha rồi truyền xuống — sẽ treo hoặc crash im lặng.
    """
    global _EXTRACTOR, _ARGS
    _ARGS = dict(opts)
    _EXTRACTOR = HolisticExtractor(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _probe_worker(model_complexity: int):
    HolisticExtractor(static_image_mode=False, model_complexity=model_complexity).close()


def check_mediapipe(model_complexity: int) -> None:
    """Thử khởi tạo MediaPipe trong MỘT tiến trình con trước khi mở Pool.

    Nếu initializer của Pool lỗi, Pool sẽ tạo lại worker mãi mãi và in cùng
    một traceback vô hạn lần. Kiểm tra trước để lỗi hiện đúng một lần.
    """
    from multiprocessing import Process
    p = Process(target=_probe_worker, args=(model_complexity,))
    p.start()
    p.join()
    if p.exitcode != 0:
        print("\n[!] Không khởi tạo được MediaPipe (traceback ở ngay trên). "
              "Dòng cuối của traceback là nguyên nhân thật.", file=sys.stderr)
        sys.exit(1)


def read_video(path: str, max_frames: int = 300, stride: int = 1):
    """Đọc video -> (danh sách frame RGB, fps). Tự giảm mẫu nếu clip quá dài."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0

    frames = []
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps / stride


def _process_one(task: tuple[str, str]) -> dict:
    src, dst = task
    try:
        out = Path(dst)
        if out.exists():
            d = np.load(out)
            return dict(status="skip", dst=dst, n_frames=int(d["pose"].shape[0]),
                        fps=float(d["fps"]),
                        det_rate=float(d["pose_ok"].mean()),
                        lhand_rate=float(d["lhand_ok"].mean()),
                        rhand_rate=float(d["rhand_ok"].mean()))

        frames, fps = read_video(src, max_frames=_ARGS.get("max_frames", 300),
                                 stride=_ARGS.get("stride", 1))
        if len(frames) == 0:
            return dict(status="empty", dst=dst, n_frames=0, fps=0.0)

        seq = frames_to_raw(frames, _EXTRACTOR, fps=fps)

        # Clip mà pose gần như không bao giờ được phát hiện thì vô dụng cho huấn luyện.
        det_rate = float(seq.pose_ok.mean())
        out.parent.mkdir(parents=True, exist_ok=True)
        seq.to_npz(out)
        return dict(status="ok", dst=dst, n_frames=len(seq), fps=fps,
                    det_rate=det_rate,
                    lhand_rate=float(seq.lhand_ok.mean()),
                    rhand_rate=float(seq.rhand_ok.mean()))
    except Exception:
        return dict(status="error", dst=dst, n_frames=0, fps=0.0,
                    error=traceback.format_exc(limit=3))


def collect_tasks(root: Path, out: Path) -> list[tuple[str, str]]:
    tasks = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for v in sorted(label_dir.iterdir()):
            if v.suffix in VIDEO_EXT:
                dst = out / label_dir.name / (v.stem + ".npz")
                tasks.append((str(v), str(dst)))
    return tasks


def derive_group(stem: str, pattern: str | None) -> str:
    """Khoá nhóm dùng để chia train/val/test mà không rò rỉ dữ liệu.

    Với bộ HF namwu, các file 127824.mp4 / 127824_1.mp4 / 127824_2.mp4 là biến thể
    augmentation của CÙNG một clip gốc. Nếu chia ngẫu nhiên, bản gốc rơi vào train
    còn bản _1 rơi vào test thì độ chính xác báo cáo sẽ là con số ảo.
    Regex mặc định '^(\\d+)' cắt bỏ hậu tố _N và gom chúng về một nhóm.
    """
    if not pattern:
        return stem
    m = re.match(pattern, stem)
    return m.group(1) if m else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục gốc chứa <nhãn>/<video>")
    ap.add_argument("--out", required=True, help="thư mục xuất landmark")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--group-regex", default=r"^(\d+)",
                    help="regex lấy khoá nhóm từ tên file; '' để mỗi file một nhóm")
    ap.add_argument("--signer-regex", default="",
                    help="regex lấy ID người biểu diễn từ ĐƯỜNG DẪN ĐẦY ĐỦ")
    ap.add_argument("--min-det-rate", type=float, default=0.3,
                    help="loại clip có tỉ lệ phát hiện pose thấp hơn ngưỡng")
    ap.add_argument("--split-tag", default="",
                    help="ghi nhãn split có sẵn của dataset gốc (train/test). "
                         "Dùng khi dataset ĐÃ chia sẵn theo người biểu diễn — "
                         "khi đó phải giữ nguyên phép chia đó, không tự chia lại.")
    ap.add_argument("--only-labels", default="",
                    help="file txt liệt kê các gloss cần lấy, mỗi dòng một gloss")
    args = ap.parse_args()

    global _ARGS
    _ARGS = dict(max_frames=args.max_frames, stride=args.stride)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out / "manifest.csv"

    keep = None
    if args.only_labels:
        keep = {l.strip() for l in Path(args.only_labels).read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")}
        print(f"[i] Lọc còn {len(keep)} gloss theo {args.only_labels}")

    tasks = collect_tasks(root, out)
    if keep is not None:
        tasks = [t for t in tasks if Path(t[0]).parent.name in keep]
    if not tasks:
        print(f"[!] Không tìm thấy video nào trong {root}", file=sys.stderr)
        sys.exit(1)
    print(f"[i] Tìm thấy {len(tasks)} clip trong {len(set(Path(t[0]).parent for t in tasks))} nhãn")

    check_mediapipe(args.model_complexity)

    t0 = time.time()
    results = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.model_complexity, _ARGS)) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=8), 1):
            results.append(r)
            if i % 100 == 0 or i == len(tasks):
                el = time.time() - t0
                eta = el / i * (len(tasks) - i)
                nerr = sum(1 for x in results if x["status"] == "error")
                print(f"  {i}/{len(tasks)}  đã chạy {el/60:.1f}m  "
                      f"còn ~{eta/60:.1f}m  lỗi={nerr}", flush=True)

    by_dst = {r["dst"]: r for r in results}
    rows = []
    for src, dst in tasks:
        r = by_dst.get(dst, {})
        if r.get("status") in ("error", "empty"):
            continue
        p = Path(dst)
        stem = p.stem
        rows.append(dict(
            npz=str(p),
            video=src,
            label=p.parent.name,
            stem=stem,
            group_id=derive_group(stem, args.group_regex or None),
            signer_id=(re.search(args.signer_regex, src).group(1)
                       if args.signer_regex and re.search(args.signer_regex, src) else ""),
            split=args.split_tag,
            n_frames=r.get("n_frames", 0),
            fps=r.get("fps", 0.0),
            det_rate=r.get("det_rate", 1.0),
            lhand_rate=r.get("lhand_rate", 0.0),
            rhand_rate=r.get("rhand_rate", 0.0),
        ))

    df = pd.DataFrame(rows)
    before = len(df)
    df = df[df["det_rate"] >= args.min_det_rate].reset_index(drop=True)
    df = df[df["n_frames"] >= 6].reset_index(drop=True)

    df.to_csv(manifest_path, index=False)

    n_err = sum(1 for r in results if r["status"] == "error")
    print("\n" + "=" * 58)
    print(f"Manifest : {manifest_path}")
    print(f"Clip dùng được  : {len(df)} / {len(tasks)}  (loại {before - len(df)} do "
          f"phát hiện pose kém, {n_err} lỗi đọc)")
    print(f"Số nhãn         : {df['label'].nunique()}")
    print(f"Số nhóm         : {df['group_id'].nunique()}")
    if args.split_tag:
        print(f"Nhãn split      : {args.split_tag}")
    if df["signer_id"].astype(bool).any():
        print(f"Số người biểu diễn: {df['signer_id'].nunique()}")
    else:
        print("Số người biểu diễn: KHÔNG CÓ  -> chỉ chia theo nhóm được, "
              "không đánh giá được khả năng khái quát hoá sang người mới")
    print(f"Mẫu/nhãn        : trung bình {len(df)/max(df['label'].nunique(),1):.1f}, "
          f"ít nhất {df.groupby('label').size().min()}")
    print(f"Tỉ lệ thấy tay  : trái {df['lhand_rate'].mean():.2f}  "
          f"phải {df['rhand_rate'].mean():.2f}")
    print(f"Tổng thời gian  : {(time.time()-t0)/60:.1f} phút")
    print("=" * 58)

    for r in results:
        if r["status"] == "error":
            print("\n[lỗi mẫu]", r["dst"], "\n", r.get("error", "")[:400])
            break


if __name__ == "__main__":
    main()
