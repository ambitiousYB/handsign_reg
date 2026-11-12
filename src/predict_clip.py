"""
Dự đoán trên từng clip ĐÃ CẮT SẴN (mỗi clip một ký hiệu), y hệt cách huấn luyện.

Dùng để tách hai nguồn lỗi khi demo webcam kém:
  - clip của dataset mà đoán ĐÚNG  -> mô hình và đường xử lý ổn, lỗi nằm ở khác
    biệt webcam / người ký / máy trạng thái cắt đoạn.
  - clip của dataset mà đoán SAI   -> đường xử lý trên máy còn lệch so với Kaggle.

stream.py --video KHÔNG hợp cho việc này: clip dataset ngắn, thường kết thúc
trước khi tay dừng hẳn nên máy trạng thái không bao giờ gọi mô hình.

    python src/predict_clip.py --ckpt models/vsl.onnx video1.mp4 video2.mp4
    python src/predict_clip.py --ckpt models/vsl.onnx thu_muc_chua_video/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEQ_LEN  # noqa: E402
from extract_dataset import VIDEO_EXT, read_video  # noqa: E402
from landmarks import HolisticExtractor, build_window, frames_to_raw  # noqa: E402
from stream import Recognizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx"])
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                    help="mặc định 1 = đúng mức đã dùng khi trích dữ liệu huấn luyện")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--flip", action="store_true",
                    help="lật gương video trước khi nhận diện (kiểm tra chuyện chiều ảnh)")
    ap.add_argument("videos", nargs="+", help="file video hoặc thư mục")
    args = ap.parse_args()

    paths: list[Path] = []
    for v in map(Path, args.videos):
        paths += sorted(p for p in v.rglob("*") if p.suffix in VIDEO_EXT) if v.is_dir() else [v]

    rec = Recognizer(args.ckpt, backend=args.backend)
    known = set(rec.labels.values())

    # Mỗi clip một extractor mới, như mỗi clip được xử lý độc lập lúc trích xuất
    for p in paths:
        frames, fps = read_video(str(p))
        if not frames:
            print(f"{p.name}: không đọc được video")
            continue
        if args.flip:
            frames = [np.ascontiguousarray(f[:, ::-1]) for f in frames]
        with HolisticExtractor(static_image_mode=False,
                               model_complexity=args.model_complexity) as ext:
            seq = frames_to_raw(frames, ext, fps=fps)
        prob = rec.predict(build_window(seq, SEQ_LEN))
        order = np.argsort(-prob)[:args.top]

        # Thư mục cha thường là tên gloss (frame_splited/test/<gloss>/x.mp4)
        truth = p.parent.name
        mark = ""
        if truth in known:
            rank = int(np.where(np.argsort(-prob) == next(
                k for k, v in rec.labels.items() if v == truth))[0][0]) + 1
            mark = f"  [đáp án '{truth}' xếp hạng {rank}]"
        print(f"{p.name}  ({len(frames)} frame, tay T {seq.lhand_ok.mean():.2f} "
              f"P {seq.rhand_ok.mean():.2f}){mark}")
        print("    " + "  ".join(f"{rec.labels[int(i)]} {prob[int(i)]:.2f}" for i in order))


if __name__ == "__main__":
    main()
