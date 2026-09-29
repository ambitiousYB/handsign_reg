"""
Đánh giá chế độ streaming trên tập câu liên tục tự quay.

Đây là bảng kết quả phân biệt một đồ án thật với một notebook Kaggle. Accuracy
offline trên clip đã cắt sẵn KHÔNG nói lên điều gì về hệ thống chạy thực tế; con
số trung thực là Word Error Rate khi ký hiệu liên tục.

    python eval_streaming.py --ckpt runs/bilstm_group/best.pt \\
        --videos data/continuous --refs configs/test_sentences.txt

Ba chỉ số báo cáo:
  WER              — tỉ lệ lỗi cấp từ (chèn + xoá + thay thế) / số từ tham chiếu
  Độ trễ trung bình — từ lúc kết thúc ký hiệu tới lúc hệ thống xuất ra gloss
  Cảnh báo giả/phút — số gloss xuất ra trong lúc người dùng KHÔNG ký hiệu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEQ_LEN, NOSIGN_LABEL, StreamConfig
from landmarks import HolisticExtractor, build_window
from stream import FrameBuffer, Recognizer, SegmentDetector


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein trên chuỗi từ -> (thay thế, xoá, chèn)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    B = np.zeros((n + 1, m + 1), np.int8)   # 0=match/sub 1=del 2=ins
    for i in range(1, n + 1):
        D[i, 0], B[i, 0] = i, 1
    for j in range(1, m + 1):
        D[0, j], B[0, j] = j, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cands = (D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]),
                     D[i - 1, j] + 1, D[i, j - 1] + 1)
            k = int(np.argmin(cands))
            D[i, j], B[i, j] = cands[k], k

    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i, j]
        if b == 0:
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            i, j = i - 1, j - 1
        elif b == 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def run_video(path: Path, rec: Recognizer, cfg: StreamConfig,
              model_complexity: int = 0) -> tuple[list[str], list[float], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    buf = FrameBuffer(maxlen=cfg.max_sign_frames + 20, fps=fps)
    det = SegmentDetector(cfg)
    out: list[str] = []
    lat: list[float] = []
    dt = 1.0 / fps
    frame_i = 0
    dur = 0.0

    with HolisticExtractor(model_complexity=model_complexity) as ext:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frame_i += 1
            d = ext.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            energy = buf.push(d, dt)
            hands = (d["lhand_ok"] + d["rhand_ok"]) > 0
            ev = det.step(energy, hands)

            if ev == "end":
                t0 = time.time()
                n = min(cfg.max_sign_frames, len(buf.buf))
                seq = buf.to_raw(n)
                seq.fps = fps
                prob = rec.predict(build_window(seq, SEQ_LEN))
                order = np.argsort(-prob)
                name = rec.labels[int(order[0])]
                conf = float(prob[order[0]])
                margin = conf - float(prob[order[1]]) if len(order) > 1 else conf
                if (name != NOSIGN_LABEL and conf >= cfg.conf_threshold
                        and margin >= cfg.margin_threshold):
                    out.append(name)
                    # độ trễ = số frame yên tĩnh phải chờ + thời gian suy luận
                    lat.append(cfg.quiet_frames / fps + (time.time() - t0))
    dur = frame_i / fps
    cap.release()
    return out, lat, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--videos", required=True, help="thư mục chứa clip câu liên tục")
    ap.add_argument("--refs", required=True,
                    help="file txt: mỗi dòng là chuỗi gloss tham chiếu, cùng thứ tự")
    ap.add_argument("--idle-videos", default="", help="thư mục clip nghỉ để đo báo động giả")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--model-complexity", type=int, default=0)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--out", default="runs/streaming_eval.json")
    args = ap.parse_args()

    cfg = StreamConfig()
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    rec = Recognizer(args.ckpt, backend=args.backend)

    refs = [l.strip().split() for l in
            Path(args.refs).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    vids = sorted(p for p in Path(args.videos).glob("*.mp4"))
    if len(vids) != len(refs):
        print(f"[!] {len(vids)} video nhưng {len(refs)} câu tham chiếu — "
              f"ghép theo thứ tự tên file, kiểm tra lại nếu lệch")

    S = D = I = N = 0
    all_lat: list[float] = []
    rows = []
    for v, ref in zip(vids, refs):
        hyp, lat, dur = run_video(v, rec, cfg, args.model_complexity)
        s, d, i = edit_ops(ref, hyp)
        S, D, I, N = S + s, D + d, I + i, N + len(ref)
        all_lat += lat
        rows.append(dict(video=v.name, ref=" ".join(ref), hyp=" ".join(hyp),
                         sub=s, dele=d, ins=i,
                         wer=round((s + d + i) / max(len(ref), 1), 3)))
        print(f"  {v.name:28s} WER {rows[-1]['wer']:.2f}  | {' '.join(hyp)}")

    wer = (S + D + I) / max(N, 1)
    res = dict(
        wer=round(wer, 4), sub=S, deletion=D, insertion=I, ref_words=N,
        latency_mean_s=round(float(np.mean(all_lat)), 3) if all_lat else None,
        latency_p90_s=round(float(np.percentile(all_lat, 90)), 3) if all_lat else None,
        conf_threshold=cfg.conf_threshold,
    )

    if args.idle_videos:
        fp, mins = 0, 0.0
        for v in sorted(Path(args.idle_videos).glob("*.mp4")):
            hyp, _, dur = run_video(v, rec, cfg, args.model_complexity)
            fp += len(hyp)
            mins += dur / 60.0
        res["false_emits_per_min"] = round(fp / max(mins, 1e-6), 2)
        res["idle_minutes"] = round(mins, 2)

    print("\n" + "=" * 58)
    print(f"WER tổng     : {wer:.3f}   (thay {S}, xoá {D}, chèn {I} / {N} từ)")
    if all_lat:
        print(f"Độ trễ       : TB {res['latency_mean_s']:.2f}s, "
              f"p90 {res['latency_p90_s']:.2f}s")
    if "false_emits_per_min" in res:
        print(f"Báo động giả : {res['false_emits_per_min']}/phút "
              f"trên {res['idle_minutes']} phút clip nghỉ")
    print("=" * 58)
    print("Nhiều lỗi CHÈN  -> ngưỡng chuyển động quá nhạy hoặc thiếu lớp nghỉ.")
    print("Nhiều lỗi XOÁ   -> conf_threshold quá cao, hoặc motion_hi quá cao.")
    print("Nhiều lỗi THAY  -> vấn đề ở mô hình, xem lại bảng nhầm lẫn.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        dict(summary=res, per_video=rows), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[ok] Lưu {args.out}")


if __name__ == "__main__":
    main()
