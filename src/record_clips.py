"""
Ghi clip từ webcam.

Dùng cho hai mục đích, cả hai đều BẮT BUỘC và không dataset công khai nào thay thế được:

1) Tập test liên tục (--mode sentence)
   Mọi dataset VSL công khai đều gồm clip đã cắt sẵn từng ký hiệu. Không có tập
   nào đo được hệ thống streaming của bạn hoạt động ra sao khi người dùng ký hiệu
   liên tục. Bạn cần tự quay: khoảng 15-20 câu, 2-3 người, mỗi câu 3-5 ký hiệu.
   Một buổi là xong, và đây là con số quan trọng nhất trong báo cáo.

2) Clip trạng thái nghỉ (--mode idle)
   Ngồi yên, gãi đầu, uống nước, chỉnh tóc, nói chuyện. Không có lớp negative thật,
   hệ thống sẽ phun gloss liên tục mỗi khi bạn cử động bất kỳ.

Ví dụ:
    python record_clips.py --mode sentence --out data/continuous \\
        --signer B01 --script configs/test_sentences.txt
    python record_clips.py --mode idle --out data/raw/__NOSIGN__ --signer B01 -n 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sentence", choices=["sentence", "idle", "sign"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--signer", required=True, help="mã người biểu diễn, vd B01")
    ap.add_argument("--script", default="", help="file txt, mỗi dòng một câu gloss")
    ap.add_argument("--label", default="", help="dùng cho --mode sign")
    ap.add_argument("-n", "--count", type=int, default=20)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "sentence":
        if not args.script:
            raise SystemExit("--mode sentence cần --script trỏ tới file câu")
        prompts = [l.strip() for l in Path(args.script).read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.startswith("#")]
    elif args.mode == "idle":
        prompts = ["NGỒI YÊN / cử động tự nhiên, KHÔNG ký hiệu"] * args.count
    else:
        if not args.label:
            raise SystemExit("--mode sign cần --label")
        prompts = [args.label] * args.count

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Không mở được webcam")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    meta = []
    i = 0

    print("SPACE = bắt đầu/dừng ghi   s = bỏ qua   q = thoát")
    while i < len(prompts):
        recording = False
        writer = None
        t_start = 0.0
        name = f"{args.signer}_{args.mode}_{i:03d}"
        dst = out / f"{name}.mp4"

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            disp = frame.copy()
            h, w = disp.shape[:2]

            cv2.rectangle(disp, (0, 0), (w, 70), (25, 25, 30), -1)
            cv2.putText(disp, f"[{i+1}/{len(prompts)}]", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(disp, prompts[i][:56], (12, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if recording:
                el = time.time() - t_start
                cv2.circle(disp, (w - 32, 32), 12, (0, 0, 255), -1)
                cv2.putText(disp, f"{el:4.1f}s", (w - 120, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                writer.write(frame)
                if el >= args.seconds:
                    recording = False
                    writer.release()
                    writer = None
                    meta.append(dict(file=dst.name, prompt=prompts[i],
                                     signer=args.signer, mode=args.mode,
                                     seconds=el))
                    print(f"  đã ghi {dst.name}  ({el:.1f}s)")
                    i += 1
                    break

            cv2.imshow("recorder", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                if writer:
                    writer.release()
                cap.release()
                cv2.destroyAllWindows()
                _save_meta(out, meta)
                return
            if k == ord("s") and not recording:
                i += 1
                break
            if k == ord(" "):
                if not recording:
                    writer = cv2.VideoWriter(str(dst), fourcc, args.fps,
                                             (frame.shape[1], frame.shape[0]))
                    t_start = time.time()
                    recording = True
                else:
                    el = time.time() - t_start
                    recording = False
                    writer.release()
                    writer = None
                    meta.append(dict(file=dst.name, prompt=prompts[i],
                                     signer=args.signer, mode=args.mode, seconds=el))
                    print(f"  đã ghi {dst.name}  ({el:.1f}s)")
                    i += 1
                    break

    cap.release()
    cv2.destroyAllWindows()
    _save_meta(out, meta)


def _save_meta(out: Path, meta: list[dict]) -> None:
    if not meta:
        return
    p = out / "recordings.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\n[ok] Ghi {len(meta)} clip, metadata tại {p}")


if __name__ == "__main__":
    main()
