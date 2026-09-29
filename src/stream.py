"""
Nhận diện real-time từ webcam.

Chạy trên CPU laptop. Toàn bộ chi phí nặng nằm ở MediaPipe; mô hình phân loại
chuỗi chỉ tốn vài mili-giây mỗi cửa sổ.

    python stream.py --ckpt runs/bilstm_group/best.pt --camera 0

Máy trạng thái phát hiện ranh giới:

    IDLE ──(năng lượng > motion_hi)──> ACTIVE
    ACTIVE ──(năng lượng < motion_lo trong quiet_frames)──> phân loại -> COOLDOWN
    ACTIVE ──(vượt max_sign_frames)──> cắt cưỡng bức -> phân loại -> COOLDOWN
    COOLDOWN ──(hết cooldown_frames)──> IDLE

Đây là phần mà bộ dữ liệu huấn luyện KHÔNG dạy được cho bạn: mọi dataset đều gồm
clip đã cắt sẵn, nên logic ranh giới phải tự xây và tự đo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (SEQ_LEN, FEATURE_DIM, NOSIGN_LABEL, StreamConfig,
                    L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST)
from landmarks import HolisticExtractor, RawSequence, build_window
from gloss2vi import GlossTranslator


# ==========================================================================
# Bộ suy luận (torch hoặc onnxruntime)
# ==========================================================================

class Recognizer:
    def __init__(self, ckpt: str, backend: str = "auto"):
        self.backend = backend
        p = Path(ckpt)
        if backend == "onnx" or (backend == "auto" and p.suffix == ".onnx"):
            self._init_onnx(p)
        else:
            self._init_torch(p)

    def _init_torch(self, p: Path):
        import torch
        from models import build_model
        st = torch.load(p, map_location="cpu")
        self.labels = {int(v): k for k, v in st["label_to_idx"].items()}
        self.model = build_model(st.get("arch", "bilstm"), st.get("feature_dim", FEATURE_DIM),
                                 len(self.labels), st.get("hidden", 256),
                                 st.get("layers", 2), 0.0)
        self.model.load_state_dict(st["model"])
        self.model.eval()
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2) - 1))
        self._torch = torch
        self.backend = "torch"

    def _init_onnx(self, p: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        meta = json.loads(Path(p).with_suffix(".labels.json").read_text(encoding="utf-8"))
        self.labels = {int(k): v for k, v in meta.items()}
        self.backend = "onnx"

    def predict(self, window: np.ndarray) -> np.ndarray:
        x = window[None].astype(np.float32)
        if self.backend == "torch":
            with self._torch.no_grad():
                logit = self.model(self._torch.from_numpy(x))
                return logit.softmax(-1).numpy()[0]
        logit = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        e = np.exp(logit - logit.max())
        return (e / e.sum())[0]


# ==========================================================================
# Bộ đệm frame + năng lượng chuyển động trực tuyến
# ==========================================================================

class FrameBuffer:
    """Giữ landmark của N frame gần nhất và tính năng lượng chuyển động tức thời."""

    def __init__(self, maxlen: int, fps: float):
        self.buf: deque[dict] = deque(maxlen=maxlen)
        self.fps = fps
        self._prev_wrist: np.ndarray | None = None
        self.energy = 0.0
        self._ema = 0.0

    def push(self, d: dict, dt: float) -> float:
        self.buf.append(d)

        pose = d["pose"]
        ls, rs = pose[L_SHOULDER], pose[R_SHOULDER]
        shoulder = float(np.linalg.norm(ls - rs))
        if shoulder < 1e-3 or d["pose_ok"] < 0.5:
            self._ema = self._ema * 0.7
            self.energy = self._ema
            return self.energy

        center = (ls + rs) / 2.0
        wr = (pose[[L_WRIST, R_WRIST]] - center) / shoulder
        if self._prev_wrist is not None and dt > 1e-4:
            v = float(np.linalg.norm(wr - self._prev_wrist, axis=1).mean() / dt)
        else:
            v = 0.0
        self._prev_wrist = wr
        self._ema = 0.6 * self._ema + 0.4 * v
        self.energy = self._ema
        return self.energy

    def to_raw(self, n: int | None = None) -> RawSequence:
        items = list(self.buf)[-n:] if n else list(self.buf)
        T = len(items)
        seq = RawSequence.empty(T, fps=self.fps)
        for t, d in enumerate(items):
            seq.pose[t] = d["pose"]
            seq.pose_vis[t] = d["pose_vis"]
            seq.pose_ok[t] = d["pose_ok"]
            seq.lhand[t] = d["lhand"]
            seq.rhand[t] = d["rhand"]
            seq.face[t] = d["face"]
            seq.lhand_ok[t] = d["lhand_ok"]
            seq.rhand_ok[t] = d["rhand_ok"]
        return seq


# ==========================================================================
# Máy trạng thái
# ==========================================================================

class SegmentDetector:
    IDLE, ACTIVE, COOLDOWN = 0, 1, 2

    def __init__(self, cfg: StreamConfig):
        self.cfg = cfg
        self.state = self.IDLE
        self.quiet = 0
        self.count = 0
        self.cool = 0

    def step(self, energy: float, hands_visible: bool) -> str:
        c = self.cfg
        if self.state == self.COOLDOWN:
            self.cool -= 1
            if self.cool <= 0:
                self.state = self.IDLE
            return "cooldown"

        if self.state == self.IDLE:
            if energy > c.motion_hi and hands_visible:
                self.state = self.ACTIVE
                self.count = 1
                self.quiet = 0
                return "start"
            return "idle"

        # ACTIVE
        self.count += 1
        if energy < c.motion_lo or not hands_visible:
            self.quiet += 1
        else:
            self.quiet = 0

        if self.quiet >= c.quiet_frames or self.count >= c.max_sign_frames:
            n = self.count
            self.state = self.COOLDOWN
            self.cool = c.cooldown_frames
            self.count = 0
            self.quiet = 0
            return "end" if n >= c.min_sign_frames else "discard"
        return "active"


# ==========================================================================
# Vòng lặp chính
# ==========================================================================

_FONTS: dict = {}


def _font(size: int):
    if size not in _FONTS:
        from PIL import ImageFont
        _FONTS[size] = None
        for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "/System/Library/Fonts/Supplemental/Arial.ttf"):
            try:
                _FONTS[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
    return _FONTS[size]


def put_text_vi(frame, items):
    """cv2.putText không vẽ được dấu tiếng Việt ('Xe đạp' -> 'Xe ??p'), nên
    vẽ bằng Pillow. items: [(chuỗi, (x, y_đường_chân_chữ), cỡ_px, màu_BGR)]."""
    try:
        from PIL import Image, ImageDraw
        fonts = [_font(size) for _, _, size, _ in items]
    except ImportError:
        fonts = [None]
    if None in fonts:
        for text, org, size, bgr in items:
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 32, bgr, 1)
        return
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)
    for (text, org, _, bgr), f in zip(items, fonts):
        dr.text(org, text, font=f, fill=tuple(bgr[::-1]), anchor="ls")
    frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_overlay(frame, state_name, energy, cfg, gloss_buf, sentence, fps, last):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 92), (22, 22, 26), -1)
    cv2.rectangle(frame, (0, h - 76), (w, h), (22, 22, 26), -1)

    color = {"IDLE": (150, 150, 150), "ACTIVE": (60, 200, 90),
             "COOLDOWN": (60, 150, 230)}[state_name]
    cv2.putText(frame, state_name, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"{fps:5.1f} FPS", (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    bar_w = int(min(energy / (cfg.motion_hi * 2.5), 1.0) * (w - 180))
    cv2.rectangle(frame, (150, 16), (150 + max(bar_w, 2), 34), color, -1)
    x_hi = 150 + int(min(cfg.motion_hi / (cfg.motion_hi * 2.5), 1.0) * (w - 180))
    x_lo = 150 + int(min(cfg.motion_lo / (cfg.motion_hi * 2.5), 1.0) * (w - 180))
    cv2.line(frame, (x_hi, 12), (x_hi, 38), (80, 80, 240), 2)
    cv2.line(frame, (x_lo, 12), (x_lo, 38), (80, 200, 240), 1)

    items = [("GLOSS: " + " ".join(gloss_buf[-8:]), (14, h - 46), 18, (200, 200, 200)),
             (sentence[:70], (14, h - 16), 22, (255, 255, 255))]
    if last:
        items.append((last, (14, 72), 22, (240, 240, 120)))
    put_text_vi(frame, items)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default="", help="chạy trên file video thay vì webcam")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx"])
    ap.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2],
                    help="0 = nhanh nhất, nên dùng trên CPU")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--motion-hi", type=float, default=None)
    ap.add_argument("--motion-lo", type=float, default=None)
    ap.add_argument("--dict", default="configs/gloss_vi.json")
    ap.add_argument("--log", default="", help="ghi nhật ký gloss ra file jsonl")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-flip", action="store_true",
                    help="đưa ảnh webcam GỐC (không lật gương) vào nhận diện; màn hình "
                         "vẫn hiện kiểu gương. Dùng khi video huấn luyện không phải ảnh gương")
    args = ap.parse_args()

    cfg = StreamConfig()
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.motion_hi is not None:
        cfg.motion_hi = args.motion_hi
    if args.motion_lo is not None:
        cfg.motion_lo = args.motion_lo

    rec = Recognizer(args.ckpt, backend=args.backend)
    print(f"[i] Backend {rec.backend}, {len(rec.labels)} lớp")
    translator = GlossTranslator(args.dict)

    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[!] Không mở được nguồn video", file=sys.stderr)
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or cfg.target_fps
    buf = FrameBuffer(maxlen=cfg.max_sign_frames + 20, fps=fps_src)
    det = SegmentDetector(cfg)

    gloss_buf: list[str] = []
    sentence = ""
    last_msg = ""
    pending: list[str] = []
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    t_prev = time.time()
    fps_ema = 0.0
    names = {0: "IDLE", 1: "ACTIVE", 2: "COOLDOWN"}

    with HolisticExtractor(model_complexity=args.model_complexity) as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # frame: ảnh đưa vào nhận diện; view: ảnh hiển thị (webcam luôn hiện kiểu gương)
            view = frame
            if not args.video:
                view = cv2.flip(frame, 1)
                if not args.no_flip:
                    frame = view

            now = time.time()
            dt = max(now - t_prev, 1e-4)
            t_prev = now
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            d = ext.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            energy = buf.push(d, dt)
            hands = (d["lhand_ok"] + d["rhand_ok"]) > 0

            ev = det.step(energy, hands)

            if ev == "end":
                n = min(det.cfg.max_sign_frames, len(buf.buf))
                seq = buf.to_raw(n)
                seq.fps = fps_ema if fps_ema > 1 else fps_src
                window = build_window(seq, SEQ_LEN)
                prob = rec.predict(window)

                order = np.argsort(-prob)
                top1, top2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
                name = rec.labels[top1]
                conf, margin = float(prob[top1]), float(prob[top1] - prob[top2])

                if name == NOSIGN_LABEL:
                    last_msg = "(nghỉ)"
                elif conf >= cfg.conf_threshold and margin >= cfg.margin_threshold:
                    pending.append(name)
                    if len(pending) >= 1:
                        gloss_buf.append(name)
                        sentence = translator.translate(gloss_buf)
                        last_msg = f"{name}  ({conf:.2f})"
                        pending.clear()
                        if logf:
                            logf.write(json.dumps(
                                dict(t=now, gloss=name, conf=conf, margin=margin,
                                     frames=n), ensure_ascii=False) + "\n")
                            logf.flush()
                else:
                    last_msg = f"? {name} ({conf:.2f}) - dưới ngưỡng"

            if not args.no_window:
                view = draw_overlay(view.copy(), names[det.state], energy, cfg,
                                    gloss_buf, sentence, fps_ema, last_msg)
                cv2.imshow("VSL streaming", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("c"):
                    gloss_buf.clear()
                    sentence = ""
                    last_msg = ""

    cap.release()
    cv2.destroyAllWindows()
    if logf:
        logf.close()
    print("\nGloss:", " ".join(gloss_buf))
    print("Câu  :", sentence)


if __name__ == "__main__":
    main()
