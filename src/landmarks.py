"""
Trích xuất và chuẩn hoá landmark.

Module này được DÙNG CHUNG bởi hai nơi:
  * extract_dataset.py  — chạy offline trên Kaggle để sinh dữ liệu huấn luyện
  * stream.py           — chạy real-time trên laptop

Đây là ràng buộc thiết kế quan trọng nhất của cả dự án. Nếu bạn chỉnh công thức
chuẩn hoá ở một nơi mà quên nơi kia, mô hình vẫn chạy, vẫn xuất ra nhãn, nhưng độ
chính xác sẽ sụp mà không có bất kỳ thông báo lỗi nào.

Quy ước lưu trữ:
  Trên đĩa ta lưu landmark THÔ (chưa chuẩn hoá) để sau này đổi công thức chuẩn hoá
  mà không phải chạy lại khâu trích xuất vốn tốn hàng giờ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    POSE_IDX, FACE_IDX, N_POSE, N_FACE, N_HAND,
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_WRIST, R_WRIST,
    FEATURE_DIM, DIM_GLOBAL,
)

EPS = 1e-6


# ==========================================================================
# 1. Cấu trúc landmark thô
# ==========================================================================

@dataclass
class RawSequence:
    """Landmark thô của một clip, toạ độ đã chuẩn hoá về [0, 1] theo khung hình."""

    pose: np.ndarray        # (T, 9, 2)   float32
    pose_vis: np.ndarray    # (T, 9)      float32, độ tin cậy
    lhand: np.ndarray       # (T, 21, 2)
    rhand: np.ndarray       # (T, 21, 2)
    face: np.ndarray        # (T, 20, 2)
    lhand_ok: np.ndarray    # (T,)  0/1
    rhand_ok: np.ndarray    # (T,)  0/1
    pose_ok: np.ndarray     # (T,)  0/1
    fps: float = 30.0

    def __len__(self) -> int:
        return int(self.pose.shape[0])

    def to_npz(self, path) -> None:
        np.savez_compressed(
            path,
            pose=self.pose.astype(np.float16),
            pose_vis=self.pose_vis.astype(np.float16),
            lhand=self.lhand.astype(np.float16),
            rhand=self.rhand.astype(np.float16),
            face=self.face.astype(np.float16),
            lhand_ok=self.lhand_ok.astype(np.uint8),
            rhand_ok=self.rhand_ok.astype(np.uint8),
            pose_ok=self.pose_ok.astype(np.uint8),
            fps=np.float32(self.fps),
        )

    @staticmethod
    def from_npz(path) -> "RawSequence":
        d = np.load(path)
        return RawSequence(
            pose=d["pose"].astype(np.float32),
            pose_vis=d["pose_vis"].astype(np.float32),
            lhand=d["lhand"].astype(np.float32),
            rhand=d["rhand"].astype(np.float32),
            face=d["face"].astype(np.float32),
            lhand_ok=d["lhand_ok"].astype(np.float32),
            rhand_ok=d["rhand_ok"].astype(np.float32),
            pose_ok=d["pose_ok"].astype(np.float32),
            fps=float(d["fps"]),
        )

    @staticmethod
    def empty(T: int, fps: float = 30.0) -> "RawSequence":
        return RawSequence(
            pose=np.zeros((T, N_POSE, 2), np.float32),
            pose_vis=np.zeros((T, N_POSE), np.float32),
            lhand=np.zeros((T, N_HAND, 2), np.float32),
            rhand=np.zeros((T, N_HAND, 2), np.float32),
            face=np.zeros((T, N_FACE, 2), np.float32),
            lhand_ok=np.zeros((T,), np.float32),
            rhand_ok=np.zeros((T,), np.float32),
            pose_ok=np.zeros((T,), np.float32),
            fps=fps,
        )

    def slice(self, a: int, b: int) -> "RawSequence":
        return RawSequence(
            pose=self.pose[a:b], pose_vis=self.pose_vis[a:b],
            lhand=self.lhand[a:b], rhand=self.rhand[a:b], face=self.face[a:b],
            lhand_ok=self.lhand_ok[a:b], rhand_ok=self.rhand_ok[a:b],
            pose_ok=self.pose_ok[a:b], fps=self.fps,
        )


# ==========================================================================
# 2. Bộ trích xuất MediaPipe
# ==========================================================================

class HolisticExtractor:
    """Bọc mediapipe Holistic. Dùng được cho cả video offline lẫn webcam.

    Yêu cầu:  pip install mediapipe==0.10.14
    Các bản mediapipe mới hơn đã bỏ `mp.solutions.holistic`; nếu bạn buộc phải
    dùng bản mới thì thay bằng PoseLandmarker + HandLandmarker của Tasks API và
    giữ nguyên đầu ra của hàm `process()` dưới đây.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_face_landmarks: bool = False,
    ):
        import mediapipe as mp  # import trễ để module này vẫn dùng được khi chỉ đọc npz

        self._mp = mp
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, frame_rgb: np.ndarray) -> dict:
        """Nhận một frame RGB (H, W, 3) uint8, trả về dict landmark của frame đó."""
        res = self._holistic.process(frame_rgb)

        pose = np.zeros((N_POSE, 2), np.float32)
        pose_vis = np.zeros((N_POSE,), np.float32)
        pose_ok = 0.0
        if res.pose_landmarks is not None:
            lm = res.pose_landmarks.landmark
            for i, idx in enumerate(POSE_IDX):
                pose[i, 0] = lm[idx].x
                pose[i, 1] = lm[idx].y
                pose_vis[i] = lm[idx].visibility
            pose_ok = 1.0

        face = np.zeros((N_FACE, 2), np.float32)
        if res.face_landmarks is not None:
            lm = res.face_landmarks.landmark
            n = len(lm)
            for i, idx in enumerate(FACE_IDX):
                if idx < n:
                    face[i, 0] = lm[idx].x
                    face[i, 1] = lm[idx].y

        def hand_of(h):
            arr = np.zeros((N_HAND, 2), np.float32)
            if h is None:
                return arr, 0.0
            for i, p in enumerate(h.landmark):
                arr[i, 0] = p.x
                arr[i, 1] = p.y
            return arr, 1.0

        lhand, lok = hand_of(res.left_hand_landmarks)
        rhand, rok = hand_of(res.right_hand_landmarks)

        return dict(
            pose=pose, pose_vis=pose_vis, pose_ok=pose_ok,
            lhand=lhand, lhand_ok=lok,
            rhand=rhand, rhand_ok=rok,
            face=face,
        )


def frames_to_raw(frames_rgb, extractor: HolisticExtractor, fps: float) -> RawSequence:
    """Chạy extractor trên một danh sách frame RGB -> RawSequence."""
    T = len(frames_rgb)
    seq = RawSequence.empty(T, fps=fps)
    for t, fr in enumerate(frames_rgb):
        d = extractor.process(fr)
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
# 3. Chuẩn hoá và tạo vector đặc trưng
# ==========================================================================

def _forward_fill_pose(pose: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Điền các frame mất pose bằng frame hợp lệ gần nhất phía trước (rồi phía sau)."""
    T = pose.shape[0]
    out = pose.copy()
    last: Optional[np.ndarray] = None
    for t in range(T):
        if ok[t] > 0.5:
            last = out[t]
        elif last is not None:
            out[t] = last
    nxt: Optional[np.ndarray] = None
    for t in range(T - 1, -1, -1):
        if ok[t] > 0.5:
            nxt = out[t]
        elif nxt is not None and last is None:
            out[t] = nxt
    return out


def _body_frame(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính gốc toạ độ, tỉ lệ và ma trận xoay cho từng frame.

    - Gốc  : trung điểm hai vai
    - Tỉ lệ: khoảng cách hai vai (dự phòng: khoảng cách vai-hông, rồi hằng số)
    - Xoay : đưa đường nối hai vai về nằm ngang, giúp bất biến với việc người
             nghiêng người hoặc camera đặt lệch.
    """
    T = pose.shape[0]
    ls, rs = pose[:, L_SHOULDER], pose[:, R_SHOULDER]
    center = (ls + rs) / 2.0

    shoulder_vec = ls - rs                              # (T, 2)
    scale = np.linalg.norm(shoulder_vec, axis=1)        # (T,)

    hip_center = (pose[:, L_HIP] + pose[:, R_HIP]) / 2.0
    torso = np.linalg.norm(center - hip_center, axis=1)

    bad = scale < 1e-3
    scale = np.where(bad, torso * 0.9, scale)
    still_bad = scale < 1e-3
    if still_bad.any():
        good = scale[~still_bad]
        fallback = float(np.median(good)) if good.size else 0.25
        scale = np.where(still_bad, fallback, scale)

    ang = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
    ang = np.where(bad, 0.0, ang)
    cos, sin = np.cos(-ang), np.sin(-ang)
    rot = np.stack(
        [np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)], axis=1
    )  # (T, 2, 2)
    return center, scale.astype(np.float32), rot.astype(np.float32)


def _apply_frame(pts: np.ndarray, center, scale, rot) -> np.ndarray:
    """pts: (T, K, 2) -> chuẩn hoá về hệ toạ độ thân người."""
    x = pts - center[:, None, :]
    x = np.einsum("tij,tkj->tki", rot, x)
    return x / (scale[:, None, None] + EPS)


MIN_HAND_SPAN = 0.012   # theo đơn vị khung hình đã chuẩn hoá [0, 1]


def _hand_local(hand: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hoá bàn tay theo chính nó: gốc = cổ tay, tỉ lệ = kích thước bàn tay.

    Mục đích: tách *hình dạng bàn tay* khỏi *vị trí bàn tay trong không gian*.
    Hai ký hiệu có thể cùng quỹ đạo nhưng khác handshape, và ngược lại.

    Tỉ lệ dùng khoảng cách RMS của cả 21 điểm tới cổ tay, KHÔNG dùng một cặp điểm
    đơn lẻ. Lý do: khi bàn tay hướng thẳng vào camera (nắm đấm, hoặc bàn tay nhìn
    nghiêng), khoảng cách cổ tay -> đốt giữa co lại gần bằng 0; chia cho nó sẽ
    khuếch đại nhiễu landmark lên hàng chục lần và biến đặc trưng handshape thành
    nhiễu thuần tuý. RMS trên 21 điểm ổn định hơn nhiều.

    Trả về (đặc trưng cục bộ, cờ hợp lệ đã cập nhật): bàn tay suy biến bị coi như
    không phát hiện được, thay vì tạo ra số liệu rác.
    """
    wrist = hand[:, 0:1, :]
    rel = hand - wrist
    span = np.sqrt((rel ** 2).sum(axis=2).mean(axis=1))      # (T,) RMS
    degenerate = span < MIN_HAND_SPAN
    ok_out = ok * (~degenerate).astype(np.float32)
    span = np.where(degenerate, 1.0, span)
    out = rel / (span[:, None, None] + EPS)
    return out * ok_out[:, None, None], ok_out


def featurize(seq: RawSequence) -> np.ndarray:
    """RawSequence -> (T, FEATURE_DIM) chưa tính vận tốc.

    Vận tốc được tính SAU khi resample về độ dài cố định, để đặc trưng bất biến
    với fps của nguồn video. Đây là lý do hàm này trả về phần tĩnh trước.
    """
    T = len(seq)
    if T == 0:
        return np.zeros((0, FEATURE_DIM), np.float32)

    pose = _forward_fill_pose(seq.pose, seq.pose_ok)
    center, scale, rot = _body_frame(pose)

    l_lh, lok = _hand_local(seq.lhand, seq.lhand_ok)
    l_rh, rok = _hand_local(seq.rhand, seq.rhand_ok)

    g_pose = _apply_frame(pose, center, scale, rot)                 # (T, 9, 2)
    g_lh = _apply_frame(seq.lhand, center, scale, rot) * lok[:, None, None]
    g_rh = _apply_frame(seq.rhand, center, scale, rot) * rok[:, None, None]
    g_face = _apply_frame(seq.face, center, scale, rot)

    glob = np.concatenate(
        [g_pose.reshape(T, -1), g_lh.reshape(T, -1),
         g_rh.reshape(T, -1), g_face.reshape(T, -1)], axis=1
    )                                                               # (T, 142)
    loc = np.concatenate([l_lh.reshape(T, -1), l_rh.reshape(T, -1)], axis=1)  # (T, 84)
    flags = np.stack([lok, rok], axis=1)                            # (T, 2)

    static = np.concatenate([glob, loc, flags], axis=1).astype(np.float32)
    return np.clip(static, -8.0, 8.0)


def resample(x: np.ndarray, T_out: int) -> np.ndarray:
    """Nội suy tuyến tính chuỗi (T, D) về (T_out, D)."""
    T_in, D = x.shape
    if T_in == 0:
        return np.zeros((T_out, D), np.float32)
    if T_in == 1:
        return np.repeat(x, T_out, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, T_in)
    dst = np.linspace(0.0, 1.0, T_out)
    out = np.empty((T_out, D), np.float32)
    for d in range(D):
        out[:, d] = np.interp(dst, src, x[:, d])
    return out


def add_velocity(static: np.ndarray) -> np.ndarray:
    """(T, 228) -> (T, 370): ghép thêm vận tốc của phần toạ độ toàn cục."""
    glob = static[:, :DIM_GLOBAL]
    vel = np.zeros_like(glob)
    if glob.shape[0] > 1:
        vel[1:] = glob[1:] - glob[:-1]
    return np.concatenate([static, vel], axis=1).astype(np.float32)


def build_window(seq: RawSequence, T_out: int) -> np.ndarray:
    """Đường đi hoàn chỉnh: RawSequence -> (T_out, FEATURE_DIM) sẵn sàng cho mô hình."""
    static = featurize(seq)
    static = resample(static, T_out)
    return add_velocity(static)


# ==========================================================================
# 4. Năng lượng chuyển động — dùng cho phát hiện ranh giới khi streaming
# ==========================================================================

def motion_energy(seq: RawSequence, win: int = 3) -> np.ndarray:
    """Tốc độ cổ tay chuẩn hoá theo độ rộng vai, đơn vị: (rộng vai)/giây.

    Đại lượng này bất biến với khoảng cách người tới camera và với độ phân giải,
    nên một ngưỡng cố định dùng được cho nhiều người và nhiều webcam khác nhau.
    """
    T = len(seq)
    if T < 2:
        return np.zeros((T,), np.float32)

    pose = _forward_fill_pose(seq.pose, seq.pose_ok)
    center, scale, rot = _body_frame(pose)
    wr = _apply_frame(pose[:, [L_WRIST, R_WRIST], :], center, scale, rot)  # (T, 2, 2)

    d = np.zeros((T,), np.float32)
    diff = np.linalg.norm(wr[1:] - wr[:-1], axis=2).mean(axis=1)
    d[1:] = diff * seq.fps

    if win > 1:
        k = np.ones(win, np.float32) / win
        d = np.convolve(d, k, mode="same")
    return d.astype(np.float32)
