"""
Xuất mô hình sang ONNX để chạy nhanh hơn trên CPU laptop.

    python export_onnx.py --ckpt runs/bilstm_group/best.pt --out models/vsl.onnx
    python stream.py --ckpt models/vsl.onnx --backend onnx

onnxruntime thường nhanh hơn PyTorch eager trên CPU khoảng 1,5-3 lần với mô hình
nhỏ như thế này, và quan trọng hơn là bạn không phải cài PyTorch trên máy demo.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEQ_LEN, FEATURE_DIM
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="models/vsl.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cpu")
    label_to_idx = st["label_to_idx"]
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}

    model = build_model(st.get("arch", "bilstm"), st.get("feature_dim", FEATURE_DIM),
                        len(idx_to_label), st.get("hidden", 256),
                        st.get("layers", 2), 0.0)
    model.load_state_dict(st["model"])
    model.eval()

    seq_len = st.get("seq_len", SEQ_LEN)
    dummy = torch.randn(1, seq_len, st.get("feature_dim", FEATURE_DIM))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ghi nhãn TRƯỚC khi export, để export có lỗi thì vẫn còn file nhãn
    out.with_suffix(".labels.json").write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8")

    # PyTorch >= 2.9 mặc định dùng exporter dynamo, cần thêm gói onnxscript và
    # không nhận dynamic_axes. Exporter cũ (TorchScript) ổn định với LSTM -> ép dùng.
    kw = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, **kw,
    )

    # kiểm chứng: đầu ra ONNX phải khớp PyTorch
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        a = model(dummy).detach().numpy()
        b = sess.run(None, {"x": dummy.numpy()})[0]
        diff = float(np.abs(a - b).max())
        print(f"[ok] Sai lệch lớn nhất PyTorch vs ONNX: {diff:.2e}")
        if diff > 1e-3:
            print("[!] Sai lệch lớn bất thường — kiểm tra lại phiên bản opset")
    except ImportError:
        print("[i] Chưa cài onnxruntime, bỏ qua bước kiểm chứng")

    print(f"[ok] Đã xuất {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"[ok] Nhãn: {out.with_suffix('.labels.json')}")


if __name__ == "__main__":
    main()
