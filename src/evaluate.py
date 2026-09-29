"""
Đánh giá offline và phân tích lỗi.

Sinh ra đúng những bảng/hình bạn cần dán vào báo cáo:
  - báo cáo theo từng lớp (precision / recall / F1 / số mẫu)
  - danh sách cặp nhầm lẫn nhiều nhất  <- phần phân tích đáng giá nhất
  - biểu đồ độ chính xác theo độ tin cậy, để chọn ngưỡng cho streaming
  - bảng so sánh random split vs group/signer split

    python evaluate.py --run runs/bilstm_group
    python evaluate.py --compare runs/bilstm_random runs/bilstm_group runs/bilstm_signer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def load_run(run: Path):
    d = np.load(run / "test_preds.npz")
    labels = json.loads((run / "labels.json").read_text(encoding="utf-8"))
    idx_to_label = {int(k): v for k, v in labels.items()}
    return d["y"], d["pred"], d["prob"], idx_to_label


def per_class(y, pred, idx_to_label, out: Path):
    present = sorted(set(y.tolist()) | set(pred.tolist()))
    names = [idx_to_label[i] for i in present]
    rep = classification_report(y, pred, labels=present, target_names=names,
                                output_dict=True, zero_division=0)
    rows = [dict(label=k, precision=v["precision"], recall=v["recall"],
                 f1=v["f1-score"], support=int(v["support"]))
            for k, v in rep.items() if isinstance(v, dict) and k in names]
    df = pd.DataFrame(rows).sort_values("f1")
    df.to_csv(out / "per_class.csv", index=False)
    return df


def top_confusions(y, pred, idx_to_label, out: Path, k: int = 20):
    present = sorted(set(y.tolist()) | set(pred.tolist()))
    cm = confusion_matrix(y, pred, labels=present)
    rows = []
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if i != j and cm[i, j] > 0:
                rows.append(dict(thuc_te=idx_to_label[a], du_doan=idx_to_label[b],
                                 so_lan=int(cm[i, j]),
                                 ti_le=float(cm[i, j] / max(cm[i].sum(), 1))))
    df = pd.DataFrame(rows).sort_values("so_lan", ascending=False).head(k)
    df.to_csv(out / "top_confusions.csv", index=False)
    return df


def confidence_curve(y, pred, prob, out: Path):
    """Độ chính xác và độ phủ theo ngưỡng tin cậy.

    Bảng này dùng để CHỌN `conf_threshold` cho streaming một cách có cơ sở, thay
    vì đoán. Chọn ngưỡng sao cho precision đủ cao ở mức coverage chấp nhận được:
    trong ứng dụng dịch, một từ sai gây hại hơn một từ bị bỏ qua.
    """
    conf = prob.max(axis=1)
    rows = []
    for t in np.arange(0.0, 0.96, 0.05):
        m = conf >= t
        if m.sum() == 0:
            break
        rows.append(dict(nguong=round(float(t), 2),
                         do_phu=float(m.mean()),
                         do_chinh_xac=float((y[m] == pred[m]).mean()),
                         so_mau=int(m.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(out / "confidence_curve.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--compare", nargs="*", default=[])
    args = ap.parse_args()

    if args.compare:
        rows = []
        for r in args.compare:
            p = Path(r)
            m = json.loads((p / "test_metrics.json").read_text(encoding="utf-8"))
            rows.append(dict(run=p.name, cach_chia=m.get("split_mode", "?"),
                             acc=round(m["acc"], 4),
                             macro_f1=round(m["macro_f1"], 4),
                             top5=round(m["top5"], 4)))
        df = pd.DataFrame(rows)
        print("\nBẢNG SO SÁNH CÁCH CHIA TẬP")
        print(df.to_string(index=False))
        print("\nKhoảng cách giữa dòng 'random' và dòng 'signer' chính là mức độ mà")
        print("một phép chia sai sẽ thổi phồng kết quả. Đưa bảng này vào báo cáo.")
        return

    if not args.run:
        ap.error("cần --run hoặc --compare")

    run = Path(args.run)
    y, pred, prob, idx_to_label = load_run(run)

    print(f"\nSố mẫu test: {len(y)}   Số lớp: {len(set(y.tolist()))}")
    print(f"Accuracy   : {(y == pred).mean():.4f}")

    df_cls = per_class(y, pred, idx_to_label, run)
    print("\n10 lớp YẾU NHẤT (nơi đáng đi tìm nguyên nhân):")
    print(df_cls.head(10).to_string(index=False))

    df_conf = top_confusions(y, pred, idx_to_label, run)
    print("\n10 cặp NHẦM LẪN nhiều nhất:")
    print(df_conf.head(10).to_string(index=False))
    print("\nHãy mở video của vài cặp này ra xem. Thường chúng khác nhau chỉ ở")
    print("handshape hoặc ở nét mặt — và đó là kết luận cụ thể cho báo cáo.")

    df_cv = confidence_curve(y, pred, prob, run)
    print("\nĐỘ CHÍNH XÁC THEO NGƯỠNG TIN CẬY:")
    print(df_cv.to_string(index=False))

    print(f"\nĐã lưu CSV vào {run}/")


if __name__ == "__main__":
    main()
