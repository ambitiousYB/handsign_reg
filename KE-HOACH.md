# Kế hoạch thực hiện — VSL400 (bản Cropped trên Kaggle)

Đây là tài liệu duy nhất cần đọc để biết làm gì, theo thứ tự nào, chạy file nào.
Mọi phiên bản kế hoạch trước đó (bộ PTIT, `--split-mode group`) đã lỗi thời.

---

## Dữ liệu

**Dataset:** `nguyenanfms/vsl-vietnamese-sign-language-v2` trên Kaggle — chính là
VSL400 kèm một phần video thu thập từ internet.

**Chỉ dùng nhánh `processed/processed/frame_splited/`** (khoảng 10 GB): video
front-view đã cắt 224×224, chia sẵn `train/{gloss}/` và `test/{gloss}/`. Nhánh
`raw/` 72 GB không dùng tới. Kaggle mount dataset ở chế độ chỉ đọc nên không tốn
dung lượng hay băng thông của bạn.

**Phép chia có sẵn là tài sản quý nhất.** Phần VSL400 đã chia train/test theo
signer ID — người ký trong test chưa từng xuất hiện khi huấn luyện. Luôn dùng
`--split-mode provided` để giữ nguyên phép chia này.

**Không dùng keypoint `.npy` có sẵn.** Chúng không có điểm mặt và chuẩn hoá theo
bounding box, khác với `landmarks.py`. Ta trích lại từ video bằng pipeline của
mình để khâu huấn luyện và khâu webcam nhất quán tuyệt đối.

**Từ vựng:** 150 gloss nhiều mẫu nhất trong 472. Đủ phong phú để bài có trọng
lượng, đủ gọn cho demo real-time và để viết câu test.

**Dữ liệu tự quay — vẫn bắt buộc:** 40 clip nghỉ cho lớp `__NOSIGN__`, và 15–20
câu liên tục để đo WER streaming. Không dataset công khai nào thay thế được hai thứ này.

---

## Bản đồ: chạy gì, ở đâu

| Giai đoạn | Chạy ở | File | Sản phẩm |
|---|---|---|---|
| 1. Chuẩn bị | Laptop | `src/selftest.py` | xác nhận code chạy đúng |
| 2. Trích xuất | Kaggle CPU | `notebooks/kaggle_01_extract.py` | dataset `vsl400-landmarks` |
| 3. Huấn luyện | Kaggle GPU | `notebooks/kaggle_02_train.py` | `best.pt`, bảng kết quả |
| 4. Quay dữ liệu | Laptop | `src/record_clips.py` | clip nghỉ, câu liên tục |
| 5. Huấn luyện lần cuối | Kaggle CPU rồi GPU | `extract_dataset.py`, `merge_manifests.py`, `kaggle_02_train.py` | `vsl.onnx` có lớp nghỉ |
| 6. Demo và đo đạc | Laptop | `src/stream.py`, `src/eval_streaming.py` | WER, độ trễ, báo động giả |
| 7. Báo cáo | — | — | báo cáo, slide |

---

## Giai đoạn 1 — Chuẩn bị (nửa ngày)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/selftest.py                              # phải ra "TẤT CẢ ĐỀU ĐẠT"
```

Trên Kaggle: upload thư mục `src/` thành một Dataset tên **`vsl-code`**. Giữ
nguyên tên thư mục `src/` — các notebook tìm code ở `/kaggle/input/*/src`. Mỗi lần
sửa code thì cập nhật phiên bản dataset này.

---

## Giai đoạn 2 — Trích xuất landmark (Kaggle CPU, khoảng 1 giờ)

Notebook **`kaggle_01_extract.py`**. Accelerator = **None**, Internet = On.
Input: `vsl-code` và `nguyenanfms/vsl-vietnamese-sign-language-v2`.

Notebook tự làm, từ trên xuống:

1. Tìm thư mục `frame_splited/train` và `frame_splited/test`
2. Lọc 150 gloss nhiều mẫu nhất, ghi ra `keep_labels.txt`
3. Trích riêng `train/` với `--split-tag train`, rồi `test/` với `--split-tag test`
4. Gộp bằng `merge_manifests.py` thành một `manifest.csv` duy nhất
5. In bảng kiểm tra sức khoẻ dữ liệu

**Đọc kỹ bảng kiểm tra.** Hai con số cần nhìn: số gloss có trong train mà thiếu
trong test (lớp đó sẽ không được chấm), và số clip phát hiện pose kém.

Cuối notebook: tab Output → **New Dataset** → đặt tên **`vsl400-landmarks`**.
Tải về máy luôn file `keep_labels.txt` — cần ở giai đoạn 4.

Script trích xuất chạy tiếp được: nếu phiên bị ngắt, chạy lại đúng ô đó, những
clip đã xong sẽ được bỏ qua.

---

## Giai đoạn 3 — Huấn luyện (Kaggle GPU)

Notebook **`kaggle_02_train.py`**. Accelerator = **GPU P100**, Internet = On.
Input: `vsl-code` và `vsl400-landmarks`.

Notebook tự kiểm tra manifest trước khi tốn giờ GPU: có cột `split`, có đủ train
lẫn test, đường dẫn `.npz` tồn tại thật. Sai một trong ba là dừng với thông báo rõ.

### Thí nghiệm chính — bảng quan trọng nhất của báo cáo

```bash
python train.py --split-mode provided --model bilstm --epochs 80 --out runs/bilstm_provided
python train.py --split-mode random   --model bilstm --epochs 80 --out runs/bilstm_random
python evaluate.py --compare runs/bilstm_random runs/bilstm_provided
```

| Cách chia | Ý nghĩa |
|---|---|
| `random` | cùng một người ký nằm ở cả train và test — con số bị thổi phồng |
| `provided` | người ký trong test chưa từng thấy — con số TRUNG THỰC |

Khoảng cách giữa hai dòng là kết quả đầu tiên đáng viết vào báo cáo. Nó đo bằng
chính thực nghiệm của bạn mức độ một phép chia sai làm đẹp con số.

### Phân tích lỗi

```bash
python evaluate.py --run runs/bilstm_provided
```

Mở `top_confusions.csv`, chọn 5 cặp nhầm nhiều nhất, tìm video tương ứng trong
`frame_splited/test/<gloss>/` của dataset gốc và **xem tận mắt**. Một kết luận cụ
thể về vì sao hai ký hiệu bị nhầm có giá trị hơn nhiều trang bảng số.

`confidence_curve.csv` dùng để chọn ngưỡng `--conf` cho giai đoạn 6.

### Thí nghiệm phụ

So sánh kiến trúc (Transformer, và ST-GCN nếu làm), bảng ablation đặc trưng: bỏ
mặt, bỏ tay cục bộ, bỏ vận tốc, chỉ một tay. Tất cả đều dùng `--split-mode provided`.

---

## Giai đoạn 4 — Quay dữ liệu (một buổi chiều)

**Việc phải làm TRƯỚC khi quay:** viết lại `configs/test_sentences.txt` **chỉ bằng
gloss có trong `keep_labels.txt`**. Câu chứa ký hiệu ngoài 150 gloss thì hệ thống
không thể đúng, và quay xong mới phát hiện là mất cả buổi.

```bash
# 40 clip nghỉ: ngồi yên, gãi đầu, uống nước, chỉnh tóc, nói chuyện
python src/record_clips.py --mode idle --out data/raw/__NOSIGN__ --signer B01 -n 40

# 15-20 câu liên tục, lặp lại cho từng người B01, B02, B03
python src/record_clips.py --mode sentence --out data/continuous \
    --signer B01 --script configs/test_sentences.txt
```

Phòng đủ sáng, nền đơn giản, người ngồi cách webcam sao cho thấy từ đầu tới thắt
lưng — gần giống khung hình của video VSL400 đã cắt.

---

## Giai đoạn 5 — Huấn luyện lần cuối có lớp nghỉ

Upload thư mục `data/raw/__NOSIGN__` lên Kaggle thành dataset `vsl-idle`, rồi
trong một notebook CPU:

```bash
python extract_dataset.py --root /kaggle/input/vsl-idle --out /kaggle/working/idle \
    --workers 2 --group-regex '' --signer-regex '(B\d+)'
```

**Không gắn `--split-tag`** cho clip nghỉ. Ô `split` để trống thì chế độ
`provided` tự xếp chúng vào train — đúng như mong muốn.

```bash
python merge_manifests.py \
    --inputs /kaggle/working/landmarks/manifest.csv /kaggle/working/idle/manifest.csv \
    --out /kaggle/working/all_manifest.csv

python train.py --manifest /kaggle/working/all_manifest.csv \
    --split-mode provided --model bilstm --epochs 80 \
    --nosign-ratio 0.06 --out runs/final

python export_onnx.py --ckpt runs/final/best.pt --out models/vsl.onnx
```

Hạ `--nosign-ratio` xuống 0.06 vì đã có negative thật. Tải `vsl.onnx` và
`vsl.labels.json` về laptop.

---

## Giai đoạn 6 — Demo và đo đạc (laptop)

```bash
python src/stream.py --ckpt models/vsl.onnx --backend onnx --model-complexity 0
```

Căn ngưỡng bằng thanh năng lượng trên màn hình:

| Triệu chứng | Sửa |
|---|---|
| Đứng yên mà thanh đã vượt vạch đỏ | tăng `--motion-hi` |
| Ký hiệu chậm mà máy không kích hoạt | giảm `--motion-hi` |
| Phun gloss liên tục | tăng `--conf`, kiểm tra lớp `__NOSIGN__` có trong nhãn không |
| Dưới 10 FPS | giữ `--model-complexity 0`, giảm độ phân giải |

Rồi đo con số thật:

```bash
python src/eval_streaming.py --ckpt models/vsl.onnx \
    --videos data/continuous --refs configs/test_sentences.txt \
    --idle-videos data/raw/__NOSIGN__ --out runs/streaming_eval.json
```

Chạy ở vài mức `--conf` khác nhau để dựng đường cong đánh đổi giữa WER và báo động giả.

---

## Giai đoạn 7 — Báo cáo

1. Bài toán và phạm vi — vì sao nhận diện ký hiệu rời ở chế độ streaming
2. Dữ liệu — VSL400, phép chia theo người ký, lý do trích lại landmark
3. Phương pháp — pipeline, đặc trưng 370 chiều, kiến trúc
4. Kết quả offline — bảng random so với provided, so sánh kiến trúc, ablation
5. Kết quả streaming — WER, độ trễ, báo động giả
6. Phân tích lỗi — cặp nhầm lẫn kèm ảnh khung hình thật
7. Giới hạn

---

## Những con số phải có

| Chỉ số | Lấy từ |
|---|---|
| Accuracy, macro-F1 — chia ngẫu nhiên | `runs/bilstm_random/test_metrics.json` |
| Accuracy, macro-F1 — người ký chưa thấy | `runs/bilstm_provided/test_metrics.json` |
| So sánh kiến trúc | `evaluate.py --compare` |
| Đóng góp từng nhóm đặc trưng | bảng ablation |
| WER, độ trễ, báo động giả | `runs/streaming_eval.json` |
| FPS trên CPU laptop | góc trên `stream.py` |

---

## Giới hạn phải nêu thẳng

1. Nhận diện **từng ký hiệu rời**, không phải dịch liên tục. Ký hiệu nối liền
   không nghỉ sẽ bị gộp hoặc bỏ sót.
2. Tập test của dataset trộn video VSL400 (chia theo người ký) với video lấy từ
   internet (chỉ chia theo gloss). Phần internet không đảm bảo người ký chưa thấy.
3. Mô-đun gloss→tiếng Việt dựa trên luật, chỉ phủ mẫu câu đơn giản.
4. Từ vựng cố định 150 gloss; ký hiệu ngoài tập sẽ bị gán vào lớp gần nhất.
5. Landmark 2D mất chiều sâu; ký hiệu chỉ khác hướng lòng bàn tay dễ bị nhầm.
6. Dữ liệu VSL400 quay trong studio, webcam thật kém hơn nhiều. Chênh lệch giữa
   accuracy offline và WER streaming phản ánh đúng khoảng cách này.

---

## Nếu chậm tiến độ

Cắt theo thứ tự, ít đau nhất trước:

1. Bỏ Transformer và ST-GCN, chỉ giữ BiLSTM
2. Rút ablation còn 2 cấu hình
3. Giảm câu test còn 10, người quay còn 2
4. Giảm từ vựng xuống 80–100 gloss

**Không bao giờ cắt:** bảng random so với provided, và phép đo WER streaming.
