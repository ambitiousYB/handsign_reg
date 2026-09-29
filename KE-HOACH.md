# Kế hoạch thực hiện — VSL400 (bản Cropped trên Kaggle)

Đây là tài liệu duy nhất cần đọc để biết làm gì, theo thứ tự nào, chạy file nào.
Mọi phiên bản kế hoạch trước đó (bộ PTIT, `--split-mode group`) đã lỗi thời.
Phân vai D1–D4 xem `PHAN-CONG.md`.

---

## Trạng thái hiện tại (cập nhật 29/09/2026)

| Giai đoạn | Trạng thái |
|---|---|
| 1. Chuẩn bị | ✅ Xong |
| 2. Trích xuất landmark | ✅ Xong — dataset `vsl400-landmarks` |
| 3. Huấn luyện | ✅ Xong — BiLSTM provided/random, Transformer provided |
| — Mô hình chạy trên laptop | ✅ `models/vsl.onnx` (BiLSTM provided) chạy được bằng `stream.py` |
| 4. Quay dữ liệu | ⏳ Chưa — **chờ chốt chuyện lật ảnh (việc A2)** |
| 5. Huấn luyện lần cuối | ⏳ Chưa — cần viết `notebooks/kaggle_03_final.py` |
| 6. Demo và đo streaming | ⏳ Chưa |
| 7. Báo cáo | ⏳ Chưa |

### Kết quả offline đã có (150 gloss, tập test 2.162 clip)

| Run | Accuracy | Macro-F1 | Top-5 |
|---|---|---|---|
| BiLSTM — `random` | 84,9% | 84,1% | 97,0% |
| BiLSTM — `provided` | 87,5% | 87,1% | 97,6% |
| **Transformer — `provided`** | **89,4%** | **89,6%** | 97,5% |

**Ba phát hiện phải xử lý trước khi viết báo cáo:**

1. **`random` THẤP hơn `provided`** — ngược với dự đoán "chia ngẫu nhiên thổi
   phồng kết quả". Hai bên có lượng dữ liệu train gần bằng nhau (~7.500 và ~7.400
   clip). Giả thuyết: tập test gốc của dataset dễ hơn phần train gốc (vd phần train
   lẫn video internet). Phải kiểm chứng (việc A3) rồi mới viết được phần này.
2. **Transformer hơn BiLSTM ~2,5 điểm macro-F1** → dùng Transformer cho mô hình cuối.
3. **Cặp nhầm nặng nhất** (BiLSTM provided, `top_confusions.csv`):
   `Con heo → Dơ` (10/15 clip), `Dơ → Con heo` (7/15), `Thứ sáu → Màu vàng` (6),
   `Nếm → Dữ` (5), `Thú vị → Rửa tay` (5). Gloss yếu nhất: Con heo (F1 0,37),
   Dơ (0,42), Chú ý (0,46), Nếm (0,57). Tránh các gloss này khi demo.

**Chọn ngưỡng `--conf`** (từ `confidence_curve.csv`, BiLSTM provided):

| `--conf` | Tỉ lệ clip được trả lời | Độ chính xác trên số đó |
|---|---|---|
| 0.5 | 94% | 90,7% |
| **0.6** | **90%** | **92,2%** |
| 0.7 | 85% | 93,3% |
| 0.8 | 79% | 94,8% |

---

## VIỆC CẦN LÀM — theo thứ tự

### A. Ngay bây giờ (song song được)

- [ ] **A1 — Đồng bộ code** (D1)
  - Commit + push toàn bộ thay đổi lên GitHub (`main`).
  - Kaggle: dataset `vsl-code` → **New Version** → upload lại toàn bộ `src/`.
    Code trên Kaggle đang là bản cũ, thiếu các sửa lỗi ở `export_onnx.py`,
    `extract_dataset.py`, `eval_streaming.py`, `stream.py`.

- [ ] **A2 — Thử demo webcam, chốt chuyện lật ảnh** (D3) — **CHẶN buổi quay**
  1. Ngồi lùi xa để thấy **cả hai vai, cả hai tay, tới thắt lưng**; thêm đèn chiếu
     từ phía trước (ảnh nhiễu hạt = thiếu sáng = MediaPipe bắt tay kém).
  2. Chạy `python src/stream.py --ckpt models/vsl.onnx --conf 0.5`. Ngồi yên thì số
     `E=` góc phải phải nhỏ hơn `lo` và trạng thái là IDLE; khi ký thì `E` vượt `hi`.
     Chưa đúng thì chỉnh `--motion-hi` / `--motion-lo`.
  3. Ký 5–8 gloss dễ (Cảm ơn, Xin lỗi, Ăn, Uống, Bố, Nhà, Đi), mỗi gloss 3 lần,
     với **hai lệnh**: không cờ và có `--no-flip`. Lệnh nào đúng nhiều hơn rõ rệt
     là chiều ảnh khớp dữ liệu huấn luyện.
  4. Theo kết quả, sửa `record_clips.py` cho clip quay ra đúng chiều đó (hiện nó
     lật gương rồi mới ghi file) và đặt mặc định tương ứng trong `stream.py`.

- [ ] **A3 — Chẩn đoán random < provided** (D2)
  - Chạy ô "Chẩn đoán" trong `kaggle_02_train.py` (ngay sau ô `--compare`). Ô này
    cần `runs/bilstm_random/test_preds.npz` trong phiên → nếu phiên cũ đã tắt thì
    chạy lại ô train `random` trước (~20–40 phút GPU).
  - Nếu clip gốc "test" có acc cao hơn hẳn clip gốc "train" → tập test gốc dễ hơn;
    báo cáo phải diễn giải lại bảng random/provided theo hướng đó.

- [ ] **A4 — Phân tích lỗi** (D4)
  - Mở video trong `frame_splited/test/<gloss>/` của 5 cặp nhầm ở trên, xem tận
    mắt vì sao nhầm (hình dạng tay giống nhau? chỉ khác hướng lòng bàn tay?).
    Chụp khung hình minh hoạ cho báo cáo.

### B. Buổi quay — Giai đoạn 4 (sau A2)

- [ ] **B1** Đặt lịch, phòng đủ sáng, nền đơn giản, webcam cố định. Cả 4 người tham gia.
- [ ] **B2** 40 clip nghỉ `__NOSIGN__` (ngồi yên, gãi đầu, uống nước, chỉnh tóc, nói chuyện).
- [ ] **B3** 18 câu trong `configs/test_sentences.txt` × ít nhất 3 người (B01, B02, B03).
  Câu và từ điển `configs/gloss_vi.json` **đã viết xong**, chỉ dùng gloss trong 150 gloss.
- [ ] **B4** Xem lại clip: clip hỏng, clip thiếu tay → quay bù ngay trong buổi.

### C. Huấn luyện lần cuối — Giai đoạn 5 (sau B)

- [ ] **C1** Viết `notebooks/kaggle_03_final.py` (chưa có): trích landmark clip nghỉ,
  gộp manifest, huấn luyện **Transformer** có lớp nghỉ thật, xuất ONNX.
- [ ] **C2** Upload clip nghỉ lên Kaggle thành dataset `vsl-idle`. **Upload cả thư
  mục `data/raw/`** (hoặc file zip) để giữ thư mục con `__NOSIGN__/` — script trích
  xuất lấy tên nhãn từ tên thư mục.
- [ ] **C3** Chạy notebook 03, tải `vsl.onnx` + `vsl.labels.json` về `models/`.

### D. Demo và đo streaming — Giai đoạn 6 (sau C) — KHÔNG ĐƯỢC CẮT

- [ ] **D1** Căn ngưỡng `--motion-hi`, `--motion-lo`, `--conf` trên mô hình cuối.
- [ ] **D2** Chạy `eval_streaming.py` ở các mức `--conf` 0.4 / 0.5 / 0.6 / 0.7 → đường
  cong WER – báo động giả.
- [ ] **D3** Ghi FPS trên CPU laptop, độ trễ trung bình.
- [ ] **D4** Quay video demo dự phòng (phòng webcam hỏng lúc bảo vệ).

### E. Thí nghiệm phụ (cắt được nếu chậm)

- [ ] **E1 — Ablation**: bỏ mặt / bỏ tay cục bộ / bỏ vận tốc / chỉ một tay.
  **Chưa có code**: `train.py` chưa có cờ che nhóm đặc trưng → D2 phải thêm (vd
  `--drop-features face,local,velocity`). Chạy song song trên 4 tài khoản Kaggle.
- [ ] **E2 — ST-GCN**: **chưa có** trong `models.py` (`--model` chỉ nhận
  `bilstm`, `transformer`). Chỉ làm nếu còn thời gian.

### F. Báo cáo — Giai đoạn 7

- [ ] Viết theo dàn ý ở mục "Giai đoạn 7" bên dưới, dùng bảng "Những con số phải có".
- [ ] Slide bảo vệ, tập trả lời 3 câu hỏi khó (xem `PHAN-CONG.md`).

---

## Dữ liệu

**Dataset:** `nguyenanfms/vsl-vietnamese-sign-language-v2` trên Kaggle — chính là
VSL400 kèm một phần video thu thập từ internet.

**Chỉ dùng nhánh `processed/processed/frame_splited/`** (khoảng 10 GB): video
front-view đã cắt 224×224, chia sẵn `train/{gloss}/` và `test/{gloss}/`. Nhánh
`raw/` 72 GB và `processed_augmented/` KHÔNG dùng — bản augmented là biến thể của
cùng clip, dùng vào sẽ lệch số liệu.

**Phép chia có sẵn.** Theo mô tả dataset, phần VSL400 đã chia train/test theo
signer ID. Luôn dùng `--split-mode provided`. (Xem phát hiện 1 ở trên — cần kiểm
chứng mức độ khó của tập test này.)

**Không dùng keypoint `.npy` có sẵn** (`keypoints_splited/`). Chúng không có điểm
mặt và chuẩn hoá theo bounding box, khác với `landmarks.py`. Ta trích lại từ video
để khâu huấn luyện và khâu webcam nhất quán tuyệt đối.

**Từ vựng:** 150 gloss nhiều mẫu nhất trong 472 (`keep_labels.txt`). Ít nhất 62
clip mỗi gloss. Từ vựng KHÔNG có TÔI, BẠN, KHÔNG, GÌ… — câu dùng từ xưng hô (EM,
ANH, BỐ…) làm chủ ngữ.

**Số liệu trích xuất thực tế:** 8.664/8.667 clip train, 2.162 clip test, tỉ lệ
thấy tay trái 0,85 / phải 0,79. Mất **~7,3 giờ** trên Kaggle CPU (không phải 1 giờ).

---

## Bản đồ: chạy gì, ở đâu

| Giai đoạn | Chạy ở | File | Sản phẩm |
|---|---|---|---|
| 1. Chuẩn bị | Laptop | `src/selftest.py` | xác nhận code chạy đúng |
| 2. Trích xuất | Kaggle CPU | `notebooks/kaggle_01_extract.py` | dataset `vsl400-landmarks` |
| 3. Huấn luyện | Kaggle GPU T4 | `notebooks/kaggle_02_train.py` | `best.pt`, bảng kết quả |
| 4. Quay dữ liệu | Laptop | `src/record_clips.py` | clip nghỉ, câu liên tục |
| 5. Huấn luyện lần cuối | Kaggle CPU rồi GPU | `notebooks/kaggle_03_final.py` (chưa viết) | `vsl.onnx` có lớp nghỉ thật |
| 6. Demo và đo đạc | Laptop | `src/stream.py`, `src/eval_streaming.py` | WER, độ trễ, báo động giả |
| 7. Báo cáo | — | — | báo cáo, slide |

---

## Giai đoạn 1 — Chuẩn bị môi trường laptop

```bash
python -m venv .venv
.venv\Scripts\activate                      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install torch                           # bản CPU; chỉ cần nếu xuất ONNX trên máy
python src/selftest.py                      # phải ra "TẤT CẢ ĐỀU ĐẠT"
```

**Bẫy protobuf:** `mediapipe==0.10.14` cần `protobuf<5`, còn gói `onnx` cần
`protobuf>=6`. Demo cần mediapipe → giữ `protobuf==4.25.9`. Nếu phải xuất ONNX trên
máy mà lỗi: `pip install -U protobuf` → xuất → `pip install protobuf==4.25.9`.

Trên Kaggle: upload thư mục `src/` thành Dataset **`vsl-code`**. Notebook tự dò code
theo tên file nên Kaggle có giữ thư mục `src/` hay không đều được. **Mỗi lần sửa
code phải tạo New Version**, nếu không notebook vẫn chạy code cũ.

---

## Giai đoạn 2 — Trích xuất landmark (Kaggle CPU) — ĐÃ XONG

Notebook **`kaggle_01_extract.py`**. Accelerator = **None**, Internet = On.
Input: `vsl-code` và `nguyenanfms/vsl-vietnamese-sign-language-v2`.

1. Cài `mediapipe==0.10.14` rồi **gỡ tensorflow** (xem "Lỗi đã gặp")
2. Tự dò `processed/frame_splited/` (bỏ qua bản augmented và `raw/`)
3. Lọc 150 gloss nhiều mẫu nhất → `keep_labels.txt`
4. Trích `train/` với `--split-tag train`, `test/` với `--split-tag test`
5. Gộp bằng `merge_manifests.py` → một `manifest.csv`
6. Tab Output → **New Dataset** → `vsl400-landmarks`

Script trích xuất chạy tiếp được: phiên bị ngắt thì chạy lại đúng ô đó. Phiên dài
nên dùng **Save Version → Save & Run All** (chạy nền tối đa 12 giờ).

---

## Giai đoạn 3 — Huấn luyện (Kaggle GPU) — ĐÃ XONG

Notebook **`kaggle_02_train.py`**. Accelerator = **GPU T4** (KHÔNG dùng P100 —
PyTorch mới trên Kaggle đã bỏ hỗ trợ). Input: **chỉ** `vsl-code` và
`vsl400-landmarks` (không add dataset video 82 GB).

```bash
python train.py --split-mode provided --model bilstm --epochs 80 --out runs/bilstm_provided
python train.py --split-mode random   --model bilstm --epochs 80 --out runs/bilstm_random
python train.py --split-mode provided --model transformer --epochs 80 --layers 4 --lr 2e-4 \
    --out runs/transformer_provided
python evaluate.py --compare runs/bilstm_random runs/bilstm_provided
python evaluate.py --run runs/bilstm_provided
python export_onnx.py --ckpt runs/bilstm_provided/best.pt --out models/vsl.onnx
```

**File trong mỗi `runs/<tên>/`:**

| File | Dùng để | Cần tải về |
|---|---|---|
| `test_metrics.json` | acc, macro-F1, top-5 cho bảng chính | Có (cả 3 run) |
| `best.pt` | mô hình tốt nhất theo macro-F1 val | Chỉ run dùng làm demo |
| `last.pt` | chạy tiếp khi bị ngắt | Không |
| `confidence_curve.csv` | chọn `--conf` | Có |
| `top_confusions.csv`, `per_class.csv` | phân tích lỗi | Có |
| `history.csv`, `train_config.json` | đường cong huấn luyện, tái lập | Nên có |
| `test_preds.npz` | ma trận nhầm lẫn, chẩn đoán A3 | Tuỳ |

**Tải `.pt` về bị thành thư mục?** File `.pt` là zip, trình duyệt có thể tự giải
nén. Tải cả output dạng `.zip` (Download All), hoặc nén lại thư mục đó bằng zip
KHÔNG nén (`ZIP_STORED`) với một thư mục gốc chung.

---

## Giai đoạn 4 — Quay dữ liệu (một buổi chiều, sau việc A2)

```bash
# 40 clip nghỉ: ngồi yên, gãi đầu, uống nước, chỉnh tóc, nói chuyện
python src/record_clips.py --mode idle --out data/raw/__NOSIGN__ --signer B01 -n 40

# 18 câu liên tục, lặp lại cho từng người B01, B02, B03
python src/record_clips.py --mode sentence --out data/continuous \
    --signer B01 --script configs/test_sentences.txt
```

Ngồi sao cho webcam thấy từ đầu tới thắt lưng, cả hai tay — gần giống khung hình
video VSL400. Phòng đủ sáng. Thứ tự dòng trong `test_sentences.txt` = thứ tự quay:
`eval_streaming.py` ghép video thứ i với câu thứ i theo thứ tự tên file.

---

## Giai đoạn 5 — Huấn luyện lần cuối có lớp nghỉ thật

Trong `kaggle_03_final.py` (cần viết — việc C1), trên Kaggle:

```bash
python extract_dataset.py --root <thư mục chứa __NOSIGN__/> --out /kaggle/working/idle \
    --workers 2 --group-regex '' --signer-regex '(B\d+)'
```

**Không gắn `--split-tag`** cho clip nghỉ: ô `split` trống thì chế độ `provided` tự
xếp chúng vào train. Landmark VSL400 phải được giải nén về đúng
`/kaggle/working/landmarks/` (manifest ghi đường dẫn tuyệt đối) — làm như ô "Đưa
landmark về đúng chỗ" của notebook 02.

```bash
python merge_manifests.py \
    --inputs /kaggle/working/landmarks/manifest.csv /kaggle/working/idle/manifest.csv \
    --out /kaggle/working/all_manifest.csv

python train.py --manifest /kaggle/working/all_manifest.csv \
    --split-mode provided --model transformer --epochs 80 --layers 4 --lr 2e-4 \
    --nosign-ratio 0.06 --out runs/final

python export_onnx.py --ckpt runs/final/best.pt --out models/vsl.onnx
```

Hạ `--nosign-ratio` xuống 0.06 vì đã có negative thật. Kiểm tra dòng
`Sai lệch lớn nhất PyTorch vs ONNX` nhỏ hơn 1e-3 (Transformer chưa từng xuất ONNX
trong dự án này — nếu lỗi, dùng BiLSTM cho demo và ghi chú trong báo cáo).

---

## Giai đoạn 6 — Demo và đo đạc (laptop)

```bash
python src/stream.py --ckpt models/vsl.onnx --conf 0.6
```

Phím: `q` thoát, `c` xoá câu. Góc phải hiện `E=` (năng lượng chuyển động) cùng
ngưỡng `hi`/`lo` để căn chỉnh. Thêm `--no-flip` nếu việc A2 kết luận như vậy.

| Triệu chứng | Sửa |
|---|---|
| Ngồi yên mà `E` vẫn vượt `hi` | tăng `--motion-hi` (và `--motion-lo`); kiểm tra ánh sáng |
| Ký chậm mà máy không kích hoạt | giảm `--motion-hi` |
| Phun gloss liên tục | tăng `--conf`; kiểm tra lớp `__NOSIGN__` có trong nhãn |
| Nhận sai nhiều dù `E` ổn | ngồi xa hơn cho thấy tới thắt lưng; thử `--no-flip` |
| Dưới 10 FPS | giữ `--model-complexity 0`, giảm độ phân giải |

Đo con số thật:

```bash
python src/eval_streaming.py --ckpt models/vsl.onnx \
    --videos data/continuous --refs configs/test_sentences.txt \
    --idle-videos data/raw/__NOSIGN__ --out runs/streaming_eval.json --conf 0.6
```

Chạy ở vài mức `--conf` để dựng đường cong đánh đổi giữa WER và báo động giả.

---

## Giai đoạn 7 — Báo cáo

1. Bài toán và phạm vi — vì sao nhận diện ký hiệu rời ở chế độ streaming
2. Dữ liệu — VSL400, phép chia có sẵn, lý do trích lại landmark
3. Phương pháp — pipeline, đặc trưng, kiến trúc BiLSTM / Transformer
4. Kết quả offline — bảng random / provided (kèm chẩn đoán A3), so sánh kiến trúc, ablation
5. Kết quả streaming — WER, độ trễ, báo động giả
6. Phân tích lỗi — 5 cặp nhầm kèm ảnh khung hình thật
7. Giới hạn

---

## Những con số phải có

| Chỉ số | Lấy từ | Có chưa |
|---|---|---|
| Accuracy, macro-F1 — chia ngẫu nhiên | `runs/bilstm_random/test_metrics.json` | ✅ |
| Accuracy, macro-F1 — phép chia có sẵn | `runs/bilstm_provided/test_metrics.json` | ✅ |
| So sánh kiến trúc | `runs/transformer_provided/test_metrics.json` | ✅ |
| Giải thích random < provided | ô chẩn đoán notebook 02 | ⏳ A3 |
| Đóng góp từng nhóm đặc trưng | bảng ablation | ⏳ E1 |
| WER, độ trễ, báo động giả | `runs/streaming_eval.json` | ⏳ D2 |
| FPS trên CPU laptop | góc trên `stream.py` | ⏳ D3 |

---

## Lỗi đã gặp và cách xử lý

| Triệu chứng | Nguyên nhân | Cách xử lý (đã có trong code) |
|---|---|---|
| `SystemExit: Upload thư mục src/...` | Kaggle đặt file dataset ở gốc, không giữ thư mục `src/` | notebook dò code theo tên file |
| `Không thấy frame_splited` | đường mount Kaggle khác (`/kaggle/input/datasets/<owner>/<slug>/`) | notebook tự dò, chỉ lấy `processed/` |
| `ImportError: runtime_version from google.protobuf` | mediapipe hạ protobuf → tensorflow có sẵn của Kaggle import hỏng | `pip uninstall -y tensorflow` trong notebook 01 |
| Log traceback lặp vô hạn khi trích xuất | initializer của `Pool` lỗi → Pool tạo lại worker mãi | `extract_dataset.py` thử MediaPipe trước, lỗi thì dừng |
| `no kernel image is available` | GPU P100 không còn được PyTorch hỗ trợ | dùng GPU T4 |
| Không có `vsl.onnx` sau khi export | PyTorch ≥ 2.9 mặc định exporter dynamo (cần `onnxscript`) | `export_onnx.py` ép `dynamo=False` |
| `best.pt` tải về thành thư mục | `.pt` là zip, trình duyệt tự giải nén | tải dạng zip, hoặc nén lại |
| `pip install torch` lỗi `flit_core` trên laptop | pip cũ + index của PyTorch | `pip install torch` từ PyPI (Windows = bản CPU) |
| WER luôn ~100% | câu tham chiếu `XE-ĐẠP` ≠ nhãn mô hình `Xe đạp` | `eval_streaming.py` chuẩn hoá cả hai |
| Chữ tiếng Việt thành `??` trên màn hình | `cv2.putText` không có dấu | vẽ bằng Pillow |
| Thanh năng lượng luôn đầy khi ngồi yên | cổ tay ngoài khung hình, MediaPipe đoán vị trí nhảy lung tung | chỉ tính cổ tay nhìn thấy (`pose_vis > 0.5`) |

---

## Giới hạn phải nêu thẳng

1. Nhận diện **từng ký hiệu rời**, không phải dịch liên tục. Ký hiệu nối liền
   không nghỉ sẽ bị gộp hoặc bỏ sót.
2. Tập test của dataset trộn video VSL400 với video lấy từ internet. Phần internet
   không đảm bảo người ký chưa thấy. Thực nghiệm cho thấy chia ngẫu nhiên KHÔNG cho
   kết quả cao hơn — nêu rõ và giải thích bằng kết quả chẩn đoán A3.
3. Mô-đun gloss→tiếng Việt dựa trên luật, chỉ phủ mẫu câu đơn giản.
4. Từ vựng cố định 150 gloss, thiếu đại từ TÔI/BẠN và từ phủ định/nghi vấn; ký
   hiệu ngoài tập sẽ bị gán vào lớp gần nhất.
5. Landmark 2D mất chiều sâu; ký hiệu chỉ khác hướng lòng bàn tay dễ bị nhầm
   (vd Con heo ↔ Dơ).
6. Dữ liệu VSL400 quay trong studio, webcam thật kém hơn nhiều. Chênh lệch giữa
   accuracy offline và WER streaming phản ánh đúng khoảng cách này.

---

## Nếu chậm tiến độ

Cắt theo thứ tự, ít đau nhất trước:

1. Bỏ ST-GCN (E2)
2. Rút ablation còn 2 cấu hình (E1)
3. Giảm câu test còn 10, người quay còn 2
4. Dùng BiLSTM thay Transformer cho mô hình cuối

**Không bao giờ cắt:** bảng random so với provided (kèm giải thích), và phép đo
WER streaming.
