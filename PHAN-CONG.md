# Phân công nhiệm vụ — nhóm 4 người

Đi kèm `KE-HOACH.md` (dataset VSL400 bản Cropped). Bản này trả lời câu hỏi *ai làm gì, khi nào, và giao nhau ở đâu*.

---

## Nguyên tắc thiết kế phân công

**Vấn đề cần tránh:** pipeline vốn tuần tự — trích xuất, rồi huấn luyện, rồi
streaming, rồi đánh giá. Nếu chia theo đúng thứ tự đó thì người thứ ba và thứ tư
ngồi chơi ba tuần đầu, rồi cả nhóm dồn việc vào hai tuần cuối.

**Cách giải:** tách thành bốn luồng mà ba trong số đó **không cần mô hình đã
huấn luyện** mới bắt đầu được.

- Máy trạng thái phát hiện ranh giới chỉ cần landmark, không cần mô hình. Căn
  ngưỡng chuyển động làm được từ tuần 1 bằng webcam.
- Từ điển gloss→tiếng Việt chỉ cần danh sách 100 nhãn, có ngay sau tuần 1.
- Khảo sát dữ liệu, tổng quan tài liệu, khung đánh giá đều độc lập với mô hình.

**Một thay đổi quan trọng so với kế hoạch cá nhân:** buổi quay video chuyển từ
tuần 5 lên **tuần 2**. Có 4 người thì đây là việc một buổi chiều, và nó nằm trên
đường găng của mô hình cuối. Không có lý do gì để đợi.

**Lợi thế ít ai để ý:** 4 người = 4 tài khoản Kaggle = khoảng 120 giờ GPU mỗi
tuần thay vì 30. Các thí nghiệm huấn luyện độc lập nhau nên chạy song song được
trên 4 tài khoản. Đây là lý do nhóm 4 người có thể làm ST-GCN và bảng ablation
đầy đủ, còn một người thì không.

---

## Bốn vai

| Vai | Phụ trách | Sản phẩm cuối |
|---|---|---|
| **D1 — Dữ liệu & hạ tầng** | Trích xuất landmark, manifest, quản lý Kaggle Dataset, tổ chức buổi quay | `manifest.csv`, `all_manifest.csv`, dataset landmark trên Kaggle |
| **D2 — Mô hình** | Kiến trúc, huấn luyện, siêu tham số | `best.pt`, bảng so sánh kiến trúc |
| **D3 — Hệ thống real-time** | Máy trạng thái, ONNX, demo webcam, gloss→tiếng Việt | Demo chạy được, `vsl.onnx` |
| **D4 — Đánh giá & báo cáo** | Khung đo, phân tích lỗi, hình vẽ, báo cáo, slide | `streaming_eval.json`, báo cáo, slide bảo vệ |

---

## Lịch theo tuần

### Tuần 1 — Khởi động song song

| Vai | Việc |
|---|---|
| **D1** | Cài môi trường, chạy `selftest.py`, upload `src/` thành Kaggle Dataset, chạy `kaggle_01_extract.py`, đọc bảng kiểm tra (gloss thiếu trong test, clip pose kém), xuất `vsl400-landmarks` và gửi `keep_labels.txt` cho cả nhóm |
| **D2** | Đọc `models.py` và `dataset.py`, chạy thử `train.py` trên tập nhỏ 10 nhãn để chắc pipeline thông, chuẩn bị bảng thí nghiệm |
| **D3** | Chạy `stream.py --ckpt` với mô hình ngẫu nhiên chỉ để xem thanh năng lượng, ghi lại dải giá trị khi đứng yên và khi cử động, đề xuất `motion_hi`/`motion_lo` ban đầu |
| **D4** | Tổng quan tài liệu: ST-GCN, 2s-AGCN, WLASL, AUTSL. Viết nháp mục 1–2 của báo cáo (bài toán, phạm vi). Khảo sát phân bố lớp từ manifest của D1 |

**Giao nộp cuối tuần:** D1 đưa dataset `vsl400-landmarks` và file `keep_labels.txt` (150 gloss). Toàn nhóm phụ thuộc vào đây — D3 cần danh sách gloss để viết câu test.

---

### Tuần 2 — Baseline và buổi quay

| Vai | Việc |
|---|---|
| **D1** | Tổ chức buổi quay: đặt lịch, chuẩn bị phòng đủ sáng, kiểm tra webcam. Quay 40 clip nghỉ + 15–20 câu × 3 người |
| **D2** | Chạy `kaggle_02_train.py`: huấn luyện `--split-mode provided` và `--split-mode random`, lấy bảng so sánh hai cách chia |
| **D3** | Viết lại `configs/gloss_vi.json` cho đúng 100 nhãn thật. Viết `configs/test_sentences.txt` chỉ dùng gloss có trong tập |
| **D4** | Dựng khung bảng kết quả trong báo cáo, viết mục phương pháp, chuẩn bị script sinh hình từ `history.csv` |

**Điểm giao nhau quan trọng:** D3 phải giao `test_sentences.txt` **trước** buổi
quay của D1. Quay xong mới phát hiện câu chứa ký hiệu ngoài tập từ vựng thì mất
cả buổi và phải hẹn lại mọi người.

**Cả bốn người cùng tham gia buổi quay** với vai trò người biểu diễn. Càng nhiều
người khác nhau, con số cross-signer càng có ý nghĩa.

---

### Tuần 3 — Mở rộng mô hình

| Vai | Việc |
|---|---|
| **D1** | Trích landmark cho clip tự quay, chạy `merge_manifests.py`, giao `all_manifest.csv` |
| **D2** | Cài đặt ST-GCN: định nghĩa đồ thị xương 71 keypoint, tích chập không gian–thời gian, tích hợp vào `build_model()` |
| **D3** | Hoàn thiện máy trạng thái, viết `export_onnx.py` vào quy trình, kiểm tra sai lệch PyTorch vs ONNX |
| **D4** | Chạy song song nhánh Transformer trên tài khoản Kaggle của mình, giao kết quả cho D2 tổng hợp |

---

### Tuần 4 — Ablation

Bốn cấu hình ablation chạy **song song trên bốn tài khoản Kaggle**, mỗi người một cấu hình:

| Vai | Cấu hình che |
|---|---|
| **D1** | bỏ nét mặt (40 chiều `g_face`) |
| **D2** | bỏ đặc trưng tay cục bộ (84 chiều) |
| **D3** | bỏ vận tốc (142 chiều cuối) |
| **D4** | chỉ một tay (zero tay trái) |

Quy ước đặt tên để kết quả không đụng nhau: `runs/abl_<tên-cấu-hình>_<tên-người>`.
Mỗi người nộp file `test_metrics.json`, D4 gộp thành một bảng.

Song song: D2 huấn luyện mô hình chính trên `all_manifest.csv` có lớp `__NOSIGN__`.

---

### Tuần 5 — Ghép hệ thống

| Vai | Việc |
|---|---|
| **D1** | Kiểm tra chất lượng dữ liệu quay: clip hỏng, tỉ lệ phát hiện tay thấp, quay bù nếu cần |
| **D2** | Chốt mô hình tốt nhất, xuất `best.pt`, viết mô tả kiến trúc cho báo cáo |
| **D3** | Tích hợp mô hình thật vào `stream.py`, căn ngưỡng bằng thanh năng lượng, đo FPS |
| **D4** | Chạy `evaluate.py`, lấy `top_confusions.csv`, **mở video 5 cặp nhầm nhiều nhất ra xem tận mắt** |

---

### Tuần 6 — Đo con số thật

| Vai | Việc |
|---|---|
| **D1** | Hỗ trợ D4 đối soát nhãn tham chiếu với video test |
| **D2** | Huấn luyện lại nếu phân tích lỗi chỉ ra vấn đề cụ thể |
| **D3** | Chạy `eval_streaming.py` ở nhiều mức `--conf`, dựng đường cong WER–báo động giả |
| **D4** | Viết mục kết quả, dựng toàn bộ hình và bảng |

---

### Tuần 7 — Hoàn thiện

| Vai | Việc |
|---|---|
| **D1** | Dọn repo, viết hướng dẫn tái lập, đóng gói dữ liệu nộp kèm |
| **D2** | Viết mục mô hình và ablation |
| **D3** | Quay video demo dự phòng, đề phòng webcam hỏng lúc bảo vệ |
| **D4** | Ghép báo cáo hoàn chỉnh, dựng slide |

---

### Tuần 8 — Tập bảo vệ

Cả nhóm. Mỗi người trình bày phần của mình, ba người còn lại đóng vai giảng viên
đặt câu hỏi khó. Chuẩn bị sẵn câu trả lời cho ba câu gần như chắc chắn bị hỏi:

1. *Vì sao không dùng ảnh thô mà dùng landmark?*
2. *MediaPipe là mô hình pretrained, vậy nhóm tự dựng cái gì?*
3. *Tập test của dataset có thật sự toàn người ký chưa thấy không?* — Không hoàn toàn: phần video lấy từ internet chỉ chia theo gloss. Trả lời thẳng điều này.

---

## Hợp đồng giao diện

Bốn luồng chỉ chạy song song được nếu các định dạng trao đổi được đóng băng từ
tuần 1. Ai muốn đổi phải báo cả nhóm.

**`manifest.csv`** — D1 sở hữu. Cột bắt buộc: `npz`, `label`, `stem`, `group_id`,
`signer_id`, **`split`**, `n_frames`, `fps`, `det_rate`, `lhand_rate`, `rhand_rate`.
Cột `split` mang phép chia theo người ký của tác giả dataset — mất cột này là mất
con số trung thực nhất của cả đồ án.

**`best.pt`** — D2 sở hữu. Phải chứa `model`, `label_to_idx`, `arch`, `hidden`,
`layers`, `feature_dim`, `seq_len`. D3 đọc các khoá này trong `Recognizer._init_torch`.

**`config.py`** — D1 sở hữu, **không ai được sửa một mình**. Mọi hằng số dùng chung
nằm ở đây. `FEATURE_DIM`, `SEQ_LEN`, và các chỉ số keypoint phải giống hệt nhau
giữa khâu huấn luyện và khâu streaming; lệch nhau là mô hình sập mà không báo lỗi.

**`labels.json`** — D2 sinh ra, D3 dùng để viết từ điển, D4 dùng để đọc bảng nhầm lẫn.

---

## Ai được sửa file nào

| File | Chủ |
|---|---|
| `config.py` | D1 (sửa phải có đồng thuận) |
| `landmarks.py`, `extract_dataset.py`, `merge_manifests.py`, `splits.py` | D1 |
| `models.py`, `train.py`, `dataset.py` | D2 |
| `stream.py`, `export_onnx.py`, `gloss2vi.py`, `record_clips.py` | D3 |
| `evaluate.py`, `eval_streaming.py` | D4 |
| `configs/*.json`, `configs/*.txt` | D3 |
| `selftest.py` | ai sửa `landmarks.py` thì cập nhật luôn |

Quy tắc Git: mỗi người một nhánh `d1/`, `d2/`, `d3/`, `d4/`. Merge vào `main` phải
có ít nhất một người khác xem qua. Chạy `python src/selftest.py` trước mỗi lần merge.

---

## Phòng rủi ro

**Điểm chết một người.** Nếu D1 ốm tuần 1 thì cả nhóm đứng. Khắc phục: D1 đưa
`manifest.csv` và dataset landmark lên Kaggle ở chế độ chia sẻ cho cả nhóm ngay
khi có, không giữ trên máy cá nhân.

**Cặp đôi dự phòng:** D1↔D4 (đều làm việc với dữ liệu và thống kê), D2↔D3 (đều
làm việc với mô hình và tensor). Mỗi người đọc code của người kia ít nhất một lần
trước tuần 4.

**Quota Kaggle cạn.** Nếu một tài khoản hết giờ GPU, chuyển thí nghiệm sang tài
khoản khác. Đây là lý do quy ước đặt tên run phải có tên người.

**Buổi quay hỏng.** Ánh sáng kém, webcam mờ, người quay bận. Đặt lịch dự phòng
ngay tuần 3, đừng đợi đến lúc hỏng mới tìm lịch.

---

## Đóng góp từng người trong báo cáo

Nhiều môn yêu cầu ghi rõ đóng góp. Gợi ý cách diễn đạt:

- **D1** — xây dựng pipeline dữ liệu, phát hiện và xử lý vấn đề bản sao
  augmentation trong dataset, tổ chức thu thập dữ liệu bổ sung
- **D2** — thiết kế và huấn luyện các kiến trúc mạng, thực nghiệm siêu tham số,
  phân tích so sánh BiLSTM / Transformer / ST-GCN
- **D3** — thiết kế máy trạng thái phát hiện ranh giới, tối ưu suy luận CPU,
  xây dựng mô-đun chuyển gloss sang tiếng Việt
- **D4** — thiết kế khung đánh giá streaming, phân tích lỗi, tổng hợp báo cáo
