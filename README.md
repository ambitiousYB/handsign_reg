# VSL Streaming — Nhận diện ngôn ngữ ký hiệu tiếng Việt thời gian thực

Pipeline hoàn chỉnh: video → landmark → mô hình chuỗi → gloss → câu tiếng Việt,
chạy real-time từ webcam. Thiết kế cho ràng buộc **laptop không GPU, huấn luyện
trên Kaggle**.

---

## 0. Quyết định quan trọng nhất trước khi bắt đầu

**Làm isolated recognition ở chế độ streaming, KHÔNG làm continuous translation.**

Dịch ngôn ngữ ký hiệu liên tục (mô hình tự học ranh giới từ dữ liệu có nhãn cấp
câu) là bài toán cấp nghiên cứu, cần hàng chục nghìn câu có nhãn. Tiếng Việt không
có nguồn đó. Ôm hướng này thì hết kỳ bạn sẽ không có kết quả nào để nộp.

Thay vào đó: mô hình phân loại từng ký hiệu, còn **ranh giới do một máy trạng thái
dựa trên năng lượng chuyển động phát hiện**. Trải nghiệm người dùng vẫn đúng như
bạn muốn — bật camera, ký hiệu liên tiếp, chữ hiện ra dần — nhưng khả thi trong
một học kỳ, và phần máy trạng thái lại chính là đóng góp kỹ thuật để viết báo cáo.

---

## 1. Kiến trúc pipeline

```
         ┌──────────── OFFLINE, trên Kaggle ─────────────┐
video ──▶│ MediaPipe Holistic ──▶ landmark thô (.npz)    │
         │                              │                │
         │                              ▼                │
         │                   chuẩn hoá theo thân người    │
         │                   + đặc trưng tay cục bộ       │
         │                   + vận tốc  = 370 chiều       │
         │                              │                │
         │                              ▼                │
         │              CNN1D + BiLSTM + attention pool   │
         │                              │                │
         └──────────────────────────────┼────────────────┘
                                        ▼
                                   best.pt / vsl.onnx
                                        │
         ┌──────────── REAL-TIME, trên laptop ───────────┐
webcam ─▶│ MediaPipe ─▶ đệm vòng ─▶ năng lượng chuyển động│
         │                              │                │
         │                    máy trạng thái ranh giới    │
         │              IDLE ─▶ ACTIVE ─▶ (kết thúc)      │
         │                              │                │
         │                              ▼                │
         │         cửa sổ 48 frame ─▶ mô hình ─▶ softmax  │
         │                              │                │
         │            lọc ngưỡng + margin + lớp __NOSIGN__│
         │                              │                │
         │                              ▼                │
         │              bộ đệm gloss ─▶ luật ngữ pháp     │
         │                              ▼                │
         │                      câu tiếng Việt            │
         └────────────────────────────────────────────────┘
```

**Nguyên tắc bất di bất dịch:** khâu offline và khâu real-time dùng CHUNG file
`landmarks.py`. Nếu hai bên chuẩn hoá khác nhau, mô hình vẫn chạy, vẫn xuất nhãn,
nhưng độ chính xác sụp mà không có thông báo lỗi nào. Đây là lỗi khó phát hiện
nhất trong loại dự án này.

---

## 2. Vì sao dùng landmark chứ không dùng ảnh thô

| | Ảnh thô (I3D, TimeSformer…) | Landmark (pipeline này) |
|---|---|---|
| Real-time trên CPU laptop | Không | Có, ~15-25 FPS |
| Dữ liệu cần để hội tụ | Rất nhiều | Vài chục mẫu/lớp là đủ |
| Học tắt theo nền/quần áo | Rất dễ mắc | Gần như miễn nhiễm |
| Huấn luyện trên Kaggle | Hàng giờ mỗi lần | 15-25 phút |
| Trần hiệu năng | Cao hơn | Thấp hơn một chút |

Với ràng buộc của bạn, landmark là lựa chọn đúng. Và việc nêu rõ bảng đánh đổi này
trong báo cáo chính là phần "lý do thiết kế" mà giảng viên tìm kiếm.

Đặc trưng 370 chiều mỗi frame gồm:
- **142** — 71 điểm (9 thân trên + 21×2 bàn tay + 20 mặt) chuẩn hoá theo tâm vai,
  tỉ lệ vai, và xoay về phương ngang → bất biến với vị trí, khoảng cách, độ nghiêng
- **84** — bàn tay chuẩn hoá theo chính nó (gốc = cổ tay, tỉ lệ = RMS toàn bàn tay)
  → tách *hình dạng bàn tay* khỏi *vị trí bàn tay*
- **142** — vận tốc, tính SAU khi resample → bất biến với fps nguồn
- **2** — cờ có/không phát hiện được mỗi bàn tay

---

## 3. Cài đặt

### Trên laptop (chỉ để demo và quay dữ liệu, không cần torch)

```bash
git clone <repo-của-bạn> && cd vsl
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/selftest.py        # phải ra "TẤT CẢ ĐỀU ĐẠT"
```

`selftest.py` kiểm tra bằng dữ liệu giả trong 2 giây: đúng shape, bất biến với
tỉ lệ/vị trí/fps, chịu được mất pose và mất bàn tay, máy trạng thái hoạt động,
công thức WER đúng, luật ngữ pháp đúng. **Luôn chạy nó trước khi tốn 30 phút
trích xuất landmark thật.**

### Trên Kaggle

Upload thư mục `src/` thành một Kaggle Dataset (đặt tên vd `vsl-code`) để notebook
nào cũng dùng được. Mỗi lần sửa code thì cập nhật phiên bản dataset đó.

---

## 4. Quy trình từng bước

### Bước 1 — Trích xuất landmark (Kaggle, CPU, ~30 phút, chạy một lần)

Notebook: `notebooks/kaggle_01_extract.py`. Cài đặt: **Accelerator = None**,
Internet = On. Bật GPU ở bước này chỉ tốn quota vô ích vì MediaPipe chạy CPU.

```bash
python extract_dataset.py \
    --root /kaggle/input/visignlanguage-video/train \
    --out  /kaggle/working/landmarks \
    --workers 4 --model-complexity 1 \
    --group-regex '^(\d+)'
```

**`--group-regex '^(\d+)'` là cờ quan trọng nhất trong cả lệnh.** Bộ HF namwu chứa
các file `127824.mp4`, `127824_1.mp4`, `127824_2.mp4`, `127824_3.mp4` có cùng kích
thước và cùng thời lượng — chúng là biến thể augmentation của **cùng một clip gốc**.
Nếu chia ngẫu nhiên, bản gốc rơi vào train còn bản `_1` rơi vào test, và accuracy
bạn báo cáo sẽ là con số ảo trên 95%. Regex này gom chúng về một nhóm.

Script **chạy tiếp được**: nếu phiên Kaggle bị ngắt, chạy lại đúng lệnh, những clip
đã xong sẽ bị bỏ qua.

Cuối bước, xuất `landmarks/` thành Kaggle Dataset mới (tab Output → New Dataset).
Notebook huấn luyện sẽ dùng nó làm input, không phải chạy lại khâu này bao giờ nữa.

### Bước 2 — Huấn luyện (Kaggle, GPU, ~20 phút mỗi lần)

Notebook: `notebooks/kaggle_02_train.py`. Accelerator = GPU P100.

```bash
python train.py --manifest .../manifest.csv --split-mode group \
                --model bilstm --epochs 80 --out runs/bilstm_group
```

Chạy **cả ba** cách chia tập. Đây là bảng kết quả chính của đồ án:

```bash
python evaluate.py --compare runs/bilstm_random runs/bilstm_group runs/bilstm_signer
```

| Cách chia | Ý nghĩa | Con số mong đợi |
|---|---|---|
| `random` | Có rò rỉ. Chỉ để chứng minh nó thổi phồng ra sao | Rất cao, vô nghĩa |
| `group` | Không cho bản sao nằm hai phía | Thực tế hơn nhiều |
| `signer` | Không cho người biểu diễn nằm hai phía | Con số TRUNG THỰC |

Khoảng cách giữa dòng `random` và dòng `signer` chính là phần phân tích có giá trị
nhất trong báo cáo. Rất nhiều bài báo về nhận diện ngôn ngữ ký hiệu báo cáo con số
kiểu `random` mà không nói rõ — việc bạn chỉ ra điều đó bằng thực nghiệm của chính
mình là một điểm cộng thật sự.

Lưu ý: với bộ HF namwu, `--split-mode signer` **sẽ báo lỗi** vì manifest không có
ID người biểu diễn. Đó là kết quả đúng, và chính là lý do bạn cần dataset thứ hai
(VSL400 hoặc Multi-VSL).

Script lưu checkpoint sau **mỗi** epoch. Phiên Kaggle bị cắt giữa chừng thì chạy
lại với `--resume`.

### Bước 3 — Phân tích lỗi

```bash
python evaluate.py --run runs/bilstm_group
```

Sinh ra `per_class.csv`, `top_confusions.csv`, `confidence_curve.csv`.

Việc cần làm bằng tay: mở video của 5-10 cặp nhầm lẫn nhiều nhất ra **xem tận mắt**.
Thường chúng chỉ khác nhau ở handshape hoặc ở nét mặt. Một câu kết luận cụ thể kiểu
"mô hình nhầm X với Y vì hai ký hiệu chỉ khác ở vị trí ngón cái, mà MediaPipe hay
mất dấu ngón cái khi bàn tay nghiêng" có giá trị hơn mười trang bảng số.

`confidence_curve.csv` dùng để **chọn ngưỡng** `conf_threshold` cho streaming một
cách có cơ sở. Trong ứng dụng dịch, một từ sai gây hại hơn một từ bị bỏ qua, nên
hãy ưu tiên precision.

### Bước 4 — Quay dữ liệu của riêng bạn (bắt buộc)

Không dataset công khai nào giúp được hai việc sau, vì tất cả đều là clip đã cắt sẵn.

**4a. Clip trạng thái nghỉ.** Ngồi yên, gãi đầu, uống nước, chỉnh tóc, nói chuyện.

```bash
python record_clips.py --mode idle --out data/raw/__NOSIGN__ --signer B01 -n 40
```

Không có lớp này, hệ thống sẽ phun gloss liên tục mỗi khi bạn cử động bất kỳ.
Pipeline có sinh negative tổng hợp (`--nosign-ratio`), nhưng clip thật tốt hơn hẳn.

**4b. Tập test câu liên tục.** Khoảng 15-20 câu, 2-3 người, mỗi câu 3-5 ký hiệu.

```bash
python record_clips.py --mode sentence --out data/continuous \
    --signer B01 --script configs/test_sentences.txt
```

Một buổi là xong. Đây là tập nhỏ nhưng cho ra con số quan trọng nhất của đồ án.

### Bước 5 — Demo real-time

```bash
python export_onnx.py --ckpt runs/bilstm_group/best.pt --out models/vsl.onnx
python stream.py --ckpt models/vsl.onnx --backend onnx --model-complexity 0
```

Phím: `q` thoát, `c` xoá bộ đệm câu.

Trên màn hình có thanh năng lượng chuyển động kèm hai vạch ngưỡng. **Hãy dùng nó
để căn chỉnh**: ký hiệu bình thường vài lần và quan sát thanh này. Nếu lúc bạn
đứng yên mà thanh đã vượt vạch đỏ thì tăng `--motion-hi`; nếu ký hiệu chậm mà máy
không kích hoạt thì giảm xuống.

### Bước 6 — Đánh giá streaming (con số quan trọng nhất)

```bash
python eval_streaming.py --ckpt models/vsl.onnx \
    --videos data/continuous --refs configs/test_sentences.txt \
    --idle-videos data/raw/__NOSIGN__
```

Ba chỉ số để đưa vào báo cáo:

- **WER** — tỉ lệ lỗi cấp từ khi ký hiệu liên tục
- **Độ trễ** — từ lúc kết thúc ký hiệu tới lúc chữ hiện ra
- **Báo động giả/phút** — số gloss xuất ra khi người dùng không ký hiệu

Đọc kết quả:
- Nhiều lỗi **chèn** → ngưỡng chuyển động quá nhạy, hoặc thiếu lớp nghỉ
- Nhiều lỗi **xoá** → `conf_threshold` hoặc `motion_hi` quá cao
- Nhiều lỗi **thay thế** → vấn đề ở mô hình, quay lại bảng nhầm lẫn

Khoảng cách giữa accuracy offline và WER streaming sẽ rất lớn. **Đừng giấu nó** —
việc bạn đo được và giải thích được khoảng cách đó chính là điểm mạnh của bài.

---

## 5. Kế hoạch 8 tuần

> **Xem `KE-HOACH.md`** — đó là trình tự chính thức, dùng dataset VSL400 bản
> Cropped trên Kaggle với `--split-mode provided`. Các mục về bộ PTIT và
> `--split-mode group` trong README này đã lỗi thời.

### Bản tóm tắt

| Tuần | Việc | Mốc kiểm tra |
|---|---|---|
| 1 | Cài đặt, `selftest.py` đạt, tải bộ namwu, trích xuất landmark | Có `manifest.csv` với >3.000 clip |
| 2 | Huấn luyện baseline, chạy cả ba cách chia tập | BiLSTM vượt rõ mức ngẫu nhiên trên split `group` |
| 3 | Xin/tải VSL400 hoặc Multi-VSL, trích xuất, huấn luyện lại | **Mốc quyết định**: có `signer_id` chưa? |
| 4 | So sánh kiến trúc, bảng ablation đặc trưng | Có bảng ablation hoàn chỉnh |
| 5 | Quay clip nghỉ + tập câu liên tục | 40 clip nghỉ, 15-20 câu, ≥2 người |
| 6 | Xây và căn chỉnh máy trạng thái streaming | Demo webcam chạy được đầu-cuối |
| 7 | Đo WER, độ trễ, báo động giả; căn ngưỡng | Có `streaming_eval.json` |
| 8 | Phân tích lỗi, viết báo cáo, chuẩn bị demo | Bản nháp hoàn chỉnh |

**Mốc dừng tuần 3:** nếu đến hết tuần 3 vẫn không có dataset nào kèm ID người biểu
diễn, hãy chuyển kế hoạch: tự quay một tập nhỏ 20-30 ký hiệu với 5-8 người (nhờ bạn
cùng lớp, mỗi người 30 phút). Từ vựng nhỏ hơn nhưng bạn có được phép chia theo
người — và một kết quả trung thực trên 25 lớp có giá trị hơn một con số ảo trên 100 lớp.

---

## 6. Ba thí nghiệm để bài nổi bật

Đồ án chỉ báo cáo accuracy sẽ chìm nghỉm. Ba thí nghiệm sau đều chạy được trong
quota Kaggle và đều cho ra bảng/hình đáng đưa vào báo cáo.

**1. Bảng so sánh cách chia tập.** Đã mô tả ở bước 2. Rẻ nhất, thuyết phục nhất.

**2. Ablation nhóm đặc trưng.** Che (zero-out) từng khối trong `featurize` rồi
huấn luyện lại: bỏ nét mặt, bỏ đặc trưng tay cục bộ, bỏ vận tốc, chỉ giữ một tay.
Trả lời được câu hỏi "nét mặt có thực sự cần không" bằng số liệu, thay vì trích dẫn.

**3. Đường cong hiệu quả nhãn.** Huấn luyện với 25%, 50%, 75%, 100% số mẫu mỗi lớp.
Cho biết thu thêm dữ liệu có đáng không, và là biểu đồ rất được đánh giá cao.

Nếu còn thời gian, hướng thứ tư đáng giá nhất: **pretrain trên một dataset ngôn ngữ
ký hiệu khác** (AUTSL của Thổ Nhĩ Kỳ, WLASL của Mỹ) rồi fine-tune sang VSL. Vì
pipeline dùng landmark đã chuẩn hoá, đặc trưng gần như tương thích xuyên ngôn ngữ,
và câu chuyện "transfer learning xuyên ngôn ngữ ký hiệu" rất mạnh cho báo cáo.

---

## 7. Những chỗ dễ sai

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| Accuracy >95% ở lần chạy đầu | Rò rỉ dữ liệu. Kiểm tra `--group-regex` |
| Offline tốt, real-time tệ | Train và stream chuẩn hoá khác nhau, hoặc thiếu lớp `__NOSIGN__` |
| Máy phun gloss liên tục | Thiếu clip nghỉ; tăng `--motion-hi` và `--conf` |
| Không bao giờ kích hoạt | `motion_hi` quá cao; xem thanh năng lượng trên màn hình |
| Webcam dưới 10 FPS | Dùng `--model-complexity 0` và giảm độ phân giải |
| `mp.solutions.holistic` không tồn tại | Bản mediapipe quá mới, pin về `0.10.14` |
| Kaggle hết dung lượng | Landmark lưu float16 đã nén; xoá video gốc khỏi `/kaggle/working` |
| Phiên Kaggle bị cắt | Chạy lại với `--resume`, checkpoint lưu mỗi epoch |

---

## 8. Cấu trúc thư mục

```
vsl/
├── README.md
├── requirements.txt
├── configs/
│   ├── gloss_vi.json          từ điển gloss -> tiếng Việt (mở rộng dần)
│   └── test_sentences.txt     kịch bản quay tập test liên tục
├── notebooks/
│   ├── kaggle_01_extract.py   notebook trích xuất (CPU)
│   └── kaggle_02_train.py     notebook huấn luyện (GPU)
└── src/
    ├── config.py              hằng số dùng chung — SỬA Ở ĐÂY, không sửa rải rác
    ├── landmarks.py           trích xuất + chuẩn hoá (dùng chung offline/real-time)
    ├── extract_dataset.py     trích xuất hàng loạt, đa tiến trình, chạy tiếp được
    ├── merge_manifests.py     gộp manifest công khai với clip tự quay
    ├── splits.py              chia tập chống rò rỉ (không cần torch)
    ├── dataset.py             Dataset PyTorch + augmentation + negative tổng hợp
    ├── models.py              BiLSTM và Transformer
    ├── train.py               huấn luyện, checkpoint mỗi epoch, resume
    ├── evaluate.py            phân tích lỗi, bảng so sánh cách chia
    ├── export_onnx.py         xuất ONNX cho CPU
    ├── stream.py              demo webcam real-time
    ├── record_clips.py        quay clip nghỉ và tập câu liên tục
    ├── eval_streaming.py      WER, độ trễ, báo động giả
    └── selftest.py            kiểm thử nhanh bằng dữ liệu giả
```

---

## 9. Giới hạn cần nêu thẳng trong báo cáo

Nêu rõ giới hạn được đánh giá cao hơn nhiều so với giả vờ chúng không tồn tại.

1. Hệ thống nhận diện **từng ký hiệu rời**, không phải dịch liên tục thật sự. Ranh
   giới do heuristic chuyển động quyết định, nên ký hiệu nối liền không nghỉ sẽ bị
   gộp hoặc bỏ sót.
2. Mô-đun gloss→tiếng Việt dựa trên luật, chỉ phủ được các mẫu câu đơn giản.
3. Bộ từ vựng cố định (closed-set). Ký hiệu ngoài tập huấn luyện sẽ bị gán nhầm
   vào lớp gần nhất — lớp `__NOSIGN__` và ngưỡng tin cậy giảm nhẹ chứ không giải
   quyết triệt để.
4. Landmark 2D làm mất chiều sâu; các ký hiệu chỉ khác nhau ở hướng lòng bàn tay
   rất dễ bị nhầm.
5. Dữ liệu huấn luyện quay trong điều kiện studio, còn webcam thật có ánh sáng và
   nền kém hơn nhiều. Chênh lệch giữa accuracy offline và WER streaming phản ánh
   đúng khoảng cách này.
