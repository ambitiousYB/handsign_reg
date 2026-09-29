"""
Chuyển chuỗi gloss -> câu tiếng Việt.

Đây là phần mà phần lớn đồ án nhận diện ngôn ngữ ký hiệu bỏ qua, và cũng là phần
dễ ghi điểm nhất, vì nó cho thấy bạn hiểu rằng ngôn ngữ ký hiệu là MỘT NGÔN NGỮ
RIÊNG chứ không phải tiếng Việt được mã hoá bằng tay.

Khác biệt ngữ pháp chính giữa VSL và tiếng Việt nói:
  1. Trạng ngữ thời gian đứng đầu câu:  HÔM-QUA TÔI ĐI CHỢ
  2. Phủ định đứng cuối:                TÔI ĐI KHÔNG   -> "Tôi không đi"
  3. Lược bỏ hư từ:                     không có "là", "thì", "của", "các"
  4. Trật tự chủ đề - bình luận:        CHỢ TÔI ĐI      -> "Chợ thì tôi đi"
  5. Thì thể hiện bằng trạng ngữ, không bằng "đã/đang/sẽ"

Chiến lược: luật trước, mô hình sau. Luật đủ tốt cho 80% câu ngắn và chạy tức thì
trên CPU; nếu còn thời gian thì huấn luyện thêm một seq2seq nhỏ và SO SÁNH hai
cách trong báo cáo — đó là bảng kết quả có giá trị.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Từ điển tối thiểu. Mở rộng dần theo bộ từ vựng bạn chọn.
DEFAULT_LEXICON: dict[str, dict] = {
    # đại từ
    "TÔI": {"vi": "tôi", "pos": "PRON"},
    "BẠN": {"vi": "bạn", "pos": "PRON"},
    "ANH": {"vi": "anh", "pos": "PRON"},
    "CHỊ": {"vi": "chị", "pos": "PRON"},
    "CHÚNG-TÔI": {"vi": "chúng tôi", "pos": "PRON"},
    # thời gian
    "HÔM-NAY": {"vi": "hôm nay", "pos": "TIME", "tense": "pres"},
    "HÔM-QUA": {"vi": "hôm qua", "pos": "TIME", "tense": "past"},
    "NGÀY-MAI": {"vi": "ngày mai", "pos": "TIME", "tense": "fut"},
    "TUẦN-TRƯỚC": {"vi": "tuần trước", "pos": "TIME", "tense": "past"},
    "SÁNG": {"vi": "sáng", "pos": "TIME"},
    "TỐI": {"vi": "tối", "pos": "TIME"},
    # động từ
    "ĐI": {"vi": "đi", "pos": "VERB"},
    "ĂN": {"vi": "ăn", "pos": "VERB"},
    "UỐNG": {"vi": "uống", "pos": "VERB"},
    "HỌC": {"vi": "học", "pos": "VERB"},
    "NGHE": {"vi": "nghe", "pos": "VERB"},
    "THƯƠNG": {"vi": "thương", "pos": "VERB"},
    "CỨU": {"vi": "cứu", "pos": "VERB"},
    "AN-ỦI": {"vi": "an ủi", "pos": "VERB"},
    "XIN-LỖI": {"vi": "xin lỗi", "pos": "VERB"},
    "NGHỈ-NGƠI": {"vi": "nghỉ ngơi", "pos": "VERB"},
    # danh từ
    "CHỢ": {"vi": "chợ", "pos": "NOUN"},
    "BỆNH-VIỆN": {"vi": "bệnh viện", "pos": "NOUN"},
    "HỌC-SINH": {"vi": "học sinh", "pos": "NOUN"},
    "THỨC-ĂN": {"vi": "thức ăn", "pos": "NOUN"},
    "XE-ĐẠP": {"vi": "xe đạp", "pos": "NOUN"},
    "Ô-TÔ": {"vi": "ô tô", "pos": "NOUN"},
    "KHẨU-TRANG": {"vi": "khẩu trang", "pos": "NOUN"},
    "BẠN-THÂN": {"vi": "bạn thân", "pos": "NOUN"},
    # phủ định / nghi vấn
    "KHÔNG": {"vi": "không", "pos": "NEG"},
    "CHƯA": {"vi": "chưa", "pos": "NEG"},
    "GÌ": {"vi": "gì", "pos": "WH"},
    "AI": {"vi": "ai", "pos": "WH"},
    "ĐÂU": {"vi": "đâu", "pos": "WH"},
    "BAO-NHIÊU": {"vi": "bao nhiêu", "pos": "WH"},
    # tính từ / trạng thái
    "SỐT": {"vi": "sốt", "pos": "ADJ"},
    "MỆT": {"vi": "mệt", "pos": "ADJ"},
    "VUI": {"vi": "vui", "pos": "ADJ"},
}


class GlossTranslator:
    def __init__(self, lexicon_path: str | Path | None = None,
                 insert_tense: bool = True):
        self.lex = dict(DEFAULT_LEXICON)
        self.insert_tense = insert_tense
        if lexicon_path and Path(lexicon_path).exists():
            extra = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
            for k, v in extra.items():
                self.lex[self._norm(k)] = v if isinstance(v, dict) else {"vi": v}

    @staticmethod
    def _norm(g: str) -> str:
        return re.sub(r"\s+", "-", g.strip()).upper()

    def _entry(self, gloss: str) -> dict:
        g = self._norm(gloss)
        if g in self.lex:
            return {"gloss": g, **self.lex[g]}
        # nhãn không có trong từ điển: dùng chính nhãn, hạ chữ thường
        return {"gloss": g, "vi": g.replace("-", " ").lower(), "pos": "X"}

    def translate(self, glosses: list[str]) -> str:
        """Chuỗi gloss -> một câu tiếng Việt."""
        items = [self._entry(g) for g in glosses
                 if g and not g.startswith("__")]
        if not items:
            return ""

        # --- Luật 1: đưa trạng ngữ thời gian lên đầu ---
        times = [x for x in items if x.get("pos") == "TIME"]
        rest = [x for x in items if x.get("pos") != "TIME"]

        # --- Luật 2: phủ định cuối câu -> đặt trước động từ ---
        negs = [x for x in rest if x.get("pos") == "NEG"]
        if negs:
            rest = [x for x in rest if x.get("pos") != "NEG"]
            vi_idx = next((i for i, x in enumerate(rest)
                           if x.get("pos") in ("VERB", "ADJ")), None)
            if vi_idx is not None:
                rest.insert(vi_idx, negs[0])
            else:
                rest.append(negs[0])

        # --- Luật 3: chèn "là" giữa hai danh ngữ liền nhau ---
        out: list[str] = []
        seq = times + rest
        has_neg = any(x.get("pos") == "NEG" for x in seq)

        tense = next((t.get("tense") for t in times if t.get("tense")), None)
        inserted_tense = False

        for i, x in enumerate(seq):
            prev = seq[i - 1] if i else None

            if (prev and prev.get("pos") in ("NOUN", "PRON")
                    and x.get("pos") == "NOUN"
                    and not any(y.get("pos") == "VERB" for y in seq)):
                out.append("là")

            # --- Luật 4: chèn dấu hiệu thì trước động từ chính ---
            if (self.insert_tense and not inserted_tense and not has_neg
                    and x.get("pos") == "VERB" and tense):
                if tense == "past":
                    out.append("đã")
                elif tense == "fut":
                    out.append("sẽ")
                inserted_tense = True

            out.append(x["vi"])

        s = " ".join(out).strip()
        s = re.sub(r"\s+", " ", s)
        if not s:
            return ""

        is_q = any(x.get("pos") == "WH" for x in seq)
        s = s[0].upper() + s[1:]
        return s + ("?" if is_q else ".")

    def translate_batch(self, sentences: list[list[str]]) -> list[str]:
        return [self.translate(s) for s in sentences]


def _demo() -> None:
    t = GlossTranslator()
    cases = [
        ["HÔM-QUA", "TÔI", "ĐI", "CHỢ"],
        ["TÔI", "ĐI", "KHÔNG"],
        ["BẠN", "ĂN", "GÌ"],
        ["NGÀY-MAI", "TÔI", "HỌC"],
        ["TÔI", "HỌC-SINH"],
        ["TÔI", "MỆT", "KHÔNG"],
    ]
    for c in cases:
        print(f"  {' '.join(c):32s} ->  {t.translate(c)}")


if __name__ == "__main__":
    _demo()
