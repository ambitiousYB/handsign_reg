"""
Kiến trúc mô hình.

Cả hai mô hình đều nhận (B, T, D) với D = 370 và trả về logits (B, C).
Chúng đủ nhỏ để chạy real-time trên CPU laptop — đó là ràng buộc thiết kế, không
phải sự thoả hiệp. Toàn bộ chi phí thị giác nặng nằm ở MediaPipe; phần phân loại
chuỗi chỉ tốn vài mili-giây.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalStem(nn.Module):
    """Vài lớp Conv1d để gom thông tin cục bộ theo thời gian trước khi vào RNN/Attention."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_out, d_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
        )

    def forward(self, x):                  # (B, T, D)
        x = x.transpose(1, 2)              # (B, D, T)
        x = self.net(x)
        return x.transpose(1, 2)           # (B, T, d_out)


class AttentionPool(nn.Module):
    """Gộp chuỗi có trọng số — tốt hơn mean pooling vì ký hiệu có phần 'lõi'
    mang nghĩa nằm ở giữa, còn hai đầu là chuyển động vào/ra vị trí."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x):                  # (B, T, d)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)   # (B, T)
        return torch.einsum("bt,btd->bd", w, x)


class BiLSTMClassifier(nn.Module):
    """Đường cơ sở chủ lực. Chính là kiến trúc CNN + BiLSTM, nhưng đầu vào là
    chuỗi landmark thay vì spectrogram."""

    def __init__(self, d_in: int, n_classes: int, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.stem = TemporalStem(d_in, hidden, dropout=dropout * 0.3)
        self.rnn = nn.LSTM(
            hidden, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x, _ = self.rnn(x)
        x = self.pool(x)
        return self.head(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Mô hình đối chứng. Với dữ liệu vài nghìn clip thì Transformer thường KHÔNG
    thắng BiLSTM — và việc bạn chỉ ra điều đó bằng thực nghiệm, thay vì mặc định
    cho rằng mô hình mới hơn thì tốt hơn, là một điểm cộng trong báo cáo."""

    def __init__(self, d_in: int, n_classes: int, hidden: int = 256,
                 layers: int = 4, heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.stem = TemporalStem(d_in, hidden, dropout=dropout * 0.3)
        self.pos = PositionalEncoding(hidden)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.pool = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x)


def build_model(name: str, d_in: int, n_classes: int, hidden: int = 256,
                layers: int = 2, dropout: float = 0.3) -> nn.Module:
    name = name.lower()
    if name == "bilstm":
        return BiLSTMClassifier(d_in, n_classes, hidden, layers, dropout)
    if name == "transformer":
        return TransformerClassifier(d_in, n_classes, hidden,
                                     layers=max(layers, 3), dropout=dropout)
    raise ValueError(f"Không biết mô hình '{name}'")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
