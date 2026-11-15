"""BiLSTM and attention classifier over frozen WavLM frame features."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WavLMEmotionHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(256)
        self.pool_score = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            features,
            lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=features.shape[1],
        )
        positions = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
        padding_mask = positions >= lengths.to(sequence.device)[:, None]
        attended, _ = self.self_attention(
            sequence,
            sequence,
            sequence,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        sequence = self.norm(sequence + attended)
        scores = self.pool_score(sequence).squeeze(-1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)
