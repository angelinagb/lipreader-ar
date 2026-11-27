"""
Arquitectura del notebook 05_model_architecture (STCNN -> proj -> BiGRU -> clasificador CTC)
sin variables globales, más un Predictor para inferencia con un checkpoint (best_cer.pth, etc.).

Entrada del modelo: (B, T, 3, 96, 96) float32 en [0, 1], RGB, 25 fps.
No hay otra normalización: la hace MaskedBatchNorm3d dentro de la red.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MaskedBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, lengths):
        B, C, T, H, W = x.shape
        valid = torch.arange(T, device=x.device)[None, :] < lengths.to(x.device)[:, None]
        mask = valid[:, None, :, None, None].to(x.dtype)

        if self.training:
            count = mask.sum() * H * W
            mean = (x * mask).sum(dim=(0, 2, 3, 4)) / count
            var = (((x - mean[None, :, None, None, None]) ** 2) * mask).sum(dim=(0, 2, 3, 4)) / count
            with torch.no_grad():
                n = count.item()
                unbiased_var = var.detach() * (n / (n - 1)) if n > 1 else var.detach()
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(unbiased_var, self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean[None, :, None, None, None]) / torch.sqrt(var[None, :, None, None, None] + self.eps)
        x = x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        return x * mask


class STCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # BLOCK 1
        self.bn1 = MaskedBatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn2 = MaskedBatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout1 = nn.Dropout3d(p=0.1)
        # BLOCK 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn3 = MaskedBatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout2 = nn.Dropout3d(p=0.1)
        # BLOCK 3
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn4 = MaskedBatchNorm3d(96)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def apply_temporal_mask(self, x, lengths):
        valid = torch.arange(x.size(2), device=x.device)[None, :] < lengths.to(x.device)[:, None]
        return x * valid[:, None, :, None, None]

    def forward(self, x, lengths):
        x = self.bn1(x, lengths)
        x = self.conv1(x)
        x = self.bn2(x, lengths)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.apply_temporal_mask(x, lengths)

        x = self.conv2(x)
        x = self.bn3(x, lengths)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.apply_temporal_mask(x, lengths)

        x = self.conv3(x)
        x = self.bn4(x, lengths)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.apply_temporal_mask(x, lengths)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2, rnn_type="gru", dropout=0.2):
        super().__init__()
        if rnn_type.lower() == "gru":
            rnn_class = nn.GRU
        elif rnn_type.lower() == "lstm":
            rnn_class = nn.LSTM
        else:
            raise ValueError("rnn_type debe ser 'gru' o 'lstm'")
        self.rnn = rnn_class(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=False, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=False, total_length=x.size(0))
        return output


class CharacterClassifier(nn.Module):
    def __init__(self, input_size=512, vocab_size=71):
        super().__init__()
        self.linear = nn.Linear(input_size, vocab_size)

    def forward(self, x):
        return self.linear(x)


class LipReadingModel(nn.Module):
    def __init__(self, in_channels=3, vocab_size=71, hidden_size=256,
                 rnn_type="gru", proj_size=512, upsample=2):
        super().__init__()
        self.upsample = upsample
        self.visual_encoder = STCNN(in_channels=in_channels)
        self.proj = nn.Sequential(
            nn.Linear(96 * 6 * 6, proj_size),
            nn.LayerNorm(proj_size),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.temporal_encoder = TemporalEncoder(
            input_size=proj_size, hidden_size=hidden_size, num_layers=2,
            rnn_type=rnn_type, dropout=0.2,
        )
        self.classifier = CharacterClassifier(input_size=hidden_size * 2, vocab_size=vocab_size)

    def forward(self, x, input_lengths):
        x = x.permute(0, 2, 1, 3, 4)                      # (B, C, T, H, W)
        x = self.visual_encoder(x, input_lengths)         # (B, 96, T, 6, 6)
        x = x.permute(2, 0, 1, 3, 4)                      # (T, B, 96, 6, 6)
        x = x.reshape(x.shape[0], x.shape[1], -1)         # (T, B, 3456)
        x = self.proj(x)                                  # (T, B, proj_size)
        if self.upsample > 1:
            x = x.repeat_interleave(self.upsample, dim=0)
        out_lengths = input_lengths * self.upsample
        x = self.temporal_encoder(x, out_lengths)
        log_probs = torch.log_softmax(self.classifier(x), dim=-1)
        return log_probs, out_lengths                     # (T*up, B, V), (B,)


# ----------------------------------------------------------------------------
class Predictor:
    """Carga un checkpoint del notebook y devuelve log-probs / texto para un clip RGB."""

    def __init__(self, ckpt_path, device=None, rnn_type="gru", hidden_size=256):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            ck = torch.load(ckpt_path, map_location=self.device)
        except Exception:
            # PyTorch >= 2.6 usa weights_only=True por defecto; es tu propio checkpoint
            ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.blank = int(ck.get("blank_idx", 0))
        self.idx_to_char = {int(i): c for c, i in ck["char_to_idx"].items()}
        vocab_size = int(ck["vocab_size"])
        upsample = int(ck.get("upsample", 2))

        self.model = LipReadingModel(
            in_channels=3, vocab_size=vocab_size, hidden_size=hidden_size,
            rnn_type=rnn_type, upsample=upsample,
        ).to(self.device)

        # strict=False por si el checkpoint viene de LipReadingModelVisemas (trae viseme_head)
        res = self.model.load_state_dict(ck["model_state_dict"], strict=False)
        extra = [k for k in res.unexpected_keys if not k.startswith("viseme_head")]
        if res.missing_keys or extra:
            raise RuntimeError(f"El checkpoint no coincide con el modelo: {res}")
        self.model.eval()
        self.info = {k: ck.get(k) for k in ("epoch", "val_cer", "val_wer")}

    @torch.inference_mode()
    def warmup(self, frames=30):
        x = torch.zeros(1, frames, 3, 96, 96, device=self.device)
        self.model(x, torch.tensor([frames]))

    @torch.inference_mode()
    def log_probs(self, clip_rgb):
        """clip_rgb: uint8 (T, 96, 96, 3) RGB -> log-probs (T*upsample, V) float32 en numpy."""
        x = torch.from_numpy(np.ascontiguousarray(clip_rgb)).to(self.device).float().div_(255.0)
        x = x.permute(0, 3, 1, 2).unsqueeze(0)            # (1, T, 3, H, W)
        lengths = torch.tensor([x.shape[1]])
        lp, out_len = self.model(x, lengths)
        return lp[: out_len[0].item(), 0].float().cpu().numpy()

    def greedy(self, lp):
        """Greedy CTC: argmax por paso, colapsa repetidos, saca blanks (igual que el notebook)."""
        out, prev = [], None
        for t in lp.argmax(axis=-1).tolist():
            if t != prev and t != self.blank:
                out.append(self.idx_to_char[t])
            prev = t
        return " ".join("".join(out).split())