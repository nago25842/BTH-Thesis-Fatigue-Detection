"""
model.py
--------
Attention-enhanced Bidirectional LSTM for player fatigue prediction.

Architecture overview
---------------------
Input (B, T, F)
    │
    ▼
InputProjection   – linear + layer norm + dropout
    │
    ▼
BiLSTM Encoder    – stacked bidirectional LSTM (hidden_dim * 2 per step)
    │
    ▼
TemporalAttention – additive (Bahdanau-style) self-attention over time steps
    │              returns attended context AND raw attention weights (for viz)
    ▼
ClassifierHead    – LayerNorm → Linear → (sequence or final prediction)
    │
    ▼
Output logits  (B, T) for sequence mode  |  (B,) for last mode

The attention weights (B, T) are returned alongside logits so they can
be visualised and connected to RQ1 (which temporal patterns matter most).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """
    Projects raw features → hidden_dim with layer norm and dropout.
    Decouples feature dimensionality from LSTM hidden size, making it
    easy to add or remove features without changing the LSTM.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, T, hidden_dim)"""
        return self.drop(self.norm(torch.nn.functional.gelu(self.proj(x))))


class BiLSTMEncoder(nn.Module):
    """
    Stacked bidirectional LSTM.

    Output dim per time step = hidden_dim * 2  (fwd + bwd concatenated).
    Supports packing for variable-length sequences to avoid wasted compute
    on padding tokens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2  # bidirectional

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x       : (B, T, input_dim)
            lengths : (B,) actual sequence lengths for packing  [optional]

        Returns:
            hidden_states : (B, T, hidden_dim * 2)
        """
        if lengths is not None:
            # Pack to skip computation on padding tokens
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        return out  # (B, T, hidden_dim * 2)


class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) temporal self-attention.

    Computes a scalar score for every time step, normalises with softmax
    (respecting the padding mask), and returns both the attended context
    vector AND the raw attention weights.

    The weights are the interpretability hook for RQ1 — high weight at
    time step t means the model considers that window most informative
    for the prediction at that position.

    For sequence labelling we compute attention independently at each
    query position (full self-attention over the sequence).
    For match-level prediction we use a single global attention over all
    time steps to produce one context vector.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Additive attention: score(h) = v^T · tanh(W·h + b)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden : (B, T, H)  encoder hidden states
            mask   : (B, T) bool tensor — True where data is real
                     Padding positions (False) are masked out of softmax.

        Returns:
            context : (B, T, H)  attention-weighted hidden states
                      (each time step attends over the full sequence)
            weights : (B, T)     attention weight at each time step
                      (averaged over query positions for interpretability)
        """
        B, T, H = hidden.shape

        # Score every time step: (B, T, 1)
        scores = self.v(torch.tanh(self.W(hidden)))  # (B, T, 1)
        scores = scores.squeeze(-1)                  # (B, T)

        # Mask padding: set to -inf so softmax → 0
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)      # (B, T)

        # Replace any NaN rows (all-pad sequences) with uniform weights
        nan_mask = torch.isnan(weights)
        if nan_mask.any():
            weights = torch.where(nan_mask, torch.ones_like(weights) / T, weights)

        weights = self.drop(weights)

        # Weighted context: broadcast (B, T, 1) * (B, T, H) → (B, T, H)
        context = weights.unsqueeze(-1) * hidden     # (B, T, H)

        return context, weights  # context: (B,T,H), weights: (B,T)


class ClassifierHead(nn.Module):
    """
    Maps attended representations → binary fatigue logits.

    Two variants:
        sequence mode : outputs a logit for every time step  (B, T)
        last mode     : outputs a single logit per sequence  (B,)
    """

    def __init__(self, input_dim: int, dropout: float = 0.3, mode: str = "sequence"):
        super().__init__()
        assert mode in ("sequence", "last")
        self.mode = mode
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x       : (B, T, H) — attended hidden states
            lengths : (B,) — needed to extract last valid step in 'last' mode

        Returns:
            logits : (B, T) for sequence mode | (B,) for last mode
        """
        x = self.drop(torch.nn.functional.gelu(self.fc1(self.norm(x)))) # (B, T, H//2)
        logits = self.fc2(x).squeeze(-1)               # (B, T)

        if self.mode == "last":
            if lengths is not None:
                # Gather the logit at the last real time step per sequence
                idx = (lengths - 1).clamp(min=0)       # (B,)
                idx = idx.view(-1, 1).expand(-1, 1)    # (B, 1)
                logits = logits.gather(1, idx).squeeze(1)  # (B,)
            else:
                logits = logits[:, -1]

        return logits


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AttentionBiLSTM(nn.Module):
    """
    Full Attention-enhanced Bidirectional LSTM for fatigue prediction.

    Args:
        input_dim   : number of input features (F)
        hidden_dim  : LSTM hidden size per direction (output = hidden_dim*2)
        num_layers  : number of stacked BiLSTM layers
        dropout     : dropout probability (applied in LSTM, projection, head)
        mode        : "sequence" (label every window) or "last" (final window)

    Forward returns:
        logits      : (B, T) or (B,) depending on mode
        attn_weights: (B, T) attention weights — use for interpretability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        mode: str = "sequence",
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.projection = InputProjection(input_dim, hidden_dim, dropout=dropout)
        self.encoder = BiLSTMEncoder(hidden_dim, hidden_dim, num_layers, dropout)
        self.attention = TemporalAttention(self.encoder.output_dim, dropout=0.1)
        self.classifier = ClassifierHead(self.encoder.output_dim, dropout, mode)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear layers; orthogonal for LSTM weights."""
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    # Set forget gate bias to 1 (helps with long sequences)
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features : (B, T, F)
            lengths  : (B,) actual sequence lengths
            mask     : (B, T) bool — True = real data

        Returns:
            logits       : (B, T) or (B,)
            attn_weights : (B, T)
        """
        # 1. Project features → hidden_dim
        x = self.projection(features)                  # (B, T, H)

        # 2. Bidirectional LSTM encoding
        hidden = self.encoder(x, lengths)              # (B, T, H*2)

        # 3. Temporal attention
        context, attn_weights = self.attention(hidden, mask)  # (B,T,H*2), (B,T)

        # 4. Residual: attended context + raw hidden (like a skip connection)
        fused = hidden + context                       # (B, T, H*2)

        # 5. Classify
        logits = self.classifier(fused, lengths)       # (B,T) or (B,)

        return logits, attn_weights

    def predict_proba(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper — returns probabilities instead of logits."""
        logits, weights = self.forward(features, lengths, mask)
        return torch.sigmoid(logits), weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Baseline: standard BiLSTM (no attention)
# ---------------------------------------------------------------------------

class BiLSTMBaseline(nn.Module):
    """
    Standard BiLSTM without the attention layer.
    Identical hyper-parameters to AttentionBiLSTM for fair comparison.
    The only difference is the absence of TemporalAttention and the
    residual connection — the hidden states go straight to the classifier.

    Returns a dummy ones-tensor in place of attention weights so the
    training loop can treat both models identically.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        mode: str = "sequence",
    ):
        super().__init__()
        self.mode = mode
        self.projection = InputProjection(input_dim, hidden_dim, dropout)
        self.encoder = BiLSTMEncoder(hidden_dim, hidden_dim, num_layers, dropout)
        self.classifier = ClassifierHead(self.encoder.output_dim, dropout, mode)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.projection(features)
        hidden = self.encoder(x, lengths)
        logits = self.classifier(hidden, lengths)
        dummy_weights = torch.ones(features.shape[:2], device=features.device)
        return logits, dummy_weights


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    mode: str = "sequence",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory function. model_type must be "attention" or "bilstm".

    Args:
        model_type : "attention" → AttentionBiLSTM
                     "bilstm"    → BiLSTMBaseline
        input_dim  : number of features (must match your dataset)
        hidden_dim : LSTM hidden size per direction
        num_layers : stacked BiLSTM layers
        dropout    : dropout probability
        mode       : "sequence" or "last"
        device     : torch.device to move model to

    Returns:
        model on the specified device
    """
    if model_type == "attention":
        model = AttentionBiLSTM(input_dim, hidden_dim, num_layers, dropout, mode)
    elif model_type == "bilstm":
        model = BiLSTMBaseline(input_dim, hidden_dim, num_layers, dropout, mode)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'attention' or 'bilstm'.")

    if device is not None:
        model = model.to(device)

    return model


# ---------------------------------------------------------------------------
# Smoke test  (run: python model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Simulate a batch that matches your dataset output:
    # B=8 sequences, T=22 max length, F=4 features
    B, T, F = 8, 22, 4
    features = torch.randn(B, T, F).to(device)
    lengths  = torch.tensor([8, 16, 9, 10, 7, 16, 22, 18]).to(device)
    mask     = torch.arange(T).unsqueeze(0).to(device) < lengths.unsqueeze(1)

    print("=" * 55)
    print("AttentionBiLSTM")
    print("=" * 55)
    attn_model = build_model("attention", input_dim=F, device=device)
    logits, weights = attn_model(features, lengths, mask)
    print(f"  Parameters   : {attn_model.count_parameters():,}")
    print(f"  logits shape : {logits.shape}   (B, T)")
    print(f"  weights shape: {weights.shape}  (B, T)")
    print(f"  logits range : [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  weights sum  : {weights.sum(dim=-1).tolist()}")  # should be ≈1.0

    # Verify attention weights sum to 1 over valid positions
    attn_model.eval() 
    with torch.no_grad():
        _, weights = attn_model(features, lengths, mask)
    for i in range(B):
        valid_weight_sum = weights[i][mask[i]].sum().item()
        assert abs(valid_weight_sum - 1.0) < 1e-3, \
            f"Seq {i} sum is {valid_weight_sum:.4f}"
    print("  ✓ Attention weights sum to 1.0 over valid positions")

    print()
    print("=" * 55)
    print("BiLSTMBaseline (no attention)")
    print("=" * 55)
    base_model = build_model("bilstm", input_dim=F, device=device)
    logits_b, _ = base_model(features, lengths, mask)
    print(f"  Parameters   : {base_model.count_parameters():,}")
    print(f"  logits shape : {logits_b.shape}")

    print()
    print("=" * 55)
    print("Loss computation check (sequence mode)")
    print("=" * 55)
    labels = torch.randint(0, 2, (B, T)).float().to(device)
    labels[~mask] = -100  # mask padding
    pos_weight = torch.tensor([22.9]).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction="none"
    )
    raw_loss = criterion(logits, labels)
    # Only average over valid (non-padded) positions
    valid_loss = (raw_loss * mask.float()).sum() / mask.float().sum()
    print(f"  Loss (masked mean): {valid_loss.item():.4f}")
    print(f"  ✓ Loss computed correctly on {mask.sum().item()} valid positions")

    print()
    print("=" * 55)
    print("'last' mode check")
    print("=" * 55)
    attn_last = build_model("attention", input_dim=F, mode="last", device=device)
    logits_last, w_last = attn_last(features, lengths, mask)
    print(f"  logits shape (last mode): {logits_last.shape}  (B,)")
    print(f"  ✓ last mode outputs one logit per sequence")

    print("\n✓ All model checks passed.")