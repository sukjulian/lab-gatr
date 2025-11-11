import torch
from gatr.baselines.transformer import BaselineLayerNorm
from gatr.layers import ApplyRotaryPositionalEncoding
from einops import rearrange
from gatr.utils.tensors import to_nd, expand_pairwise
from gatr.primitives.attention import scaled_dot_product_attention


class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout_prob=None, positional_encoding_base=None):
        super().__init__()

        self.norm = BaselineLayerNorm()

        # Number of hidden channels has to be divisible by four for xFormers block-diagonal attention bias
        hidden_channels = channels // num_heads - channels // num_heads % 4

        self.attention = CrossAttention(
            in_kv_channels=channels,
            in_q_channels=channels,
            out_channels=channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            positional_encoding_base=positional_encoding_base
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, 2 * channels),
            torch.nn.Dropout(dropout_prob) if dropout_prob else torch.nn.Identity(),
            torch.nn.GELU(),
            torch.nn.Linear(2 * channels, channels),
            torch.nn.Dropout(dropout_prob) if dropout_prob else torch.nn.Identity()
        )

    def forward(self, inputs_kv: torch.Tensor, inputs_q: torch.Tensor, attention_mask=None) -> torch.Tensor:

        h_kv = self.norm(inputs_kv)
        h_q = self.norm(inputs_q)

        h = self.attention(h_kv, h_q, attention_mask)

        outputs = inputs_q + h

        h = self.norm(outputs)

        h = self.mlp(h)

        outputs = outputs + h

        return outputs


class CrossAttention(torch.nn.Module):

    def __init__(
        self,
        in_kv_channels: int,
        in_q_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_heads: int,
        dropout_prob=None,
        positional_encoding_base=None
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        self.qkv_linear = MultiQueryQKVLinear(in_kv_channels, in_q_channels, hidden_channels, num_heads)
        self.out_linear = torch.nn.Linear(hidden_channels * num_heads, out_channels)

        self.dropout = torch.nn.Dropout(dropout_prob) if dropout_prob else torch.nn.Identity()

        if positional_encoding_base:
            self.positional_encoding = ApplyRotaryPositionalEncoding(
                hidden_channels,
                item_dim=-2,
                base=positional_encoding_base
            )
        else:
            self.positional_encoding = torch.nn.Identity()

    def forward(self, inputs_kv: torch.Tensor, inputs_q: torch.Tensor, attention_mask=None) -> torch.Tensor:
        q, k, v = self.qkv_linear(inputs_kv, inputs_q)

        q = self.positional_encoding(q)
        k = self.positional_encoding(k)

        h = self._attend(q, k, v, attention_mask)

        h = rearrange(h, "... num_heads num_items hidden_channels -> ... num_items (num_heads hidden_channels)")
        outputs = self.out_linear(h)

        outputs = self.dropout(outputs)

        return outputs

    @staticmethod
    def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask=None) -> torch.Tensor:

        bh_shape = q.shape[:-2]
        q = to_nd(q, 4)
        k = to_nd(k, 4)
        v = to_nd(v, 4)

        outputs = scaled_dot_product_attention(*expand_pairwise(q, k, v, exclude_dims=(-2,)), attn_mask=attention_mask)

        outputs = outputs.view(*bh_shape, *outputs.shape[-2:])

        return outputs


class MultiQueryQKVLinear(torch.nn.Module):
    def __init__(self, in_kv_channels: int, in_q_channels: int, hidden_channels: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        self.q_linear = torch.nn.Linear(in_q_channels, hidden_channels * num_heads)
        self.k_linear = torch.nn.Linear(in_kv_channels, hidden_channels)
        self.v_linear = torch.nn.Linear(in_kv_channels, hidden_channels)

    def forward(self, inputs_kv: torch.Tensor, inputs_q: torch.Tensor) -> torch.Tensor:

        q = rearrange(
            self.q_linear(inputs_q),
            "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
            num_heads=self.num_heads
        )
        k = self.k_linear(inputs_kv)[..., None, :, :]
        v = self.v_linear(inputs_kv)[..., None, :, :]

        return q, k, v
