import torch
import gatr
from gatr.layers.layer_norm import EquiLayerNorm
from dataclasses import replace
from gatr.layers.attention.cross_attention import CrossAttention
from gatr.layers.mlp.mlp import GeoMLP


class CrossAttentionBlock(torch.nn.Module):

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        attention: gatr.layers.attention.config.SelfAttentionConfig,
        mlp: gatr.layers.mlp.config.MLPConfig,
        dropout_prob=None
    ):
        super().__init__()

        self.norm = EquiLayerNorm()

        self.attention = CrossAttention(
            replace(
                attention,
                in_mv_channels=mv_channels,
                out_mv_channels=mv_channels,
                in_s_channels=s_channels,
                out_s_channels=s_channels,
                output_init="small",
                dropout_prob=dropout_prob
            ),
            in_q_mv_channels=mv_channels,
            in_q_s_channels=s_channels
        )

        self.mlp = GeoMLP(
            replace(
                mlp,
                mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
                s_channels=(s_channels, 2 * s_channels, s_channels),
                dropout_prob=dropout_prob
            )
        )

    def forward(
        self,
        multivectors_kv: torch.Tensor,
        multivectors_q: torch.Tensor,
        scalars_kv: torch.Tensor,
        scalars_q: torch.Tensor,
        reference_mv=None,
        attention_mask=None
    ) -> tuple:

        h_mv_kv, h_s_kv = self.norm(multivectors_kv, scalars=scalars_kv)
        h_mv_q, h_s_q = self.norm(multivectors_q, scalars=scalars_q)

        h_mv, h_s = self.attention(h_mv_kv, h_mv_q, scalars_kv=h_s_kv, scalars_q=h_s_q, attention_mask=attention_mask)

        outputs_mv = multivectors_q + h_mv
        outputs_s = scalars_q + h_s

        h_mv, h_s = self.norm(outputs_mv, scalars=outputs_s)

        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)

        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s
