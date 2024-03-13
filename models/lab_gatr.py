import torch
import gatr
from nn.class_token import class_token_forward_wrapper
from typing import Callable
from data import Data
from xformers.ops.fmha import BlockDiagonalMask

from nn.mlp.geometric_algebra import MLP
from nn.gnn import pool
from torch_scatter import scatter
import nn.gnn
from gatr.interface import embed_translation


class LaBGATr(torch.nn.Module):

    def __init__(
        self,
        geometric_algebra_interface: object,
        d_model: int,
        num_blocks: int,
        num_attn_heads: int,
        num_latent_channels=None,
        use_class_token: bool = False,
        dropout_probability=None
    ):
        super().__init__()

        num_latent_channels = num_latent_channels or d_model

        self.tokeniser = Tokeniser(
            geometric_algebra_interface,
            d_model,
            num_latent_channels=4 * num_latent_channels,
            dropout_probability=dropout_probability
        )

        self.gatr = gatr.GATr(
            in_mv_channels=d_model,
            out_mv_channels=d_model,
            hidden_mv_channels=num_latent_channels,
            in_s_channels=geometric_algebra_interface.num_input_scalars,
            out_s_channels=geometric_algebra_interface.num_input_scalars,
            hidden_s_channels=4 * num_latent_channels,
            attention=gatr.SelfAttentionConfig(num_heads=num_attn_heads),
            mlp=gatr.MLPConfig(),
            num_blocks=num_blocks,
            dropout_prob=dropout_probability
        )
        self.gatr.forward = self.gatr_dual_reference_forward

        if use_class_token:
            self.gatr.forward = class_token_forward_wrapper(self.gatr.forward)

        self.num_attn_heads = num_attn_heads

        print(f"LaB-GATr ({sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)} parameters)")

    # Made redundant by patch to GATr repository (will change soon)
    def gatr_dual_reference_forward(
        self,
        multivectors: torch.Tensor,
        scalars=None,
        attention_mask=None,
        reference_multivector=None
    ) -> torch.Tensor:

        additional_qk_features_mv, additional_qk_features_s = self.gatr._construct_reinserted_channels(multivectors, scalars)

        h_mv, h_s = self.gatr.linear_in(multivectors, scalars=scalars)

        for block in self.gatr.blocks:
            if self.gatr._checkpoint_blocks:

                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    reference_mv=reference_multivector,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

            else:

                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    reference_mv=reference_multivector,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        outputs_mv, outputs_s = self.gatr.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def forward(self, data: Data) -> torch.Tensor:
        multivectors, scalars, reference_multivector = self.tokeniser(data)

        multivectors, scalars = self.gatr(
            multivectors,
            scalars=scalars,
            attention_mask=self.get_attn_mask(data),
            reference_multivector=reference_multivector
        )

        return self.tokeniser.lift(multivectors, scalars)

    @staticmethod
    def get_attn_mask(data: Data):

        if data.batch is None:
            attn_mask = None

        else:
            batch = data.batch[data.scale0_sampling_index]
            attn_mask = BlockDiagonalMask.from_seqlens(torch.bincount(batch).tolist())

        return attn_mask


class Tokeniser(torch.nn.Module):
    def __init__(self, geometric_algebra_interface: object, d_model: int, num_latent_channels=None, dropout_probability=None):
        super().__init__()
        self.geometric_algebra_interface = geometric_algebra_interface()

        num_input_channels = self.geometric_algebra_interface.num_input_channels
        num_output_channels = self.geometric_algebra_interface.num_output_channels

        num_input_scalars = self.geometric_algebra_interface.num_input_scalars
        num_output_scalars = self.geometric_algebra_interface.num_output_scalars

        num_latent_channels = num_latent_channels or d_model

        self.point_cloud_pooling = PointCloudPooling(MLP(
            (num_input_channels + 1, num_latent_channels, d_model),
            num_input_scalars,
            num_output_scalars=num_input_scalars,
            plain_last=False,
            use_norm_in_first=False,
            dropout_probability=dropout_probability
        ), node_dim=0)

        self.mlp = MLP(
            (d_model + num_input_channels, *[num_latent_channels] * 2, num_output_channels),
            num_input_scalars=2 * num_input_scalars,
            num_output_scalars=num_output_scalars,
            use_norm_in_first=False,
            dropout_probability=dropout_probability
        )

        self.cache = None

    def forward(self, data: Data) -> torch.Tensor:
        multivectors, scalars = self.geometric_algebra_interface.embed(data)

        self.cache = {
            'multivectors': multivectors,
            'scalars': scalars,
            'data': data,
            'reference_multivector': self.construct_reference_multivector(multivectors, data.batch)
        }

        (multivectors, scalars), self.cache['pos'] = pool(
            self.point_cloud_pooling,
            multivectors,
            data.pos,
            data,
            scale_id=0,
            scalars=scalars,
            reference_multivector=self.cache['reference_multivector']
        )

        return multivectors, scalars, self.cache['reference_multivector'][data.scale0_sampling_index]

    @staticmethod
    def construct_reference_multivector(x: torch.Tensor, batch=None):

        if batch is None:
            reference_multivector = x.mean(dim=(0,1)).expand(x.size(0), 1, -1)

        else:
            reference_multivector = scatter(x, batch, dim=0, reduce='mean').mean(dim=1, keepdim=True)[batch]

        return reference_multivector

    def lift(self, multivectors: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:

        if multivectors.size(0) == self.cache['data'].scale0_sampling_index.numel():
            multivectors, scalars = interp(
                self.mlp,
                multivectors,
                self.cache['multivectors'],
                scalars,
                self.cache['scalars'],
                self.cache['pos'],
                self.cache['data'].pos,
                self.cache['data'],
                scale_id=0,
                reference_multivector=self.cache['reference_multivector']
            )

        else:
            multivectors, scalars = self.extract_class(
                self.mlp,
                multivectors,
                self.cache['multivectors'],
                scalars,
                self.cache['scalars'],
                self.cache['data']
            )

        return self.geometric_algebra_interface.dislodge(multivectors, scalars)

    def extract_class(
        self,
        mlp: MLP,
        multivectors: torch.Tensor,
        multivectors_skip: torch.Tensor,
        scalars: torch.Tensor,
        scalars_skip: torch.Tensor,
        data: Data
    ) -> torch.Tensor:

        if data.batch is None:
            multivectors_skip = multivectors_skip.mean(dim=0, keepdim=True)
            scalars_skip = scalars_skip.mean(dim=0, keepdim=True)

            reference_multivector = self.cache['reference_multivector'][0:1]

        else:
            multivectors_skip = scatter(multivectors_skip, data.batch, dim=0, reduce='mean')
            scalars_skip = scatter(scalars_skip, data.batch, dim=0, reduce='mean')

            reference_multivector = self.cache['reference_multivector'][data.ptr[:-1]]

        multivectors = torch.cat((multivectors, multivectors_skip), dim=-2)
        scalars = torch.cat((scalars, scalars_skip), dim=-1)

        return mlp(multivectors, scalars, reference_mv=reference_multivector)


class PointCloudPooling(nn.gnn.PointCloudPooling):

    def message(
        self,
        x_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        scalars_j: torch.Tensor,
        reference_multivector_j: torch.Tensor
    ) -> torch.Tensor:

        multivectors, scalars = self.mlp(
            torch.cat((x_j, embed_translation(pos_j - pos_i).unsqueeze(-2)), dim=-2),
            scalars=scalars_j,
            reference_mv=reference_multivector_j
        )

        return multivectors, scalars

    def aggregate(self, inputs: tuple, index: torch.Tensor, ptr=None, dim_size=None) -> torch.Tensor:
        multivectors, scalars = (self.aggr_module(tensor, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim) for tensor in inputs)

        return multivectors, scalars


def interp(
    mlp: torch.nn.Module,
    multivectors: torch.Tensor,
    multivectors_skip: torch.Tensor,
    scalars: torch.Tensor,
    scalars_skip: torch.Tensor,
    pos_source: torch.Tensor,
    pos_target: torch.Tensor,
    data: Data,
    scale_id: int,
    reference_multivector: torch.Tensor
):

    pos_diff = pos_source[data[f'scale{scale_id}_interp_source']] - pos_target[data[f'scale{scale_id}_interp_target']]
    squared_pos_dist = torch.clamp(torch.sum(pos_diff ** 2, dim=-1), min=1e-16).view(-1, 1, 1)

    denominator = scatter(1. / squared_pos_dist, data[f'scale{scale_id}_interp_target'].long(), dim=0, reduce='sum')

    multivectors = scatter(
        multivectors[data[f'scale{scale_id}_interp_source']] / squared_pos_dist,
        data[f'scale{scale_id}_interp_target'].long(),
        dim=0,
        reduce='sum'
    ) / denominator

    scalars = scatter(
        scalars[data[f'scale{scale_id}_interp_source']] / squared_pos_dist.view(-1, 1),
        data[f'scale{scale_id}_interp_target'].long(),
        dim=0,
        reduce='sum'
    ) / denominator.view(-1, 1)

    multivectors = torch.cat((multivectors, multivectors_skip), dim=-2)
    scalars = torch.cat((scalars, scalars_skip), dim=-1)

    return mlp(multivectors, scalars, reference_mv=reference_multivector)
