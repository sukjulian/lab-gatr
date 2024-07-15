import torch
import gatr.baselines
from lab_gatr.nn.class_token import class_token_forward_wrapper
from lab_gatr.data import Data
from lab_gatr.nn.attn_mask import get_attn_mask

from lab_gatr.nn.mlp.vanilla import MLP
from lab_gatr.nn.gnn import PointCloudPooling, pool
from torch_scatter import scatter


class LaBVaTr(torch.nn.Module):

    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        d_model: int,
        num_blocks: int,
        num_attn_heads: int,
        num_latent_channels=None,
        use_class_token: bool = False
    ):
        super().__init__()

        num_latent_channels = num_latent_channels or d_model

        self.tokeniser = Tokeniser(num_input_channels, num_output_channels, d_model, num_latent_channels=4 * num_latent_channels)

        self.vatr = gatr.baselines.BaselineTransformer(
            in_channels=d_model,
            out_channels=d_model,
            hidden_channels=num_latent_channels,
            num_blocks=num_blocks,
            num_heads=num_attn_heads,
            multi_query=True
        )

        if use_class_token:
            self.vatr.forward = class_token_forward_wrapper(self.vatr.forward)
            self.tokeniser.mlp.norm_layers = torch.nn.ModuleList([torch.nn.Identity()] * len(self.tokeniser.mlp.norm_layers))

        self.num_parameters = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        print(f"LaB-VaTr ({self.num_parameters} parameters)")

    def forward(self, data: Data) -> torch.Tensor:
        x = self.tokeniser(data)

        x = self.vatr(x, attention_mask=get_attn_mask(data.batch[data.scale0_sampling_index] if data.batch is not None else data.batch))

        return self.tokeniser.lift(x)


class Tokeniser(torch.nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, d_model: int, num_latent_channels=None):
        super().__init__()

        num_latent_channels = num_latent_channels or d_model

        self.point_cloud_pooling = PointCloudPooling(MLP(
            (num_input_channels + 3, num_latent_channels, d_model),
            plain_last=False,
            use_norm_in_first=False
        ), node_dim=0)

        self.mlp = MLP((d_model + num_input_channels, *[num_latent_channels] * 2, num_output_channels), use_norm_in_first=False)

        self.cache = {}

    def forward(self, data: Data) -> torch.Tensor:
        self.cache['data'] = data

        x, self.cache['pos'] = pool(self.point_cloud_pooling, data.x, data.pos, data, scale_id=0)

        return x

    def lift(self, x: torch.Tensor) -> torch.Tensor:

        if x.size(0) == self.cache['data'].scale0_sampling_index.numel():
            x = interp(self.mlp, x, self.cache['data'].x, self.cache['pos'], self.cache['data'].pos, self.cache['data'], scale_id=0)

        else:
            x = self.extract_class(self.mlp, x, self.cache['data'].x, self.cache['data'])

        return x

    def extract_class(self, mlp: MLP, x: torch.Tensor, x_skip: torch.Tensor, data: Data) -> torch.Tensor:
        x_skip = data.x.mean(dim=0, keepdim=True) if data.batch is None else scatter(data.x, data.batch, dim=0, reduce='mean')

        return mlp(torch.cat((x, x_skip), dim=-1))


def interp(
    mlp: torch.nn.Module,
    x: torch.Tensor,
    x_skip: torch.Tensor,
    pos_source: torch.Tensor,
    pos_target: torch.Tensor,
    data: Data,
    scale_id: int
) -> torch.Tensor:

    pos_diff = pos_source[data[f'scale{scale_id}_interp_source']] - pos_target[data[f'scale{scale_id}_interp_target']]
    squared_pos_dist = torch.clamp(torch.sum(pos_diff ** 2, dim=-1, keepdim=True), min=1e-16)

    x = scatter(
        x[data[f'scale{scale_id}_interp_source']] / squared_pos_dist,
        data[f'scale{scale_id}_interp_target'].long(),
        dim=0,
        reduce='sum'
    ) / scatter(
        1. / squared_pos_dist,
        data[f'scale{scale_id}_interp_target'].long(),
        dim=0,
        reduce='sum'
    )

    return mlp(torch.cat((x, x_skip), dim=-1))
