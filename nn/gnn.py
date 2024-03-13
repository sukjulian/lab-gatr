import torch_geometric as pyg
import torch
from data import Data
from torch_geometric.nn.conv import MessagePassing


def pool(layer: pyg.nn.conv.message_passing, x: torch.Tensor, pos: torch.Tensor, data: Data, scale_id: int, **kwargs) -> tuple:
    sampling_idcs = data[f'scale{scale_id}_sampling_index']

    edge_index = torch.cat((data[f'scale{scale_id}_pool_source'][None, :], data[f'scale{scale_id}_pool_target'][None, :]), dim=0)
    kwargs = {key: (value, value[sampling_idcs]) for key, value in kwargs.items()}

    return layer((x, x[sampling_idcs]), (pos, pos[sampling_idcs]), edge_index.long(), **kwargs), pos[sampling_idcs]


class PointCloudPooling(MessagePassing):
    def __init__(self, mlp: torch.nn.Module, **kwargs):
        kwargs.setdefault('aggr', 'mean')

        super().__init__(**kwargs)

        self.mlp = mlp

    def message(self, x_j: torch.Tensor, pos_i: torch.Tensor, pos_j: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((x_j, pos_j - pos_i), dim=-1))

    def forward(self, x: tuple, pos: tuple, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.propagate(edge_index, x=x, pos=pos, **kwargs)
