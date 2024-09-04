import torch
from torch.nn import Linear, BatchNorm1d, Identity, ReLU, Dropout


class MLP(torch.nn.Module):

    def __init__(
        self,
        num_channels: tuple,
        plain_last: bool = True,
        use_norm_in_first: bool = True,
        dropout_probability=None,
        use_running_stats_in_norm: bool = True
    ):
        super().__init__()

        self.linear_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        self.linear_layers.append(Linear(*num_channels[:2]))
        if use_norm_in_first:
            self.norm_layers.append(BatchNorm1d(num_channels[1], track_running_stats=use_running_stats_in_norm))
        else:
            self.norm_layers.append(Identity())
        self.activations.append(ReLU())

        for num_channels_in, num_channels_out in zip(num_channels[1:-2], num_channels[2:-1]):
            self.linear_layers.append(Linear(num_channels_in, num_channels_out))
            self.norm_layers.append(BatchNorm1d(num_channels_out, track_running_stats=use_running_stats_in_norm))
            self.activations.append(ReLU())

        self.linear_layers.append(Linear(*num_channels[-2:]))

        if plain_last:
            self.norm_layers.append(Identity())
            self.activations.append(Identity())
        else:
            self.norm_layers.append(BatchNorm1d(num_channels[-1], track_running_stats=use_running_stats_in_norm))
            self.activations.append(ReLU())

        self.dropout = Dropout(dropout_probability) if dropout_probability else Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for linear_layer, norm_layer, activation in zip(self.linear_layers, self.norm_layers, self.activations):
            x = activation(norm_layer(linear_layer(self.dropout(x))))

        return x
