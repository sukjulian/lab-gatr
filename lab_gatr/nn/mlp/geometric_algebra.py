import gatr
import torch
from gatr.layers.mlp.geometric_bilinears import GeometricBilinear
from gatr.layers.linear import EquiLinear
from gatr.layers.dropout import GradeDropout


REFERENCE_MULTIVECTOR = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.])


class Identity(torch.nn.Module):

    def forward(self, *args) -> tuple:
        return args


class EquiLayerNorm(gatr.layers.EquiLayerNorm):

    def forward(self, multivectors: torch.Tensor, scalars) -> tuple:

        multivectors = gatr.primitives.equi_layer_norm(multivectors)
        scalars = torch.nn.functional.layer_norm(scalars, normalized_shape=scalars.shape[-1:]) if scalars is not None else None

        return multivectors, scalars


class ScalarGatedNonlinearity(gatr.layers.ScalarGatedNonlinearity):

    def forward(self, multivectors: torch.Tensor, scalars) -> tuple:

        multivectors = self.gated_nonlinearity(multivectors, gates=multivectors[..., [0]])
        scalars = self.scalar_nonlinearity(scalars) if scalars is not None else None

        return multivectors, scalars


class MLP(torch.nn.Module):

    def __init__(
        self,
        num_channels: tuple,
        num_input_scalars=None,
        num_output_scalars=None,
        plain_last: bool = True,
        use_norm_in_first: bool = True,
        dropout_probability=None
    ):
        super().__init__()

        self.linear_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        self.linear_layers.append(GeometricBilinear(
            *num_channels[:2],
            in_s_channels=num_input_scalars,
            out_s_channels=4 * num_channels[1]
        ))
        self.norm_layers.append(EquiLayerNorm() if use_norm_in_first else Identity())
        self.activations.append(ScalarGatedNonlinearity('gelu'))

        for num_channels_in, num_channels_out in zip(num_channels[1:-2], num_channels[2:-1]):
            self.linear_layers.append(EquiLinear(
                in_mv_channels=num_channels_in,
                out_mv_channels=num_channels_out,
                in_s_channels=4 * num_channels_in,
                out_s_channels=4 * num_channels_out
            ))
            self.norm_layers.append(EquiLayerNorm())
            self.activations.append(ScalarGatedNonlinearity('gelu'))

        self.linear_layers.append(EquiLinear(
            *num_channels[-2:],
            in_s_channels=4 * num_channels[-2],
            out_s_channels=num_output_scalars
        ))

        if plain_last:
            self.norm_layers.append(Identity())
            self.activations.append(Identity())
        else:
            self.norm_layers.append(EquiLayerNorm())
            self.activations.append(ScalarGatedNonlinearity('gelu'))

        self.dropout = GradeDropout(dropout_probability) if dropout_probability else Identity()

    def forward(self, multivectors: torch.Tensor, scalars, reference_mv: torch.Tensor = REFERENCE_MULTIVECTOR) -> tuple:

        multivectors, scalars = self.activations[0](*self.norm_layers[0](*self.linear_layers[0](
            multivectors,
            scalars=scalars,
            reference_mv=reference_mv.to(device=multivectors.device, dtype=multivectors.dtype)
        )))

        for linear_layer, norm_layer, activation in zip(self.linear_layers[1:], self.norm_layers[1:], self.activations[1:]):
            multivectors, scalars = activation(*norm_layer(*linear_layer(*self.dropout(multivectors, scalars))))

        return multivectors, scalars
