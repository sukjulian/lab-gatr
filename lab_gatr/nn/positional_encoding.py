import torch
from torch.nn.functional import pad


class PositionalEncoding:
    """Inspired by the continuous sine-cosine embedding of AB-UPT (Alkin et al., 2025).
    """
    def __init__(self, num_channels: int, base: float = 1e4):
        self.num_channels = num_channels
        self.base = base

    def __call__(self, coord: torch.Tensor) -> torch.Tensor:
        num_pos, num_coord = coord.shape
        assert self.num_channels >= 2 * num_coord, "Positional encoding does not fit into channels."

        coord = self.scale(coord)

        num_frequencies = self.num_channels // (2 * num_coord)
        num_dim_padding = self.num_channels % (2 * num_coord)

        exponent = torch.arange(0, 2 * num_frequencies, step=2, device=coord.device) / (2 * num_frequencies)
        frequencies = 1. / self.base ** exponent

        coord = coord[:, :, None] * frequencies[None, None, :]
        coord = torch.cat((coord.sin(), coord.cos()), dim=-1)
        coord = coord.view(num_pos, -1)

        return pad(coord, (0, num_dim_padding))

    def scale(self, coord: torch.Tensor) -> torch.Tensor:
        coord = coord - coord.min()
        coord = coord / coord.max() * 0.1 * self.base

        return coord
