import torch_geometric as pyg
from .data import Data
import torch
from torch_cluster import fps, knn


class PointCloudPoolingScales():
    """Nested hierarchy of sub-sampled point clouds. Each coarse-scale point is mapped to a cluster of fine-scale points. Proportional
    interpolation from the coarse to the fine scales. For correct batching, "torch_geometric.data.Data.__inc__()" has to be overridden.

    Args:
        rel_sampling_ratios (tuple): relative ratios for successive farthest point sampling
        interp_simplex (str): reference simplex for proportional interpolation ('triangle' or 'tetrahedron')
    """

    def __init__(self, rel_sampling_ratios: tuple, interp_simplex: str):
        self.rel_sampling_ratios = rel_sampling_ratios
        self.interp_simplex = interp_simplex

        self.dim_interp_simplex = {'triangle': 2, 'tetrahedron': 3}[interp_simplex]

    def __call__(self, data: pyg.data.Data) -> Data:

        pos = data.pos
        batch = data.surface_id.long() if hasattr(data, 'surface_id') else torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        for i, sampling_ratio in enumerate(self.rel_sampling_ratios):

            sampling_idcs = fps(pos, batch, ratio=sampling_ratio)  # takes some time but is worth it
            # sampling_idcs = torch.arange(0, pos.size(0), 1. / sampling_ratio, dtype=torch.int)

            pool_source, pool_target = knn(pos[sampling_idcs], pos, 1, batch[sampling_idcs], batch)
            interp_target, interp_source = knn(pos[sampling_idcs], pos, self.dim_interp_simplex + 1, batch[sampling_idcs], batch)

            data[f'scale{i}_pool_target'], data[f'scale{i}_pool_source'] = pool_target.int(), pool_source.int()
            data[f'scale{i}_interp_target'], data[f'scale{i}_interp_source'] = interp_target.int(), interp_source.int()
            data[f'scale{i}_sampling_index'] = sampling_idcs.int()

            pos = pos[sampling_idcs]
            batch = batch[sampling_idcs]

        return Data(**data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rel_sampling_ratios={self.rel_sampling_ratios}, interp_simplex={self.interp_simplex})"
