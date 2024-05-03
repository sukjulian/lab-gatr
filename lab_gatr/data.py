import torch_geometric as pyg


class Data(pyg.data.Data):

    def __cat_dim__(self, key: str, value, *args, **kwargs) -> int:
        if 'index' in key or key == 'face' or key == 'tets' or 'coo' in key:
            return -1
        else:
            return 0

    def __inc__(self, key: str, value, *args, **kwargs) -> int:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'pool_source' in key or 'interp_target' in key or 'sampling_index' in key:
            if int(key[5]) == 0:
                return self.num_nodes
            else:
                return self[f'scale{int(key[5]) - 1}_sampling_index'].size(dim=0)
        elif 'index' in key or key == 'face' or key == 'tets':
            return self.num_nodes
        elif 'pool_target' in key or 'interp_source' in key:
            return self[f'scale{key[5]}_sampling_index'].size(dim=0)
        else:
            return 0
