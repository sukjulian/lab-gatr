import torch_geometric as pyg
import torch
import os
from glob import glob
import h5py
from tqdm import tqdm
from torch_geometric.data import Data
from pathlib import Path


class Dataset(pyg.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        self.path_to_hdf5_file = sorted(glob(os.path.join(self.root, "raw", "*.hdf5")))[0]

        with h5py.File(self.path_to_hdf5_file, 'r') as hdf5_file:
            sample_ids = [os.path.join(self.path_to_hdf5_file, sample_id) for sample_id in hdf5_file]

        return [os.path.relpath(sample_id, os.path.join(self.root, "raw")) for sample_id in sample_ids]

    @property
    def processed_file_names(self):
        return [f"data_{idx}.pt" for idx in range(len(self.raw_file_names))]

    def download(self):
        return

    def process(self):
        for idx, path in enumerate(tqdm(self.raw_paths, desc="Reading & transforming", leave=False)):
            data = self.read_hdf5_data(path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, self.processed_file_names[idx]))

    @staticmethod
    def read_hdf5_data(path):
        path_to_hdf5_file, sample_id = os.path.split(path)

        with h5py.File(path_to_hdf5_file, 'r') as hdf5_file:

            data = Data(
                y=torch.from_numpy(hdf5_file[sample_id]['velocity'][()]),
                pos=torch.from_numpy(hdf5_file[sample_id]['pos_tets'][()]),
                tets=torch.from_numpy(hdf5_file[sample_id]['tets'][()].T),
                inlet_index=torch.from_numpy(hdf5_file[sample_id]['inlet_idcs'][()]),
                lumen_wall_index=torch.from_numpy(hdf5_file[sample_id]['lumen_wall_idcs'][()]),
                outlets_index=torch.from_numpy(hdf5_file[sample_id]['outlets_idcs'][()])
            )

        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)

        return data
