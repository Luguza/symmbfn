import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import numpy as np
from torch_geometric.data import Data, Batch
import os
from symmbfn.common.utils import PROJECT_ROOT
from symmbfn.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode, save_path: ValueNode,
                 data_augmentation: ValueNode, scaling: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.data_augmentation = data_augmentation
        self.scaling = scaling

        self.preprocess(save_path, preprocess_workers, prop)


    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=True)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, ks, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        frac_coords=torch.Tensor(frac_coords)
        if self.data_augmentation:
            frac_coords = (frac_coords + torch.rand((3))) %1.0
        if self.scaling:
            lengths = lengths / (num_atoms**(1/3))

        data = Data(
            frac_coords=frac_coords,
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=torch.tensor([data_dict['asym_num_atoms']]), # special attribute used for batching in pytorch geometric
            prop = float(data_dict[self.prop]),
            spacegroup= int(data_dict['spacegroup']),
            asym_species = torch.LongTensor(data_dict['asym_species']),
            asym_coords = torch.Tensor(data_dict['asym_coords']),
            site_syms = torch.LongTensor(data_dict['site_syms']),
            asym_num_atoms = data_dict['asym_num_atoms'],
            ks = torch.Tensor(ks)
            
        )
        
        
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        
    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
