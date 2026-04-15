import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from symmbfn.scripts.eval_utils import load_model, lattices_to_params_shape, get_crystals_list
from symmbfn.common.data_utils import lattice_from_ks
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
from pyxtal.symmetry import Wyckoff_position, search_cloest_wp, Group

import pymatgen.core as pmg
import numpy as np
from p_tqdm import p_map

import pdb
import warnings
import os
warnings.filterwarnings('ignore')




train_dist = {
    'perov_5' : [0, 0, 0, 0, 0, 1],
    'carbon_24' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp_20' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301]
}




def gen_crystals(loader, model):

    frac_coords = []
    num_atoms = []
    atom_types = []
    ks = []
    syms = []
    spacegroups = []
    input_data_list = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            start = time.time()
            spacegroups.append(batch.spacegroup)
            if torch.cuda.is_available():
                batch.cuda()
            outputs, traj = model(batch)
            print(time.time()-start)
            #new_frac_coords, new_num_atoms, new_atom_types, new_site_symm = modify_frac_coords_one(, , , 225)


            frac_coords.append(outputs['frac_coords'].detach().cpu())
            num_atoms.append(outputs['num_atoms'].detach().cpu())
            atom_types.append(outputs['atom_types'].detach().cpu())
            syms.append(outputs['syms'].detach().cpu())
            ks.append(outputs['ks'].detach().cpu())
            #if idx > 3:
            #    break
            
        spacegroups = torch.cat(spacegroups, dim=0)
        frac_coords = torch.cat(frac_coords, dim=0)
        num_atoms = torch.cat(num_atoms, dim=0)
        atom_types = torch.cat(atom_types, dim=0)
        syms = torch.cat(syms, dim=0)
        ks = torch.cat(ks, dim=0)


    return (
        frac_coords, atom_types, syms, ks, num_atoms, spacegroups
    )

class GenDataset(Dataset):

    def __init__(self, dataset, total_num, dist_path):
        super().__init__()
        self.total_num = total_num
        dist = torch.load(dist_path)
        self.spacegroup_distribution = dist['spacegroups']
        self.num_atom_distribution = dist['spacegroup_to_num']
        self.spacegroups = np.random.choice(len(self.spacegroup_distribution), total_num, p = self.spacegroup_distribution)
        self.num_atoms = []
        for i in range(0, total_num):
            self.num_atoms.append(np.random.choice(len(self.num_atom_distribution[self.spacegroups[i]]), 1, p = self.num_atom_distribution[self.spacegroups[i]])[0])
        self.is_carbon = dataset == 'carbon_24'

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        #num_atom = 2
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            prop = -2,
            spacegroup = torch.LongTensor([self.spacegroups[index]])
        )
        #if self.is_carbon:
        #    data.atom_types = torch.LongTensor([6] * num_atom)
        return data


def main(model_path, dist_path, n_samples):
    
    model_Path = Path(model_path)
    model, test_loader, cfg = load_model(
        model_Path,
        load_data=False,
    )
    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the model.')

    test_set = GenDataset('mp_20', n_samples, dist_path)
    test_loader = DataLoader(test_set, batch_size = 256)


    start_time = time.time()
    (frac_coords, atom_types, syms, ks, num_atoms, spacegroups) = gen_crystals(test_loader, model)

    #if args.label == '':
    gen_out_name = 'eval_gen.pt'
    #else:
    #    gen_out_name = f'eval_gen_{args.label}.pt'

    torch.save({
        'eval_setting': 'mp_20',
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'syms': syms,
        'ks': ks,
        'spacegroups': spacegroups
    }, str(model_path + "/" + gen_out_name))
      


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_samples', required=True)
    parser.add_argument('--dist_path', required=True)


    args = parser.parse_args()

    subfolders = [entry.name for entry in os.scandir(args.model_path) if entry.is_dir()]
    print(subfolders)
    for folder in subfolders:
        model_path = os.path.join(args.model_path, folder)
        print(model_path)
        main(model_path, args.dist_path, int(args.n_samples))