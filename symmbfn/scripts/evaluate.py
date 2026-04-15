import time
import argparse
import torch
import hydra
import os
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data.batch import Batch
from hydra import initialize_config_dir
from hydra.experimental import compose




from symmbfn.scripts.eval_utils import load_model, lattices_to_params_shape, get_crystals_list


def csp(loader, model, num_evals):

    frac_coords = []
    num_atoms = []
    atom_types = []
    syms = []
    ks = []
    spacegroups = []
    input_data_list = []
    input_frac_coords = torch.tensor([])
    input_lengths = torch.tensor([])
    input_angles = torch.tensor([])
    input_num_atoms = torch.tensor([], dtype=int)
    input_atom_types = torch.tensor([])
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            input_num_atoms = torch.cat((input_num_atoms,batch.num_atoms))
            input_angles = torch.cat((input_angles,batch.angles))
            input_lengths = torch.cat((input_lengths, batch.lengths))
            input_frac_coords = torch.cat((input_frac_coords, batch.frac_coords))
            input_atom_types = torch.cat((input_atom_types, batch.atom_types))
            start = time.time()
            spacegroups.append(batch.spacegroup)
            atom_types.append(batch.asym_species)
            num_atoms.append(batch.asym_num_atoms)
            if torch.cuda.is_available():
                batch.cuda()
            outputs, traj = model(batch)
            print(time.time()-start)
            #new_frac_coords, new_num_atoms, new_atom_types, new_site_symm = modify_frac_coords_one(, , , 225)

            frac_coords.append(outputs['frac_coords'].detach().cpu())
            syms.append(outputs['syms'].detach().cpu())
            ks.append(outputs['ks'].detach().cpu())

            
            #input_data_list = input_data_list + batch.to_data_list()
            if idx > 1:
                break

        spacegroups = torch.cat(spacegroups, dim=0)
        frac_coords = torch.cat(frac_coords, dim=0)
        num_atoms = torch.cat(num_atoms, dim=0)
        atom_types = torch.cat(atom_types, dim=0)
        syms = torch.cat(syms, dim=0)
        ks = torch.cat(ks, dim=0)


    input_data_batch = {'frac_coords': input_frac_coords, 
                        'num_atoms': input_num_atoms,
                        'atom_types': input_atom_types,
                        'lengths': input_lengths,
                        'angles': input_angles}
    return (
        frac_coords, atom_types, syms, ks, num_atoms, spacegroups, input_data_batch
    )


def run(model_path):
    model_Path = Path(model_path)
    model, test_loader, cfg = load_model(
        model_Path,
        load_data=True,
    )
    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the model.')
   
    (frac_coords, atom_types, syms, ks, num_nodes, spacegroups, input_data_batch) = csp(
        test_loader, model, 1)
    file_path = model_path + "/eval_csp.pt"

    torch.save({
        #'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_nodes,
        'atom_types': atom_types,
        'syms': syms,
        'ks': ks,
        'spacegroups': spacegroups
        #'time': time.time() - start_time,
    }, file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
  
    args = parser.parse_args()

    
    subfolders = [entry.name for entry in os.scandir(args.model_path) if entry.is_dir()]
    print(subfolders)
    for folder in subfolders:
        model_path = os.path.join(args.model_path, folder)
        print(model_path)
        run(model_path)