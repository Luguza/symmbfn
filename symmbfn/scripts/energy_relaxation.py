from __future__ import annotations

import warnings
import os
import pickle
from pathlib import Path
import json
import numpy as np
import torch
from chgnet.model import CHGNet
from chgnet.model import StructOptimizer
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry, PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
import pandas as pd
import matgl
#from matgl.ext.ase import Relaxer
import warnings
#from hydra.experimental import compose
#from hydra import initialize_config_dir
from pymatgen.entries.compatibility import (
    MaterialsProject2020Compatibility,
    MaterialsProjectCompatibility,
)
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from p_tqdm import p_map
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import argparse


def process_structure(structure_dict, ppd, relaxer):
    try:
        structure = Structure.from_dict(structure_dict)
        relax_results = relaxer.relax(structure, verbose=False)
        final_structure = relax_results['final_structure']
        energy = relax_results['trajectory'].energies[-1]
        unrelaxed_energy = relax_results['trajectory'].energies[0]
        e_hull = get_e_hull_from_phase_diagram(ppd, structure)
        e_above_hull = (energy / structure.num_sites) - e_hull
        e_above_hull_unrelaxed = (unrelaxed_energy /structure.num_sites) - e_hull
        if e_above_hull < 0:
            print(structure)
        else:
            print(e_above_hull)
        return [structure_dict, final_structure.to_json(), energy, e_hull, e_above_hull, unrelaxed_energy, e_above_hull_unrelaxed]
    except:
        return [structure_dict, structure_dict, np.nan, np.nan, np.nan, np.nan, np.nan]


def get_patched_phase_diagram_mp(path: Path) -> PatchedPhaseDiagram:
    with open(path, "rb") as f:
        ppd_mp = pickle.load(f)
    return ppd_mp

def get_e_hull_from_phase_diagram(
    ppd_mp: PatchedPhaseDiagram,
    structure: Structure,
) -> float:
    """returns e_hull_per_atom"""
    try:
        return ppd_mp.get_hull_energy_per_atom(structure.composition)
    except (ValueError, AttributeError, ZeroDivisionError):
            return float("nan")


def load_config(model_path):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
    return cfg


def n_elem_sun_plot(model_path):
    df_path = model_path + "/sun.json"
    result_df = pd.read_json(df_path)
    samples = [Structure.from_dict(x) for x in  result_df['gen_structure']]
    pred_nelems = [len(set(c.species))
                       for c in samples]
    bins = [i+0.5 for i in range(1, 7)]
    plt.hist([pred_nelems,], label=['CrystalBFN'] , density=True,bins=bins, rwidth=0.8)
    plt.xlabel('#Elemets in S.U.N')
    plt.ylabel('normalized amount')
    plt.legend()
    fig_path = model_path + "/nelem_sun.png"
    plt.savefig(fig_path)
    plt.close()

def e_hull_plot(model_path):
    result_path = model_path + "/all_energies.json"
    result_df = pd.read_json(result_path)
    e_hull = result_df['corr_e_above_hull']
    bins = [i*0.01 for i in range(-20, 160)]
    plt.hist(e_hull, label=['CrystalBFN'], bins=bins)

    plt.xlabel('E above hull [eV/Atom]')
    plt.ylabel('Counts')
    plt.legend()
    fig_path = model_path + "/ehull.png"
    plt.savefig(fig_path)
    plt.close()


def main(model_path, n_samples):
    ppd_path = "" #enter path to patched phase diagram for mp_20 here
    ppd_mp = get_patched_phase_diagram_mp(ppd_path)
    df_path = model_path + "/eval_gen.json"
    metrics_out_file = model_path + "/eval_metrics.json"
    df = pd.read_json(df_path)
    num_samples = n_samples
    df = df.head(num_samples)
    relaxer = StructOptimizer()
    result = [process_structure(x, ppd_mp, relaxer) for x in df['gen_structure']]#[valid]['gen_structure']]

    result_df = pd.DataFrame(result, columns=['gen_structure', 'relax_structure', 'energy', 'e_hull', 'e_above_hull', 'energy_unrelaxed', 'e_above_hull_unrelaxed'])
    result_path = model_path + "/all_energies.json"
    result_df.to_json(result_path)
   
    stabel_df = result_df[(result_df['e_above_hull'] <= 0)]
    metastabel_df = result_df[(result_df['e_above_hull'] <= 0.1)]
    meta_stability = len(metastabel_df.index) / num_samples
    stability = len(stabel_df.index) / num_samples
    stabel_unrelaxed_df = result_df[(result_df['e_above_hull_unrelaxed'] <= 0)]
    metastabel_unrelaxed_df = result_df[(result_df['e_above_hull_unrelaxed'] <= 0.1)]
    meta_stability_unrelaxed = len(metastabel_unrelaxed_df.index) / num_samples
    stability_unrelaxed = len(stabel_unrelaxed_df.index) / num_samples
    all_metrics = {'stability': stability, 'metastability': meta_stability, 'stability_unrelaxed': stability_unrelaxed, 'meta_stability_unrelaxed': meta_stability_unrelaxed}
    print(all_metrics)
  

    with open(metrics_out_file, 'r') as f:
        written_metrics = json.load(f)
        written_metrics.update(all_metrics)
    with open(metrics_out_file, 'w') as f:
        json.dump(written_metrics, f)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--n_samples', required=True)

    args = parser.parse_args()
   
    print(args.root_path)

    subfolders = [entry.name for entry in os.scandir(args.root_path) if entry.is_dir()]
    print(subfolders)
    for folder in subfolders:
        model_path = os.path.join(args.root_path, folder)
        print(model_path)
        main(model_path, int(args.n_samples))
        

    
    
