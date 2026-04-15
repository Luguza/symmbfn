from collections import Counter
import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import matplotlib.cm as cm
from itertools import cycle
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import matplotlib.pyplot as plt
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Wyckoff_position, search_cloest_wp, Group
from symmbfn.scripts.eval_utils import load_model, lattices_to_params_shape, get_crystals_list
from symmbfn.common.data_utils import lattice_from_ks
import pickle
from scipy.spatial.distance import jensenshannon
from collections import Counter

from symmbfn.scripts.eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, get_data_crystals_list,prop_model_eval, compute_cov)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

SITE_SYMM_AXES = 15
SITE_SYMM_PGS = 13
SITE_SYMM_DIM = SITE_SYMM_AXES * SITE_SYMM_PGS
SG_SYM = {spacegroup: Group(spacegroup) for spacegroup in range(1, 231)}
SG_TO_WP_TO_SITE_SYMM = dict()
for spacegroup in range(1, 231):
    SG_TO_WP_TO_SITE_SYMM[spacegroup] = dict()
    for wp in SG_SYM[spacegroup].Wyckoff_positions:
        wp.get_site_symmetry()
        SG_TO_WP_TO_SITE_SYMM[spacegroup][wp] = wp.get_site_symmetry_object().to_one_hot()

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

def modify_frac_coords_one(frac_coords, site_symm, atom_types, spacegroup):
    spacegroup = spacegroup.item()
    site_symm_axis = site_symm.reshape(-1, SITE_SYMM_AXES, SITE_SYMM_PGS).to(dtype=torch.float32)
    # Get site symmetry of each WP for the spacegroup
    wp_to_site_symm = SG_TO_WP_TO_SITE_SYMM[spacegroup]

    # iterate over frac coords and corresponding site-symm
    new_frac_coords, new_atom_types, new_site_symm = [], [], []
    for (sym, frac_coord, atm_type) in zip(site_symm_axis, frac_coords, atom_types):
        frac_coord = frac_coord.numpy()
        
        # Get all WPs that are closest in terms of site symmetry
        wp_to_ss_dist = {wp: torch.norm(sym.flatten() - ss.flatten()) for wp, ss in wp_to_site_symm.items()}
        min_ss_dist = min(wp_to_ss_dist.values())
        closest_ss_wps = [wp for wp, dist in wp_to_ss_dist.items() if dist==min_ss_dist]
            
        # For each WP find closest position in space
        closes = []
        for wp in closest_ss_wps:
            for orbit_index in range(len(wp.ops)):
                close = search_cloest_wp(SG_SYM[spacegroup], wp, wp.ops[orbit_index], frac_coord)%1.
                closes.append((close, wp, orbit_index, np.linalg.norm(np.minimum((close - frac_coord)%1., (frac_coord - close)%1.))))
        #try:
            # pick the nearest wp to project
        closest = sorted(closes, key=lambda x: x[-1])[0]
        wyckoff = closest[1]
        repr_index = closest[2]
        # use wp operations on frac_coord
        frac_coord = closest[0]
        for index in range(len(wyckoff)):
            coords = wyckoff[(index + repr_index) % len(wyckoff)].operate(frac_coord)%1.
            #if new_frac_coords.size == 0:
            new_frac_coords.append(coords)
            new_atom_types.append(atm_type)
            new_site_symm.append(sym)
            
    new_frac_coords = np.stack(new_frac_coords)
    new_atom_types = np.stack(new_atom_types)
    new_site_symm = np.stack(new_site_symm)
    return new_frac_coords, len(new_frac_coords), new_atom_types, new_site_symm

class Crystal(object):

    def __init__(self, crys_array_dict, dataset=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        if not dataset:
            self.syms = crys_array_dict['syms']
            self.ks = crys_array_dict['ks']
            self.spacegroup = crys_array_dict['spacegroup']
            self.lattice = lattice_from_ks(self.ks)
            self.lengths, self.angles = lattices_to_params_shape(self.lattice)
            self.frac_coords, num_atoms , self.atom_types, self.syms = modify_frac_coords_one(crys_array_dict['frac_coords'], crys_array_dict['syms'], crys_array_dict['atom_types'], self.spacegroup)
        else:
            self.lengths = crys_array_dict['lengths']
            self.angles = crys_array_dict['angles']

        self.dict = crys_array_dict
        #if len(self.atom_types.shape) > 1:
        #    self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
        #    self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
                analyzer = SpacegroupAnalyzer(self.structure, symprec= 0.1, angle_tolerance= 5)
                self.spacegroup = analyzer.get_space_group_number()
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = 'unrealistically_small_lattice'
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
                self.spacegroup = 0
           

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        try:
            self.comp_valid = smact_validity(self.elems, self.comps)
            if self.constructed:
                self.struct_valid = structure_validity(self.structure)
            else:
                self.struct_valid = False
            self.valid = self.comp_valid and self.struct_valid
        except Exception:
            self.valid = False
            self.comp_valid = False
            self.struct_valid = False
        

    def get_fingerprints(self):
        
        try:
            elem_counter = Counter(self.atom_types)
            comp = Composition(elem_counter)
            self.comp_fp = CompFP.featurize(comp)
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_valid = False

            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)



class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            #if not is_valid:
            #    print('fail')
            #    return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                if rms_dist is None:
                    print(pred.structure)
                    print(gt.structure)
                return rms_dist
            except Exception:
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]
        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}     

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics



class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=10, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
            f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')
        self.all_valid = valid_crys
        self.all_struc_valid = [c for c in pred_crys if c.struct_valid]

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_spacegroup_wdist(self):
        pred_spacegroup = [c.spacegroup for c in self.valid_samples]
        gt_spacegroup= [c.spacegroup for c in self.gt_crys]
        wdist_spacegroup = wasserstein_distance(pred_spacegroup, gt_spacegroup)
        return {'wdist_spacegroup': wdist_spacegroup}
    
    def spacegroup_jsd(self, pred_spacegroup, gt_spacegroup): 
        freq1 = Counter(pred_spacegroup)
        freq2 = Counter(gt_spacegroup)
        
        # Step 2: Create a union of all unique space groups
        all_space_groups = set(freq1.keys()).union(set(freq2.keys()))
        
        # Step 3: Convert counts to probability distributions
        prob1 = np.array([freq1[group] for group in all_space_groups], dtype=float)
        prob2 = np.array([freq2[group] for group in all_space_groups], dtype=float)
        
        # Normalize to get probabilities
        prob1 /= prob1.sum()
        prob2 /= prob2.sum()
        
        # Step 4: Compute Jensen-Shannon distance
        jsd = jensenshannon(prob1, prob2)
        return jsd
    
    def get_spacegroup_jsd(self):
        pred_spacegroup = [c.spacegroup for c in self.valid_samples]
        gt_spacegroup= [c.spacegroup for c in self.gt_crys]
        pred_spacegroup_all = [c.spacegroup for c in self.all_valid]
        pred_spacegroup_struct_valid = [c.spacegroup for c in self.all_struc_valid]
        pred_spacegroup_total = [c.spacegroup for c in self.crys]

        jsd = self.spacegroup_jsd(pred_spacegroup, gt_spacegroup)
        jsd_all = self.spacegroup_jsd(pred_spacegroup_all, gt_spacegroup)
        jsd_all_s = self.spacegroup_jsd(pred_spacegroup_struct_valid, gt_spacegroup)
        jsd_total = self.spacegroup_jsd(pred_spacegroup_total, gt_spacegroup)

        return {'jsd_spacegroup': jsd,
                'jsd_spacegroup_all': jsd_all,
                'jsd_spacegroup_all_s': jsd_all_s,
                'jsd_spacegroup_total': jsd_total}


    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict
    
    def densitiy_plot(self, model_path):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        bins = [i+0.5 for i in range(0, 25)]
        plt.hist([pred_densities, gt_densities], label=['CrystalBFN','test set'], density=True, bins=bins)

        plt.xlabel('density')
        plt.ylabel('normalized amount')
        plt.legend()
        fig_path = model_path + "/density.png"
        plt.savefig(fig_path)
        plt.close()


    def n_elem_plot(self, model_path):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        bins = [i+0.5 for i in range(0, 9)]
        plt.hist([pred_nelems, gt_nelems], label=['CrystalBFN','test set'] , density=True,bins=bins)
        plt.xlabel('#Elemets')
        plt.ylabel('normalized amount')
        plt.legend()
        fig_path = model_path + "/nelem.png"
        plt.savefig(fig_path)
        plt.close()

    def spacegroup_plot(self, model_path):
        class_labels = [c.spacegroup for c in self.valid_samples]
        #gt_spacegroup = [c.spacegroup for c in self.gt_crys]
        categories = [i for i in range(1,231)]  # Sort for consistent order
        counts = [class_labels.count(category) for category in categories]
        total = len(class_labels)
        percentages = [(count / total) * 100 for count in counts]

        # Use Matplotlib's tab10 colormap for distinct colors
        distinct_colors = plt.cm.tab20.colors  # Get 10 distinct colors
        color_cycle = cycle(distinct_colors)  # Repeat colors if needed

        # Assign colors to categories, ensuring adjacent distinction
        colors = [next(color_cycle) for _ in categories]

        # Plot a single stacked bar
        fig, ax = plt.subplots(figsize=(12, 1))  # Wider bar for better proportions
        left = 0

        for percentage, category, color in zip(percentages, categories, colors):
            ax.barh([0], [percentage], left=left, color=color, edgecolor='none')
            left += percentage

        # Add class labels directly on the bar
        left = 0
        for percentage, category in zip(percentages, categories):
            center = left + (percentage / 2)
            ax.text(center, 0, category, ha='center', va='center', fontsize=8, color='black', rotation=0)
            left += percentage

        # Remove y-axis and frame for a cleaner look
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_xlim(0, 100)  # Ensure percentages fit perfectly
        ax.axis('off')  # Turn off all axes

        # Show the plot
        plt.tight_layout()
        fig_path = model_path + "/spacegroup.png"
        plt.savefig(fig_path)
        plt.close()
    
    def eform_plot(self, model_path):
        pred_spacegroup = [c.spacegroup for c in self.valid_samples]
        gt_spacegroup = [c.spacegroup for c in self.gt_crys]
        bins = [i+0.5 for i in range(0, 240)]
        plt.hist([pred_spacegroup, gt_spacegroup], label=['CrystalBFN','test set'] , density=True,bins=bins)
        plt.xlabel('spacegroup')
        plt.ylabel('normalized amount')
        plt.legend()
        fig_path = model_path + "/spacegroup.png"
        plt.savefig(fig_path)
        plt.close()
        print(pred_spacegroup)

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_spacegroup_jsd())
        metrics.update(self.get_coverage())
        return metrics





def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['syms'],
            data['ks'],
            data['spacegroups'],
            data['num_atoms'])        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_data_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_data_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list

def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict, dataset=True) 


def main(args):
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name


    if 'gen' in args.tasks:
        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        if 'recon' not in args.tasks:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)
        csv = pd.read_csv(args.gt_file)
        gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)


    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)

def save_results(batch_results, batch_index, folder_path):
    filename = os.path.join(folder_path, f"crystals_batch_{batch_index}.pkl")
    with open(filename, "wb") as file:
        pickle.dump(batch_results, file)
    print(f"Batch {batch_index} saved to {filename}")

def run(path, gt_path):
    data_path = path + "/eval_gen.pt"
    df_path = path + "/eval_gen.json"
    crys_array_list, _ = get_crystal_array_list(data_path, batch_idx = -2)
    gen_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=32)   

    #gen_crys = [Crystal(x) for x in crys_array_list]
          
    gen = [{'gen_structure': json.loads(x.structure.to_json()),'n-ary': x.structure.n_elems ,'struct_valid':x.struct_valid, 'comp_valid':x.comp_valid} for x in gen_crys if x.constructed]

    with open(df_path,'w') as f:
        json.dump(gen, f)

    csv = pd.read_csv(gt_path)
    gt_crys = p_map(get_gt_crys_ori, csv['cif'])
    #gt_crys = [get_gt_crys_ori(x) for x in csv['cif']]
    print(len(gt_crys))
    eval = GenEval(gen_crys, gt_crys, n_samples=1000, eval_model_name='mp20')
    gen_metrics = eval.get_metrics()
    eval.n_elem_plot(path)
    eval.densitiy_plot(path)
    eval.spacegroup_plot(path)
    metrics_out_file = path + "/eval_metrics.json"
    with open(metrics_out_file, 'w') as f:
        json.dump(gen_metrics, f)

    print(gen_metrics)


def run_csp(path):
    recon_file_path = path + "/eval_csp.pt"
    crys_array_list, true_crystal_array_list = get_crystal_array_list(
        recon_file_path, batch_idx=-2)
    gt_crys = p_map(lambda x: Crystal(x, dataset=True), true_crystal_array_list)
    pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
    #for i in range(len(crys_array_list)):
    #    print(f"Processing batch {i}")
    #    pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))   

    rec_evaluator = RecEval(pred_crys, gt_crys)
    recon_metrics = rec_evaluator.get_metrics()
    metrics_out_file = path + "/eval_metrics.json"
    with open(metrics_out_file, 'w') as f:
        json.dump(recon_metrics, f)
    print(recon_metrics)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--gt_path', required=True)
    args = parser.parse_args()
    print(args.root_path)
    subfolders = [entry.name for entry in os.scandir(args.root_path) if entry.is_dir()]
    print(subfolders)
    for folder in subfolders:
        model_path = os.path.join(args.root_path, folder)
        print(model_path)
        run(model_path, args.gt_path)