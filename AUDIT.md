# Repo audit — 2026-11-17

**Scope:** whole repo. **Load assumed:** one person, single-GPU Slurm jobs on kcist plus offline eval scripts.

**Verdict:** fix 1 first (then retrain), with 5 so the fix reaches kcist, then 2 and 3.
**Lean:** about -1,500 lines possible (about 1,190 to delete, about 300 from merging CSP/DNG).

**Not checked:**
- the network internals (`CSPLayer`, `radius_graph_pbc`)
- `energy_relaxation.py`
- most of `GenEval`'s metric code
- the other three dataset configs beyond a skim
- kcist's `.env` and caches

Nothing was run on the GPU. Finding 1 was tested on the first 300 MP-20 training crystals only.

---

## Must fix

- [ ] **1. The lattice code `ks` is wrong for monoclinic, triclinic and hexagonal crystals**
  ([symmbfn/common/data_utils.py:115-120](symmbfn/common/data_utils.py#L115-L120))
  - **What this is:** `lattice_to_ks` turns the 3×3 lattice into the 6 numbers the model learns, and `lattice_from_ks` turns them back.
  - **Problem:** It takes `logm(L)` of pymatgen's lattice matrix, which is not symmetric. The 6 symmetric basis matrices drop part of that log. On the first 300 MP-20 training crystals:
    - Converting to `ks` and back changes the lengths/angles by up to **13.6** for monoclinic/triclinic crystals and **0.78** for hexagonal ones. It is exact for orthorhombic, tetragonal and cubic.
    - **69/300** crystals break their own `sg_to_ks_mask`. Hexagonal crystals get `k0 = -0.268` instead of `-log(3)/4 = -0.275`, and `k3 = 0.072` instead of 0.
  - **Fix (tested):** use the symmetric factor of the polar decomposition:
    ```python
    S = scipy.linalg.logm(L @ L.T).real / 2
    ```
    On the same 300 crystals, the round-trip error drops to 1.6e-13, and 0/300 break their mask. Afterwards, delete the cached `data/*/*_ori.pt` files, here **and on kcist** (see 5).
  - **If we skip it:** Every model trained so far learned distorted lattices for these crystal systems, and the generation and CSP metrics carry the error.

- [ ] **2. CSP evaluation scores only the first 3 batches**
  ([symmbfn/scripts/evaluate.py:57-58](symmbfn/scripts/evaluate.py#L57-L58))
  - **What this is:** `csp()` loops over the test loader and reconstructs each crystal.
  - **Problem:** A leftover `if idx > 1: break` stops after 3 batches × 256 = 768 crystals.
  - **Fix:** Delete those two lines.
  - **If we skip it:** The reported match rate and RMS come from a subset.

- [ ] **3. Generation, evaluation and metrics crash on import**
  ([symmbfn/model/symmbfn_train.py:5](symmbfn/model/symmbfn_train.py#L5))
  - **What this is:** The setuptools shim that torch 1.9's tensorboard import needs.
  - **Problem:** Only the training script loads the shim. Without it, `import pytorch_lightning` fails with `AttributeError: module 'distutils' has no attribute 'version'` (reproduced in the `symmbfn` env).
  - **Fix:** Add `import setuptools._distutils.version  # noqa: F401` to `symmbfn/__init__.py`. Every other script imports `symmbfn` before `pytorch_lightning`.
  - **If we skip it:** You cannot evaluate any model.

- [ ] **4. Renamed `WANDB_DIR` breaks old `.env` files**
  ([README.md:29](README.md#L29), [conf/default.yaml:30](conf/default.yaml#L30))
  - **What this is:** Hydra reads `${oc.env:WANDB_DIR}` at startup and fails if the variable is not set.
  - **Problem:** The README and possibly kcist's `.env` still use `WABDB_DIR`.
  - **Fix:** Rename it in the README. On kcist, run `sed -i s/WABDB_DIR/WANDB_DIR/ ~/Masterthesis/symmbfn/.env`.
  - **If we skip it:** The next cluster job dies at startup.

## Should fix

- [ ] **5. The dataset cache never notices code or setting changes**
  ([symmbfn/data/dataset.py:36-37](symmbfn/data/dataset.py#L36-L37))
  - **What this is:** Preprocessing results are saved to `train_ori.pt` and reused.
  - **Problem:** The code checks only that the file exists. After fix 1, or a change to `niggli`, `tol` or `primitive`, it silently trains on the old data.
  - **Fix:** Put a version tag in `save_path` (e.g. `train_ks2.pt`) in the data configs, and change it whenever preprocessing changes.
  - **If we skip it:** Fix 1 has no effect on kcist until someone deletes the files by hand.

- [ ] **6. Evaluation loads the last checkpoint, not the best one**
  ([symmbfn/scripts/eval_utils.py:107-109](symmbfn/scripts/eval_utils.py#L107-L109))
  - **What this is:** `load_model` picks which `.ckpt` file to load.
  - **Problem:** It prefers `last.ckpt`. Early stopping waits 500 validation checks (every 5 epochs), so `last` can be up to 2,500 epochs past the best `val_loss`.
  - **Fix:** Skip `last` and load the `ModelCheckpoint` top-1 file.
  - **If we skip it:** The metrics describe a worse model than the one you trained.

- [ ] **7. Wrong weights load without an error**
  ([symmbfn/scripts/eval_utils.py:116](symmbfn/scripts/eval_utils.py#L116))
  - **What this is:** `load_from_checkpoint(..., strict=False)`.
  - **Problem:** If the config and checkpoint disagree (e.g. a renamed layer), the missing layers stay random and nothing warns you.
  - **Fix:** Remove `strict=False`. Also, `load_from_checkpoint` is a classmethod: the model instantiated just above it is built and thrown away.
  - **If we skip it:** You get garbage samples that look like a bad model.

- [ ] **8. Nothing is seeded**
  ([conf/train/default.yaml:3](conf/train/default.yaml#L3))
  - **What this is:** The config sets `random_seed: 42`.
  - **Problem:** No code reads it. These all change from run to run:
    - training
    - the space-group and atom-count sampling in generation ([generation.py:133-136](symmbfn/scripts/generation.py#L133-L136))
    - the 1,000-sample draw in `GenEval` ([compute_metrics.py:244](symmbfn/scripts/compute_metrics.py#L244))
  - **Fix:** Call `pl.seed_everything(cfg.train.random_seed)` in `main`. Call it with a fixed seed in the generation and metrics scripts.
  - **If we skip it:** You cannot reproduce a number in the thesis.

- [ ] **9. MP-20 is hard-coded in generation and metrics**
  ([generation.py:168](symmbfn/scripts/generation.py#L168), [generation.py:181](symmbfn/scripts/generation.py#L181), [compute_metrics.py:568](symmbfn/scripts/compute_metrics.py#L568))
  - **What this is:** The dataset name chooses the coverage cutoffs and the energy percentiles.
  - **Problem:** The README offers `data=perov`, but carbon/perov models are then scored with MP-20 cutoffs (struc 0.4 instead of 0.2).
  - **Fix:** Read `cfg.data.eval_model_name` from the loaded config, as `main` already does.
  - **If we skip it:** Coverage numbers for other datasets are wrong.

- [ ] **10. `data_augmentation` and `scaling` do nothing**
  ([symmbfn/data/dataset.py:62-65](symmbfn/data/dataset.py#L62-L65))
  - **What this is:** Two dataset options: shift the coordinates, and scale the lengths by `num_atoms^(1/3)`.
  - **Problem:** Training reads `asym_coords` and `ks`, not `frac_coords` or `lengths`. Turning either option on changes nothing.
  - **Fix:** Delete both options and their config keys.
  - **If we skip it:** An "augmentation" ablation would really be the same run twice.

- [ ] **11. The CSP and DNG copies must change together**
  ([bfn_csp.py](symmbfn/model/bfn/bfn_csp.py), [bfn_dng.py](symmbfn/model/bfn/bfn_dng.py), [model.py:35-411](symmbfn/model/model.py#L35-L411))
  - **What this is:** The two tasks share about 90% of the sampler, the loss and the Lightning code.
  - **Problem:** Any sampler fix must be made twice, and the two copies already differ: DNG `forward` reads `num_atoms`, CSP reads `asym_num_atoms`.
  - **Fix:** Move `interdependency_modeling`, the coordinate/lattice/symmetry update steps and `validation_step` into `bfnBase`/`BaseModule`. Keep only the atom-type parts in DNG.
  - **If we skip it:** The two variants drift apart.

- [ ] **12. The lattice encoding has no test**
  ([symmbfn/common/data_utils.py:115-155](symmbfn/common/data_utils.py#L115-L155))
  - **What this is:** The `ks` math is the core of the symmetry handling.
  - **Problem:** Finding 1 went unnoticed because nothing checks it.
  - **Fix:** Add one test over about 50 MP-20 crystals: the lattice parameters survive the round trip, and `mask_ks` leaves `ks` unchanged. This sketch was used to verify fix 1:
    ```python
    for _, row in pd.read_csv("data/mp_20/train.csv", nrows=50).iterrows():
        crystal, info = get_symmetry_info(build_crystal(row["cif"]))
        ks = lattice_to_ks(crystal.lattice.matrix)
        rebuilt = Lattice(lattice_from_ks(ks).numpy())
        assert np.allclose(rebuilt.parameters, crystal.lattice.parameters, atol=1e-6)
        ks_mask, ks_add = sg_to_ks_mask(torch.tensor([info["spacegroup"]]))
        ks_tensor = torch.tensor(ks)[None]
        assert torch.allclose(mask_ks(ks_tensor, ks_mask.double(), ks_add.double()), ks_tensor, atol=1e-4)
    ```
  - **If we skip it:** The next lattice change can break it again without a warning.

- [ ] **13. Dead, broken `test_step`**
  ([model.py:119-158](symmbfn/model/model.py#L119-L158), [model.py:309-348](symmbfn/model/model.py#L309-L348))
  - **What this is:** Code copied from a molecule BFN.
  - **Problem:** It uses `self.cfg`, `batch.charges` and `self.charge_decode`, none of which exist. It crashes if `trainer.test` is ever called.
  - **Fix:** Delete it (80 lines).
  - **If we skip it:** It misleads anyone who reads it.

- [ ] **14. About 1,100 lines of unused code** (no callers found by grep)
  - [symmbfn/model/callbacks/basic.py](symmbfn/model/callbacks/basic.py): never imported (471 lines).
  - 12 functions in `data_utils.py` (403 lines): `min_distance_sqr_pbc`, `mard`, `batch_accuracy_precision_recall`, `refine_spacegroup`, `lengths_angles_to_volume`, `lattice_matrix_to_params`, `cart_to_frac_coords`, `radius_graph_pbc_wrapper`, `radius_graph_pbc_`, `preprocess_tensors`, `add_scaled_lattice_prop`, `get_scaler_from_data_list`.
  - `TensorCrystDataset` + `prop_model_eval`: would crash anyway. They unpack 7 values, but `build_crystal_graph` now returns 8.
  - `compute_metrics.main(args)`: reads `args.tasks`, but the parser never defines it.
  - `train_dist` in [generation.py:32](symmbfn/scripts/generation.py#L32), `recommand_step_lr` in [eval_utils.py:32](symmbfn/scripts/eval_utils.py#L32).
  - `bfnBase`: `dtime4continuous_loss`, `ctime4discreteised_loss`, `dtime_discrete_loss`, and `continuous_cyclic_var_bayesian_update`, which wraps only the noise, not the mean.
  - The unused final `crystal = Structure(...)` in `get_symmetry_info`.
  - **Fix:** Delete them.

## Nice to have

- [ ] **15. Duplicate epoch-loss tracking**
  ([model.py:101](symmbfn/model/model.py#L101), [model.py:218-225](symmbfn/model/model.py#L218-L225))
  - `train_losses` collects every step's loss with its autograd graph attached and prints the mean. `log_dict(on_epoch=True)` already logs the same value.
  - **Fix:** Delete the list and `on_train_epoch_end` in both models.

- [ ] **16. The README defaults table is wrong**
  ([README.md:74](README.md#L74))
  - It says `beta1` is 0.25 and `beta1_sym` is 2.5. The config has 0.75 and 1.5.

- [ ] **17. No Slurm log folder in fresh copies**
  ([slurm/train.sbatch:9](slurm/train.sbatch#L9))
  - `logs/` is gitignored, and Slurm does not create the `--output` folder. On a new machine, the first job fails and writes no log.
  - **Fix:** Add `mkdir -p slurm/logs` to the usage comment.

- [ ] **18. Trim the setuptools shim comment**
  ([symmbfn/model/symmbfn_train.py:1-4](symmbfn/model/symmbfn_train.py#L1-L4))
  - Make it one line. The history belongs in the commit message.

Left out as minor:
- `RecEval` builds a validity list it never uses.
- `.squeeze()` in the samplers breaks for a 1-atom batch.
- `view(-1, 15, 13)` ignores `max_axes`/`max_syms`.
