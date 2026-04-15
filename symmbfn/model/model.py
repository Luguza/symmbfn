import pytorch_lightning as pl
import torch
import copy
import hydra
from symmbfn.scripts.eval_utils import lattices_to_params_shape
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._data = self.hparams.data
            self._hparams = self.hparams.model


    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}





class CrystalDNG(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        #self.cfg = config
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        # self.logger.log_hyperparams(self.cfg.todict())
        self.cost_coord = self.hparams.cost_coord
        self.cost_type= self.hparams.cost_type
        self.cost_lattice = self.hparams.cost_lattice
        self.cost_sym = self.hparams.cost_sym
        self.val_repeats = self.hparams.val_repeats

        self.bfn = hydra.utils.instantiate(self.hparams.bfn)

    def forward(self, data):
        outputs ={}
        segment_ids = data.batch
        num_atoms = data.num_atoms
        num_crystals = num_atoms.size(0)
        prop = data.prop
        spacegroup = data.spacegroup
        n_nodes = data.batch.shape[0]
        theta_chain = self.bfn(
                n_nodes=n_nodes,
                num_atoms=num_atoms,
                num_crystals=num_crystals,
                segment_ids=segment_ids,
                spacegroup=spacegroup,
                prop=prop,
            )
        outputs['num_atoms'] = num_atoms
        outputs['frac_coords'], h, lattices, syms = theta_chain[-1]
        outputs['atom_types'] = torch.argmax(h, dim=1)
        outputs['ks'] = lattices
        outputs['syms'] = syms

        return outputs, theta_chain

    def training_step(self, data, batch_idx):
        
        segment_ids = data.batch
        frac_coords = data.asym_coords
        lengths = data.lengths
        angles = data.angles
        num_atoms = data.asym_num_atoms
        types = data.asym_species
        num_crystals = num_atoms.size(0)
        spacegroup = data.spacegroup
        site_syms= data.site_syms
        ks = data.ks
        prop = data.prop

        #different time for all crystals
        t = torch.rand([num_crystals, 1], dtype=frac_coords.dtype, device=types.device)
        
        pos_loss, type_loss, lattice_loss, sym_loss, _ = self.bfn.loss_one_step(
            t, x=types, pos=frac_coords, ks=ks, num_atoms=num_atoms, spacegroup=spacegroup, site_syms=site_syms, segment_ids=segment_ids, prop=prop
        )
        sym_loss = torch.mean(sym_loss)
        pos_loss = torch.mean(pos_loss)
        type_loss = torch.mean(type_loss)
        lattice_loss = torch.mean(lattice_loss)
        loss = self.cost_coord * pos_loss + self.cost_type * type_loss + self.cost_lattice * lattice_loss + self.cost_sym *sym_loss

        self.train_losses.append(loss)

        log_dict = {
            'loss': loss,
            'lattice_loss': lattice_loss,
            'coord_loss': pos_loss,
            'type_loss': type_loss,
            'sym_loss': sym_loss
        }

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        sample_steps = self.cfg.bfn.sample_steps
        n_nodes = batch.pos.shape[0]
        h, charges, x, edge_index, segment_ids = (
            batch.x,  # [n_nodes, n_features]
            batch.charges,  # [n_nodes, 1]
            batch.pos,  # [n_nodes, 3]
            batch.edge_index,  # [2, edge_num]
            batch.batch,  # [n_nodes]
        )
        num_molecules = batch.idx.shape[0]
        # print("train_step",batch.charges)
        h = charges
        # print("h",h,h[:,-1:])
        in_data = []
        out_data = []
        for i in range(1, sample_steps + 1):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            t = torch.clamp(t, min=self.cfg.dynamics.t_min)
            posloss, charge_loss, assets = self.dynamics.loss_one_step(
                t, x=h, pos=x, edge_index=edge_index, segment_ids=segment_ids
            )
            (mu_coord, mu_charge, coord_pred, k_hat, gamma_coord, gamma_charge) = assets
            in_atom_type = self.charge_decode(mu_charge)
            out_atom_type = self.charge_decode(k_hat)

            in_batch = copy.deepcopy(batch)
            out_batch = copy.deepcopy(batch)

            in_batch.x, in_batch.pos = in_atom_type, mu_coord
            out_batch.x, out_batch.pos = out_atom_type, coord_pred

            in_data_list = in_batch.to_data_list()
            out_data_list = out_batch.to_data_list()
            in_data.append(in_data_list)
            out_data.append(out_data_list)
        in_data = list(zip(*in_data))
        out_data = list(zip(*out_data))
        assert len(in_data) == len(out_data) == num_molecules
        return {"in_data": in_data, "out_data": out_data}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        types, frac_coords, lengths, angles, segment_ids = (
            batch.asym_species,  # [n_nodes, n_features]
            batch.asym_coords,  # [n_nodes, 3]
            batch.lengths,
            batch.angles,
            batch.batch,  # [n_nodes]
        )
        
        spacegroup = batch.spacegroup
        site_syms= batch.site_syms
        ks = batch.ks
        prop = batch.prop
        num_atoms = batch.asym_num_atoms
        num_crystals = num_atoms.size(0)

        batch_pos_loss = torch.tensor([], device=DEVICE)
        batch_type_loss = torch.tensor([], device=DEVICE)
        batch_lattice_loss = torch.tensor([], device=DEVICE)
        batch_sym_loss = torch.tensor([], device=DEVICE)


        for _ in range(self.val_repeats):
            t = torch.rand(
                    [num_crystals, 1], dtype=frac_coords.dtype, device=types.device
                )
            pos_loss, type_loss, lattice_loss, sym_loss, _ = self.bfn.loss_one_step(
                t, x=types, pos=frac_coords, ks=ks, num_atoms=num_atoms, spacegroup=spacegroup, site_syms=site_syms, segment_ids=segment_ids, prop=prop
            )
            batch_pos_loss = torch.cat((batch_pos_loss, pos_loss))
            batch_type_loss = torch.cat((batch_type_loss, type_loss))
            batch_lattice_loss = torch.cat((batch_lattice_loss, lattice_loss))
            batch_sym_loss = torch.cat((batch_sym_loss, sym_loss))


        pos_loss = torch.mean(batch_pos_loss)
        type_loss = torch.mean(batch_type_loss)
        lattice_loss = torch.mean(batch_lattice_loss)
        sym_loss = torch.mean(batch_sym_loss)
        loss = self.cost_coord * pos_loss + self.cost_type * type_loss + self.cost_lattice * lattice_loss +  self.cost_sym * sym_loss

        log_dict = {
            'val_loss': loss,
            'val_lattice_loss': lattice_loss,
            'val_coord_loss': pos_loss,
            'val_type_loss': type_loss,
            'val_sym_loss': sym_loss,
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        
        self.train_losses = []

    

class CrystalCSP(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.train_losses = []
        # self.logger.log_hyperparams(self.cfg.todict())
        self.cost_coord = self.hparams.cost_coord
        self.cost_lattice = self.hparams.cost_lattice
        self.cost_sym = self.hparams.cost_sym
        self.val_repeats = self.hparams.val_repeats

        self.bfn = hydra.utils.instantiate(self.hparams.bfn)

    def forward(self, data):
        outputs ={}
        segment_ids = data.batch
        num_atoms = data.asym_num_atoms
        num_crystals = num_atoms.size(0)
        prop = data.prop
        spacegroup = data.spacegroup
        n_nodes = data.batch.shape[0]
        theta_chain = self.bfn(
                n_nodes=n_nodes,
                num_atoms=num_atoms,
                num_crystals=num_crystals,
                segment_ids=segment_ids,
                spacegroup=spacegroup,
                types=data.asym_species,
                prop=prop,
            )
        outputs['num_atoms'] = num_atoms
        outputs['frac_coords'], h, lattices, syms = theta_chain[-1]
        outputs['atom_types'] = data.asym_species
        outputs['ks'] = lattices
        outputs['syms'] = syms

        return outputs, theta_chain

    def training_step(self, data, batch_idx):
        
        segment_ids = data.batch
        frac_coords = data.asym_coords
        lengths = data.lengths
        angles = data.angles
        num_atoms = data.asym_num_atoms
        types = data.asym_species
        num_crystals = num_atoms.size(0)
        spacegroup = data.spacegroup
        site_syms= data.site_syms
        ks = data.ks
        prop = data.prop

        #different time for all crystals
        t = torch.rand([num_crystals, 1], dtype=frac_coords.dtype, device=types.device)
        
        pos_loss, lattice_loss, sym_loss, _ = self.bfn.loss_one_step(
            t, x=types, pos=frac_coords, ks=ks, num_atoms=num_atoms, spacegroup=spacegroup, site_syms=site_syms, segment_ids=segment_ids, prop=prop
        )
        sym_loss = torch.mean(sym_loss)
        pos_loss = torch.mean(pos_loss)
        lattice_loss = torch.mean(lattice_loss)
        loss = self.cost_coord * pos_loss  + self.cost_lattice * lattice_loss + self.cost_sym *sym_loss

        self.train_losses.append(loss)

        log_dict = {
            'loss': loss,
            'lattice_loss': lattice_loss,
            'coord_loss': pos_loss,
            'sym_loss': sym_loss
        }

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        sample_steps = self.cfg.bfn.sample_steps
        n_nodes = batch.pos.shape[0]
        h, charges, x, edge_index, segment_ids = (
            batch.x,  # [n_nodes, n_features]
            batch.charges,  # [n_nodes, 1]
            batch.pos,  # [n_nodes, 3]
            batch.edge_index,  # [2, edge_num]
            batch.batch,  # [n_nodes]
        )
        num_molecules = batch.idx.shape[0]
        # print("train_step",batch.charges)
        h = charges
        # print("h",h,h[:,-1:])
        in_data = []
        out_data = []
        for i in range(1, sample_steps + 1):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            t = torch.clamp(t, min=self.cfg.dynamics.t_min)
            posloss, charge_loss, assets = self.bfn.loss_one_step(
                t, x=h, pos=x, edge_index=edge_index, segment_ids=segment_ids
            )
            (mu_coord, mu_charge, coord_pred, k_hat, gamma_coord, gamma_charge) = assets
            in_atom_type = self.charge_decode(mu_charge)
            out_atom_type = self.charge_decode(k_hat)

            in_batch = copy.deepcopy(batch)
            out_batch = copy.deepcopy(batch)

            in_batch.x, in_batch.pos = in_atom_type, mu_coord
            out_batch.x, out_batch.pos = out_atom_type, coord_pred

            in_data_list = in_batch.to_data_list()
            out_data_list = out_batch.to_data_list()
            in_data.append(in_data_list)
            out_data.append(out_data_list)
        in_data = list(zip(*in_data))
        out_data = list(zip(*out_data))
        assert len(in_data) == len(out_data) == num_molecules
        return {"in_data": in_data, "out_data": out_data}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        types, frac_coords, lengths, angles, segment_ids = (
            batch.asym_species,  # [n_nodes, n_features]
            batch.asym_coords,  # [n_nodes, 3]
            batch.lengths,
            batch.angles,
            batch.batch,  # [n_nodes]
        )
        
        spacegroup = batch.spacegroup
        site_syms= batch.site_syms
        ks = batch.ks
        prop = batch.prop
        num_atoms = batch.asym_num_atoms
        num_crystals = num_atoms.size(0)

        batch_pos_loss = torch.tensor([], device=DEVICE)
        batch_lattice_loss = torch.tensor([], device=DEVICE)
        batch_sym_loss = torch.tensor([], device=DEVICE)


        for _ in range(self.val_repeats):
            t = torch.rand(
                    [num_crystals, 1], dtype=frac_coords.dtype, device=types.device
                )
            pos_loss,  lattice_loss, sym_loss, _ = self.bfn.loss_one_step(
                t, x=types, pos=frac_coords, ks=ks, num_atoms=num_atoms, spacegroup=spacegroup, site_syms=site_syms, segment_ids=segment_ids, prop=prop
            )
            batch_pos_loss = torch.cat((batch_pos_loss, pos_loss))
            batch_lattice_loss = torch.cat((batch_lattice_loss, lattice_loss))
            batch_sym_loss = torch.cat((batch_sym_loss, sym_loss))


        pos_loss = torch.mean(batch_pos_loss)
        lattice_loss = torch.mean(batch_lattice_loss)
        sym_loss = torch.mean(batch_sym_loss)
        loss = self.cost_coord * pos_loss  + self.cost_lattice * lattice_loss +  self.cost_sym * sym_loss

        log_dict = {
            'val_loss': loss,
            'val_lattice_loss': lattice_loss,
            'val_coord_loss': pos_loss,
            'val_sym_loss': sym_loss,
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        
        self.train_losses = []