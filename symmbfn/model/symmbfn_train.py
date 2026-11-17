# torch 1.9.0's tensorboard import does `from setuptools import distutils` and then
# accesses `distutils.version`, which setuptools>=60 no longer imports as a side effect.
# Force-load the submodule first so the import works with modern setuptools (kept >=64
# for the PEP 660 editable build). Must run before torch/pytorch_lightning are imported.
import setuptools._distutils.version  # noqa: F401

from pathlib import Path

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig
from omegaconf.errors import InterpolationToMissingValueError
import pytorch_lightning as pl
from pytorch_lightning import Callback
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from symmbfn.common.utils import PROJECT_ROOT

import warnings
warnings.filterwarnings("ignore")


def build_callbacks(cfg: DictConfig) -> list[Callback]:
    callbacks: list[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        try:
            hydra_dir = Path(HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir)
        except InterpolationToMissingValueError:
            hydra_dir = Path(HydraConfig.get().run.dir)
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=hydra_dir,
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    return callbacks


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):
    try:
        hydra_dir = Path(HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir)
    except InterpolationToMissingValueError:
        hydra_dir = Path(HydraConfig.get().run.dir)

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
   
    model =hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    
    callbacks: list[Callback] = build_callbacks(cfg=cfg)
    

    wandb_config = cfg.logging.wandb
    wandb_logger = WandbLogger(
        **wandb_config,
        tags=cfg.core.tags,
    )
    wandb_logger.watch(
        model,
        log=cfg.logging.wandb_watch.log,
        log_freq=cfg.logging.wandb_watch.log_freq,
    )

    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        callbacks=callbacks,
        **cfg.train.pl_trainer,
    )
    trainer.fit(model, datamodule=datamodule)
    
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
   main()
    
    