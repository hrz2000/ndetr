import os
import hydra
from pathlib import Path
from omegaconf import OmegaConf

import pytorch_lightning as pl
# from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from util.logging import setup_logging, sync_wandb
from training.dataloader import get_dataloader
from training.PlanT.lit_module import LitHFLM

check_val_every_n_epoch=1
@hydra.main(config_path=f"../config", config_name="config", version_base=None)
def main(cfg):

    # print config
    print(OmegaConf.to_yaml(cfg))

    # setup debug mode
    overfit = 0.0
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
        cfg.expname = "debug"
        overfit = 5  # use only 5 fixed batches for debugging

    if cfg.overfit > 0:
        overfit = cfg.overfit

    # use data caching for ML-Cloud #TODO
    shared_dict = None
    if cfg.use_caching:
        from diskcache import Cache
        tmp_folder = str(os.environ.get('SCRATCH'))
        print("Tmp folder for dataset cache: ", tmp_folder)
        tmp_folder = tmp_folder + "/dataset_cache"
        # We use a local diskcache to cache the dataset on the faster SSD drives on our cluster.
        shared_dict = Cache(directory=tmp_folder ,size_limit=int(768 * 1024 ** 3))
        # shared_dict = Cache(size_limit=int(768 * 1024**3))

    # if we use mutliple GPUs and want wandb online it does need too much 
    # time on the MLCLoud and the training freezes or is too slow
    # log only local and sync afterwards with wandb sync [OPTIONS] [PATH]
    if cfg.gpus > 1:
        os.environ["WANDB_MODE"] = "offline"

    # setup logging
    pl.seed_everything(cfg.seed)
    setup_logging(cfg)

    # setup lightning logger
    # csvlogger = CSVLogger(cfg.model.training.ckpt_path, "CSVLogger")
    # wandblogger = WandbLogger(
    #     project=cfg.exp_folder_name,
    #     name=cfg.wandb_name,
    #     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    #     entity="seqdrive",
    # )
    Path(f"{cfg.model.training.ckpt_path}/TBLogger").mkdir(parents=True, exist_ok=True)
    TBlogger = TensorBoardLogger(cfg.model.training.ckpt_path, name="TBLogger")

    # resume training
    resume_path = cfg.resume_path
    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
        
    checkpoint_path = cfg.load_from
    if checkpoint_path!=None and os.path.exists(checkpoint_path):
        checkpoint_path = checkpoint_path
    else:
        checkpoint_path = None
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath=cfg.cp_dirpath,
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=2,
    )

    train_loader, val_loader = get_dataloader(cfg, shared_dict=shared_dict, clas='all')

    if checkpoint_path != None:
        model = LitHFLM.load_from_checkpoint(checkpoint_path, cfg=cfg)
        # save=
    else:
        import pdb;pdb.set_trace()
        model = LitHFLM(cfg=cfg)

    # wandblogger.watch(model)
    to_train = False
    paras = dict(
        num_sanity_val_steps=0, 
        callbacks=checkpoint_callback,
        logger=[TBlogger],
        log_every_n_steps=2,
        resume_from_checkpoint=resume_path,
        check_val_every_n_epoch=check_val_every_n_epoch,
        max_epochs=cfg.model.training.max_epochs,
        overfit_batches=overfit,
    )
    if cfg.gpus > 1:
        replace_sampler_ddp = not cfg.custom_sampler
        trainer = Trainer(
            gpus=cfg.gpus,
            strategy="ddp",
            replace_sampler_ddp=replace_sampler_ddp,
            **paras
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            **paras
        )
    if to_train:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.test(model, train_loader, checkpoint_path, verbose=True) # No `test_step()` method defined to run `Trainer.test`.

    # if cfg.gpus > 1:
    #     sync_wandb(cfg)
    #     # os.system('wandb sync ./wandb/offline*')


if __name__ == "__main__":
    main()
