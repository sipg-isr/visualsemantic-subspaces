import os
import numpy as np

import hydra
from omegaconf import DictConfig

import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from model import Model
from dataset import *

from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks.learning_rate_monitor import LearningRateMonitor
from torchtnt.framework.callbacks.module_summary import ModuleSummary
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from torchtnt.framework.fit import fit
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torchtnt.utils.env import init_from_env

import logging

from unit import Trainer

@hydra.main(config_path="./configs", version_base=None)
def main(cfg: DictConfig) -> None:    
    torch.cuda.empty_cache()

    torch.manual_seed(cfg.general.seed)
    np.random.seed(cfg.general.seed)

    hydra_wd = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(f"Experiment directory: {hydra_wd}")

    tb_path = os.path.join(hydra_wd, "tensorboard")

    if cfg.general.distributed:
        device = init_from_env()
    else:
        device = torch.device(cfg.general.device)

    train_loader = get_loader(
        dataset=cfg.train.dataset,
        split="train",
        img_size=cfg.general.img_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.val.persistent_workers,
    )
    
    val_loader = get_loader(
        dataset=cfg.val.dataset,
        split="val",
        img_size=cfg.general.img_size,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        pin_memory=cfg.val.pin_memory,
        persistent_workers=cfg.val.persistent_workers,
    )
    
    module = Model(
        dim=cfg.model.dim,
        activation=torch.nn.functional.leaky_relu,
        alpha=cfg.train.loss_coefs.alpha,
        beta=cfg.train.loss_coefs.beta,
        device=device,
    ).to(device)

    optimizer = SGD(
        module.parameters(),
        lr=cfg.train.optimizer.lr,
        momentum=cfg.train.optimizer.momentum,
    )

    lr_scheduler = StepLR(
        optimizer,
        cfg.train.scheduler.step_size,
        gamma=cfg.train.scheduler.gamma,
    )

    tb_logger = TensorBoardLogger(tb_path)
    lr_monitor = LearningRateMonitor(tb_logger)
    progress_bar = TQDMProgressBar(refresh_rate=1)
    module_summary = ModuleSummary()

    callbacks: List[Callback] = [
        progress_bar,
        module_summary,
        lr_monitor,
    ]

    trainer = Trainer(
        module,
        optimizer,
        lr_scheduler,
        device,
        tb_logger,
        cfg,
    )
    
    fit(
        trainer,
        train_loader,
        val_loader,
        max_epochs=cfg.train.max_epochs,
        max_steps=cfg.train.max_steps,
        callbacks=callbacks,
    )

if __name__ == "__main__":
    main()