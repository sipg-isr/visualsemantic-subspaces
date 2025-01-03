import os

from omegaconf import DictConfig
from typing import Union, Dict
import logging

import torch 
import torch.nn as nn

from torchtnt.framework.state import State
from torchtnt.framework.unit import EvalUnit, PredictUnit, TrainUnit
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.device import copy_data_to_device
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torcheval.metrics import MulticlassAccuracy

from dataset import Batch


class Trainer(TrainUnit[Batch], EvalUnit[Batch], PredictUnit[Batch]):
    def __init__(
        self,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        device: Union[str, torch.device],
        tb_logger: TensorBoardLogger,
        cfg: DictConfig
    ) -> None:
        super().__init__()

        self._module = module
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._device = device
        self._tb_logger = tb_logger
        self._cfg = cfg

        self._metrics = {"multiclass_accuracy" : MulticlassAccuracy()}

    def tb_log_metrics(self, split: str) -> None:
        step_count = self.train_progress.num_steps_completed
        for metric_name, metric in self._metrics.items():
            name = f"{metric_name}/{split}"
            self._tb_logger.log(name, metric.compute(), step_count)
            metric.reset()

    def train_step(self, state: State, data: Batch) -> None:
        self._module.train()
        self._optimizer.zero_grad()

        data = copy_data_to_device(data, device=self._device)
        output = self._module(data)

        try:
            output["loss"].backward()
        except:
            logging.error("Error in backward pass")

        if self._cfg.train.optimizer.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self._module.parameters(),
                self._cfg.train.optimizer.clip_grad_norm
            )
        self._optimizer.step()

        step_count = self.train_progress.num_steps_completed

        if (step_count+1) % 50 == 0 and get_global_rank() == 0:
            self._tb_logger.log("loss/train", output["loss"].detach(), step_count)
            self._tb_logger.log("x norm/train", output["x_norm"].detach(), step_count)
            self._tb_logger.log("z norm/train", output["z_norm"].detach(), step_count)
        
            self._module.eval()
            self._module.update_minterms()
            minterm_preds, minterm_targets = self._module.evaluate(data)
            self._metrics["multiclass_accuracy"].update(minterm_preds, minterm_targets)

    def on_train_epoch_end(self, state: State) -> None:
        self._module.update_minterms()
        
        step_count = self.train_progress.num_steps_completed

        self._lr_scheduler.step()

        with open(os.path.join(self._cfg.general.experiment_path, "state_dict.pt"), "wb") as f:
            torch.save(self._module.state_dict(), f)
        
        # Log inner products of basis vectors
        x = torch.cat(self._module._minterm_samples, dim=0)
        x = x[torch.randperm(len(x))[:512].sort()[0]]
        K = self._module.kernel(x, x)
        self._tb_logger.writer.add_image("Minterm kernel", K ** 2, step_count, dataformats="HW")

        self.tb_log_metrics("train")

    @torch.no_grad()
    def eval_step(self, state: State, data: Batch) -> None:
        self._module.eval()
        data = copy_data_to_device(data, self._device)
        minterm_preds, minterm_targets = self._module.evaluate(data)
        self._metrics["multiclass_accuracy"].update(minterm_preds, minterm_targets)

    def on_eval_epoch_end(self, state: State) -> None:
        self.tb_log_metrics("val")
        self._module.forget()

    def predict_step(self, state: State, data: Batch) -> None:
        pass