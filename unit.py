import os

from omegaconf import DictConfig
from typing import Union
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
    def __init__(self,
                 module: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,
                 device: Union[str, torch.device],
                 tb_logger: TensorBoardLogger,
                 cfg: DictConfig) -> None:
        super().__init__()

        self._module = module
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._device = device
        self._tb_logger = tb_logger
        self._cfg = cfg

        self._metrics = {"multiclass_accuracy": MulticlassAccuracy()}

    def train_step(self, state: State, data: Batch) -> None:
        self._module.train()
        self._optimizer.zero_grad()

        step_count = self.train_progress.num_steps_completed

        data = copy_data_to_device(data, device=self._device)

        output = self._module(data)

        output["loss"].backward()
        self._optimizer.step()

        if step_count % 5 == 0 and get_global_rank() == 0:
            minterm_preds, minterm_targets = self._module.evaluate(data)
            self._metrics["multiclass_accuracy"].update(minterm_preds, minterm_targets)
            self._tb_logger.log("loss/train", output["loss"].detach(), step_count)

    def on_train_epoch_end(self, state: State) -> None:
        step_count = self.train_progress.num_steps_completed

        self._lr_scheduler.step()

        with open(os.path.join(self._cfg.general.experiment_path, "state_dict.pt"), "wb") as f:
            torch.save(self._module.state_dict(), f)
        
        # Log inner products of basis vectors
        kernel2 = (self._module._minterm_vecs @ self._module._minterm_vecs.T) ** 2
        self._tb_logger.writer.add_image("Minterm kernel", kernel2, step_count, dataformats="HW")

        for metric_name, metric in self._metrics.items():
            self._tb_logger.log(f"{metric_name}/train", metric.compute(), step_count)
            metric.reset()

    @torch.no_grad()
    def eval_step(self, state: State, data: Batch) -> None:
        self._module.eval()
        data = copy_data_to_device(data, self._device)
        minterm_preds, minterm_targets = self._module.evaluate(data)
        self._metrics["multiclass_accuracy"].update(minterm_preds, minterm_targets)

    def on_eval_epoch_end(self, state: State) -> None:
        step_count = self.train_progress.num_steps_completed
    
        self._module.forget()

        for metric_name, metric in self._metrics.items():
            self._tb_logger.log(f"{metric_name}/val", metric.compute(), step_count)
            metric.reset()

    def predict_step(self, state: State, data: Batch) -> None:
        pass