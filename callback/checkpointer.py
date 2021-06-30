import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import json
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_info, rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _METRIC, STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache
from pytorch_lightning.trainer.states import TrainerFn

log = logging.getLogger(__name__)
warning_cache = WarningCache()


class ModelCheckpoint(Callback):

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        save_last: bool = False,
        save_top_k: int = 0,
        is_better: Optional[Callable] = None,
        save_weights_only: bool = False,
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        every_n_val_epochs: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.is_better = is_better
        self.save_weights_only = save_weights_only
        self.auto_insert_metric_name = auto_insert_metric_name
        self.verbose = verbose

        self._last_global_step_saved = -1
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""
        self.last_model_path = ""

        self.__init_ckpt_dir(dirpath, filename, save_top_k)
        self.__init_triggers(every_n_train_steps, every_n_val_epochs)
        self.__validate_init_configuration()

        self.logs = {
            "best_k_models": {},
            "logs": []
        }
        self.log_name = "train_log.json"

    def __init_ckpt_dir(
        self,
        dirpath: Optional[Union[str, Path]],
        filename: Optional[str],
        save_top_k: Optional[int],
    ) -> None:
        self._fs = get_filesystem(str(dirpath) if dirpath else "")

        if (
            save_top_k is not None and save_top_k > 0 and dirpath is not None and self._fs.isdir(dirpath)
            and len(self._fs.ls(dirpath)) > 0
        ):
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")

        if dirpath and self._fs.protocol == "file":
            dirpath = os.path.realpath(dirpath)

        self.dirpath = dirpath
        self.filename = filename

    def __init_triggers(
        self,
        every_n_train_steps: Optional[int],
        every_n_val_epochs: Optional[int]
    ) -> None:
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_val_epochs is set
        if every_n_train_steps is None and every_n_val_epochs is None:
            self._every_n_val_epochs = 1
            self._every_n_train_steps = 0
            log.debug("Both every_n_train_steps and every_n_val_epochs are not set. Setting every_n_val_epochs=1")
        else:
            self._every_n_val_epochs = every_n_val_epochs or 0
            self._every_n_train_steps = every_n_train_steps or 0

    def __validate_init_configuration(self) -> None:
        if not isinstance(self.save_top_k, int):
            raise MisconfigurationException(f"Invalid type for save_top_k={self.save_top_k}. Must be int")
        if self.save_top_k < -1:
            raise MisconfigurationException(f"Invalid value for save_top_k={self.save_top_k}. Must be >= -1")
        if self._every_n_train_steps < 0:
            raise MisconfigurationException(
                f"Invalid value for every_n_train_steps={self._every_n_train_steps}. Must be >= 0"
            )
        if self.is_better is None and self.save_top_k >= 1:
            raise MisconfigurationException(
                f"ModelCheckpoint(save_top_k={self.save_top_k}, is_better=None) is not a valid configuration."
            )
        if self.is_better is not None and self.save_top_k <= 0:
            raise MisconfigurationException(
                f"ModelCheckpoint(save_top_k={self.save_top_k}, is_better=Callable) is not a valid configuration."
            )
        if self._every_n_val_epochs < 0:
            raise MisconfigurationException(
                f"Invalid value for every_n_val_epochs={self._every_n_val_epochs}. Must be >= 0"
            )
        if self._every_n_train_steps > 0 and self._every_n_val_epochs > 0:
            raise MisconfigurationException(
                f"Invalid values for every_n_train_steps={self._every_n_train_steps}"
                " and every_n_val_epochs={self._every_n_val_epochs}."
                " Both cannot be enabled at the same time."
            )

    def on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        When pretrain routine starts we build the ckpt dir on the fly
        """
        self.__resolve_ckpt_dir(trainer)

    def __resolve_ckpt_dir(self, trainer: "pl.Trainer") -> None:
        """
        Determines model checkpoint save directory at runtime. References attributes from the
        trainer's logger to determine where to save checkpoints.
        The base path for saving weights is set in this priority:

        1.  Checkpoint callback's path (if passed in)
        2.  The default_root_dir from trainer if trainer has no logger
        3.  The weights_save_path from trainer, if user provides it
        4.  User provided weights_saved_path

        The base path gets extended with logger name and version (if these are available)
        and subfolder "checkpoints".
        """
        # Todo: required argument `pl_module` is not used
        if self.dirpath is not None:
            return  # short circuit

        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                # the user has changed weights_save_path, it overrides anything
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str) else f"version_{trainer.logger.version}"
            )
            ckpt_path = os.path.join(save_dir, str(trainer.logger.name), version, "checkpoints")
        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        ckpt_path = trainer.training_type_plugin.broadcast(ckpt_path)

        self.dirpath = ckpt_path

        if not trainer.fast_dev_run and trainer.is_global_zero:
            self._fs.makedirs(self.dirpath, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """ Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps` """
        if self.__should_skip_saving_checkpoint(trainer):
            return
        step = trainer.global_step
        skip_batch = self._every_n_train_steps < 1 or ((step + 1) % self._every_n_train_steps != 0)
        if skip_batch:
            return
        self.save_checkpoint(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """ Save a checkpoint at the end of the validation stage. """
        skip = (
            self.__should_skip_saving_checkpoint(trainer) or self._every_n_val_epochs < 1
            or (trainer.current_epoch + 1) % self._every_n_val_epochs != 0
        )
        if skip:
            return
        self.save_checkpoint(trainer)

        # write log
        epoch = trainer.current_epoch
        step = trainer.global_step
        metrics = deepcopy(trainer.logger_connector.callback_metrics)
        for k, v in metrics.items():
            metrics[k] = v.item()
        metrics.update(epoch=epoch, step=step)
        self.logs["logs"].append(metrics)
        self.logs["best_k_models"] = {}
        for path, metrics in self.best_k_models.items():
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = v.item()
            self.logs["best_k_models"][path] = metrics
        with open(os.path.join(self.dirpath, self.log_name), "w") as f:
            f.write(json.dumps(self.logs, sort_keys=True, indent=4, ensure_ascii=False))

    def __should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        return (
            trainer.fast_dev_run  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "dirpath": self.dirpath
        }

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        callback_state: Dict[str, Any]
    ) -> None:
        self.best_model_score = callback_state["best_model_score"]
        self.best_model_path = callback_state["best_model_path"]

    def save_checkpoint(self, trainer: "pl.Trainer", unused: Optional["pl.LightningModule"] = None) -> None:
        """
        Performs the main logic around saving a checkpoint. This method runs on all ranks.
        It is the responsibility of `trainer.save_checkpoint` to correctly handle the behaviour in distributed training,
        i.e., saving only on rank 0 for data parallel use cases.
        """
        if unused is not None:
            rank_zero_deprecation(
                "`ModelCheckpoint.save_checkpoint` signature has changed in v1.3. The `pl_module` parameter"
                " has been removed. Support for the old signature will be removed in v1.5"
            )

        epoch = trainer.current_epoch
        global_step = trainer.global_step

        # track epoch when ckpt was last checked
        self._last_global_step_saved = global_step

        # what can be monitored
        metrics = self._get_metrics(trainer, epoch=epoch, step=global_step)

        # Based on `__validate_init_configuration()` method,
        # `save_top_k` is either -1 or 0 when `is_better` is None and
        # `save_top_k` is >= 1 when `is_better` is NOT None.
        if self.is_better is None:
            if self.save_top_k == -1:
                filepath = self._get_metric_interpolated_filepath_name(metrics, trainer)
                self._save_model(trainer, filepath)
                self.best_model_path = filepath
                self.best_model_score = metrics
        else:
            self._save_top_k_checkpoint(trainer, metrics)

        # save last checkpoints
        self._save_last_checkpoint(trainer, metrics)

    def _get_metrics(self, trainer: "pl.Trainer", epoch: int, step: int) -> Dict[str, _METRIC]:
        """ An example of `metrics`:
        {
            "valid_loss": tensor(1.6992, device="cuda:0"),
            "valid_Accuracy": tensor(0.9825, device="cuda:0"),
            "valid_Recall": tensor(0.9899, device="cuda:0"),
            "train_loss": tensor(10.4905, device="cuda:0"),
            "epoch": 0,
            "step": 24
        }
        """
        metrics = deepcopy(trainer.logger_connector.callback_metrics)
        metrics.update(epoch=epoch, step=step)
        return metrics

    def _save_top_k_checkpoint(self, trainer: "pl.Trainer", metrics: Dict[str, _METRIC]) -> None:
        if self.check_monitor_top_k(trainer, metrics):
            self._update_best_and_save(trainer, metrics)
        elif self.verbose:
            epoch = metrics.get("epoch")
            step = metrics.get("step")
            rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {metrics} was not in top {self.save_top_k}")

    def check_monitor_top_k(self, trainer: "pl.Trainer", metrics: Dict[str, torch.Tensor]) -> bool:
        if len(self.best_k_models) < self.save_top_k:  # save if less than k models
            return True

        new = metrics
        old = self.best_k_models[self.kth_best_model_path]
        should_update_best_and_save = self.is_better(new, old)

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = trainer.training_type_plugin.reduce_boolean_decision(should_update_best_and_save)
        return should_update_best_and_save

    def _update_best_and_save(self, trainer: "pl.Trainer", metrics: Dict[str, _METRIC]) -> None:
        del_filepath = None
        if len(self.best_k_models) == self.save_top_k:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # FIXME: do not save nan, replace with +/- inf
        filepath = self._get_metric_interpolated_filepath_name(metrics, trainer, del_filepath)
        self.best_k_models[filepath] = metrics

        if len(self.best_k_models) == self.save_top_k:  # monitor dict has reached k elements
            # re-evaluate kth_best_model and its path
            for i, (k, v) in enumerate(self.best_k_models.items()):
                if i == 0:
                    self.kth_best_model_path = k
                    kth_model_score = v
                else:
                    if self.is_better(kth_model_score, v):
                        self.kth_best_model_path = k
                        kth_model_score = v

        # re-evaluate the best_model and its path
        for i, (k, v) in enumerate(self.best_k_models.items()):
            if i == 0:
                self.best_model_path = k
                self.best_model_score = v
            else:
                if self.is_better(v, self.best_model_score):
                    self.best_model_path = k
                    self.best_model_score = v

        if self.verbose:
            epoch = metrics.get("epoch")
            step = metrics.get("step")
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: reached {monitor_candidate}"
                f" (best {self.best_model_score}), saving model to '{filepath}' as top {self.save_top_k}"
            )
        self._save_model(trainer, filepath)

        if del_filepath is not None and filepath != del_filepath:
            self._del_model(del_filepath)

    def _save_model(self, trainer: "pl.Trainer", filepath: str) -> None:
        if trainer.training_type_plugin.rpc_enabled:
            # RPCPlugin manages saving all model states
            # TODO: the rpc plugin should wrap trainer.save_checkpoint
            # instead of us having to do it here manually
            trainer.training_type_plugin.rpc_save_model(trainer, self._do_save, filepath)
        else:
            # in debugging, track when we save checkpoints
            trainer.dev_debugger.track_checkpointing_history(filepath)

            # make paths
            if trainer.is_global_zero:
                self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

            # delegate the saving to the trainer
            trainer.save_checkpoint(filepath, self.save_weights_only)

    @rank_zero_only
    def _del_model(self, filepath: str) -> None:
        if self._fs.exists(filepath):
            self._fs.rm(filepath)
            log.debug(f"Removed checkpoint: {filepath}")

    def _save_last_checkpoint(self, trainer: "pl.Trainer", metrics: Dict[str, _METRIC]) -> None:
        if not self.save_last:
            return

        filepath = self._format_checkpoint_name(self.CHECKPOINT_NAME_LAST, metrics)
        filepath = os.path.join(self.dirpath, f"{filepath}{self.FILE_EXTENSION}")

        self._save_model(trainer, filepath)

        if self.last_model_path and self.last_model_path != filepath and trainer.is_global_zero:
            self._del_model(self.last_model_path)

        self.last_model_path = filepath

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        metrics: Dict[str, _METRIC],
        prefix: str = "",
        auto_insert_metric_name: bool = True
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            for group in groups:
                name = group[1:]

                if auto_insert_metric_name:
                    filename = filename.replace(group, name + "={" + name)

                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def format_checkpoint_name(self, metrics: Dict[str, _METRIC], ver: Optional[int] = None) -> str:
        filename = self._format_checkpoint_name(
            self.filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name
        )

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    def _get_metric_interpolated_filepath_name(
        self,
        metrics: Dict[str, _METRIC],
        trainer: "pl.Trainer",
        del_filepath: Optional[str] = None,
    ) -> str:
        filepath = self.format_checkpoint_name(metrics)

        version_cnt = self.STARTING_VERSION
        while self.file_exists(filepath, trainer) and filepath != del_filepath:
            filepath = self.format_checkpoint_name(metrics, ver=version_cnt)
            version_cnt += 1

        return filepath

    def to_yaml(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the `best_k_models` dict containing the checkpoint
        paths with the corresponding scores to a YAML file.
        """
        best_k = {}
        for path, metrics in self.best_k_models.items():
            best_k[path] = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    best_k[path][k] = v.item()
                else:
                    best_k[path][k] = v
        if filepath is None:
            filepath = os.path.join(self.dirpath, "best_k_models.yaml")
        with self._fs.open(filepath, "w") as f:
            yaml.dump(best_k, f)

    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> None:
        best_k = {}
        for path, metrics in self.best_k_models.items():
            best_k[path] = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    best_k[path][k] = v.item()
                else:
                    best_k[path][k] = v
        if filepath is None:
            filepath = os.path.join(self.dirpath, "best_k_models.json")
        with self._fs.open(filepath, "w") as f:
            f.write(json.dumps(best_k, sort_keys=True, indent=4, ensure_ascii=False))

    def file_exists(self, filepath: Union[str, Path], trainer: "pl.Trainer") -> bool:
        """
        Checks if a file exists on rank 0 and broadcasts the result to all other ranks, preventing
        the internal state to diverge between ranks.
        """
        exists = self._fs.exists(filepath)
        return trainer.training_type_plugin.broadcast(exists)
