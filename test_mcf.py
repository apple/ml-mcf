import os
import shlex
import subprocess

import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar

from configs.base_config import BaseConfig
from builders.builders import build_dataloader


from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.modelrunner import modelrunner


torch.set_float32_matmul_precision("medium")


def build_tensorboard(summary_name):
    tbp = os.environ.get("TENSORBOARD_PORT")
    command = "tensorboard --logdir {} --port {} --bind_all".format(summary_name, tbp)
    print("tensorboard dir", summary_name)

    tensorboard_process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


def main(task_config):
    load_ckpt_path = task_config.resume_from_path
    mcf, task_config_checkpoint = modelrunner(
        load_ckpt_path
    )
    # Get task config from model
    task_config = OmegaConf.merge(task_config_checkpoint, task_config)
    mcf.online_sample = task_config.model_config.params["online_sample"]
    mcf.online_evaluation = task_config.model_config.params["online_evaluation"]
    mcf.sampling_fn = task_config.model_config.params.sampling_config.sampling_fn
    mcf.num_timesteps_ddim = task_config.model_config.params.sampling_config.num_timesteps_ddim

    # build data config
    data_module = build_dataloader(task_config.data_config)

    # build model
    task_config.model_config.params["data_type"] = task_config.data_config.data_type
    task_config.model_config.params.architecture_config.params.signal_num_channels = (
        task_config.model_config.params.input_signal_num_channels
    )
    task_config.model_config.params.architecture_config.params.proj_dim = (
        128  # We need to assign a random value here, this gets updated inside the model
    )
    task_config.model_config.params.architecture_config.params.coord_num_channels = (
        task_config.model_config.params.input_coord_num_channels
    )
    task_config.model_config.params["viz_dir"] = os.path.join("artifacts", "viz")
    ckpt_path = "artifacts"

    num_nodes = 1
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/mse_epoch",
        save_last=True,
        dirpath=ckpt_path,
        filename="field_ddpm-model-best",
        save_top_k=3,
        mode="min",
    )

    bar = TQDMProgressBar(refresh_rate=1)
    callback_list = [checkpoint_callback, bar]

    tb_logger = TensorBoardLogger("./logs/")
    build_tensorboard("./logs/")
    loggers = [tb_logger]

    trainer = pl.Trainer(
        devices="auto",
        num_nodes=num_nodes,
        callbacks=callback_list,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=task_config.eval_freq,
        logger=loggers,
        precision=task_config.precision,
        max_steps=0,
    )

    trainer.validate(
        mcf,
        dataloaders=[
            data_module.val_dataloader(),
        ],
    )


if __name__ == "__main__":
    task_config = BaseConfig().parse()
    main(task_config)
