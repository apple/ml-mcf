import torch
import os
import shlex
import subprocess

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar

from configs.base_config import BaseConfig
from builders.builders import build_dataloader
from utils.utils import instantiate_from_config

from pytorch_lightning import seed_everything
from utils.modelrunner import modelrunner


torch.set_float32_matmul_precision("medium")


def build_tensorboard(summary_name):
    tbp = os.environ.get("TENSORBOARD_PORT")
    command = "tensorboard --logdir {} --port {} --bind_all".format(summary_name, tbp)
    print("tensorboard dir", summary_name)

    subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


def main(task_config):
    if task_config.resume_from_path != "none":
        load_ckpt_path = task_config.resume_from_path
        mcf, task_config = modelrunner(
            load_ckpt_path
        )

        if not os.path.exists(os.path.join("artifacts", "viz")):
            os.makedirs(os.path.join("artifacts", "viz"))
        mcf.viz_dir = os.path.join("artifacts", "viz")

        data_module = build_dataloader(task_config.data_config)
    else:
        data_module = build_dataloader(task_config.data_config)
        task_config.model_config.params["data_type"] = task_config.data_config.data_type

        task_config.model_config.params.architecture_config.params.signal_num_channels = (
            task_config.model_config.params.input_signal_num_channels
        )
        task_config.model_config.params.architecture_config.params.proj_dim = (
            task_config.model_config.params.architecture_config.params.pos_embed_config.params.output_num_channels
        )

        task_config.model_config.params.architecture_config.params.coord_num_channels = (
            task_config.model_config.params.input_coord_num_channels
        )
        task_config.model_config.params["viz_dir"] = os.path.join(
            "artifacts", "viz"
        )
        task_config.model_config.params.train_config.max_steps = task_config.max_steps

        mcf = instantiate_from_config(task_config.model_config)
        load_ckpt_path = None
        mcf.opt = task_config

    num_nodes = 1

    callback_list = []
    if load_ckpt_path == None:
        load_ckpt_path = None
    plugin_list = []

    checkpoint_callback = ModelCheckpoint(
        monitor="loss/mse_epoch",
        save_last=True,
        dirpath="artifacts",
        filename="model-best-iter{step:08d}-mse_loss{loss/mse_epoch:.5f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        mode="min",
        every_n_train_steps=task_config.eval_freq,
    )

    bar = TQDMProgressBar(refresh_rate=1)
    callback_list.extend([checkpoint_callback, bar])

    tb_logger = TensorBoardLogger("./logs/")
    build_tensorboard("./logs/")
    loggers = [tb_logger]

    dataset = task_config.data_config.dataset
    model = task_config.model_config.params.architecture_config.target
    model = model.split(".")[-1]
    task_name = "{}_{}".format(dataset, model)

    seed_everything(42, workers=True)
    trainer = pl.Trainer(
        devices="auto",
        num_nodes=num_nodes,
        callbacks=callback_list,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        val_check_interval=task_config.eval_freq,
        logger=loggers,
        plugins=plugin_list,
        precision=task_config.precision,
        max_steps=task_config.max_steps,
    )

    trainer.fit(
        mcf,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=[
            data_module.val_dataloader(),
        ],
        ckpt_path=load_ckpt_path,
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