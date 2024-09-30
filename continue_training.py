import argparse
import os
import wandb
from pytorch_lightning.loggers import WandbLogger as Logger
from swinArm.datamodule import CROHMEDatamodule
from swinArm.lit_swinPreArm import LitSwinPreARM
from sconf import Config
import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin


def train(config):
    pl.seed_everything(config.seed_everything, workers=True)

    model_module = LitSwinPreARM.load_from_checkpoint("checkpoint/epoch=51-val_loss=0.35873180627822876-step=86683-val_ExpRate=0.4427.ckpt")

    data_module = CROHMEDatamodule(
        zipfile_path=config.data.zipfile_path,
        dataset_name=config.data.dataset_name,
        test_year=config.data.test_year,
        train_batch_size=config.data.train_batch_size,
        eval_batch_size=config.data.eval_batch_size,
        num_workers=config.data.num_workers,
        scale_aug=config.data.scale_aug,
    )

    logger = Logger(name=config.wandb.name,
                    project=config.wandb.project,
                    id="8pl6jd7j?nw=nwusertrunglh113",
                    log_model=config.wandb.log_model,
                    config=dict(config),
                    )
    logger.watch(model_module,
                 log="all",
                 log_freq=100
                 )

    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="lightning_logs",
        save_top_k=config.trainer.callbacks[1].init_args.save_top_k,
        monitor=config.trainer.callbacks[1].init_args.monitor,
        mode=config.trainer.callbacks[1].init_args.mode,
        filename=config.trainer.callbacks[1].init_args.filename)

    trainer = pl.Trainer(
        resume_from_checkpoint="checkpoint/epoch=51-val_loss=0.35873180627822876-step=86683-val_ExpRate=0.4427.ckpt",

        devices=config.trainer.devices,
        tpu_cores=config.trainer.tpu_cores,
        gpus=config.trainer.gpus,
        accelerator=config.trainer.accelerator,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        max_epochs=config.trainer.max_epochs,
        deterministic=config.trainer.deterministic,

        plugins=DDPPlugin(find_unused_parameters=False),
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    # print(config)
    train(config)
