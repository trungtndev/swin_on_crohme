import argparse
import os
import wandb
from pytorch_lightning.loggers import WandbLogger as Logger
from torch.distributed.pipeline.sync.checkpoint import enable_checkpointing

from swinArm.datamodule import CROHMEDatamodule
from swinArm.lit_swinPreArm import LitSwinPreARM
from sconf import Config
import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin


def train(config):
    pl.seed_everything(config.seed_everything, workers=True)

    model_module = LitSwinPreARM(
        d_model=config.model.d_model,
        # encoder
        requires_grad=config.model.requires_grad,
        drop_rate=config.model.drop_rate,
        proj_drop_rate=config.model.proj_drop_rate,
        attn_drop_rate=config.model.attn_drop_rate,
        drop_path_rate=config.model.drop_path_rate,

        # decoder
        nhead=config.model.nhead,
        num_decoder_layers=config.model.num_decoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        dc=config.model.dc,
        cross_coverage=config.model.cross_coverage,
        self_coverage=config.model.self_coverage,
        # beam search
        beam_size=config.model.beam_size,
        max_len=config.model.max_len,
        alpha=config.model.alpha,
        early_stopping=config.model.early_stopping,
        temperature=config.model.temperature,
        # training
        learning_rate=config.model.learning_rate,
        patience=config.model.patience,
    )
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
                    log_model=config.wandb.log_model,
                    config=dict(config),
                    )
    logger.watch(model_module,
                 log="all",
                 log_freq=100
                 )

    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    lasted_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoint",
        every_n_epochs=1,
        monitor=None,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="lightning_logs",
        save_top_k=config.trainer.callbacks[1].init_args.save_top_k,
        monitor=config.trainer.callbacks[1].init_args.monitor,
        mode=config.trainer.callbacks[1].init_args.mode,
        filename=config.trainer.callbacks[1].init_args.filename)

    trainer = pl.Trainer(
        gpus=config.trainer.gpus,
        accelerator=config.trainer.accelerator,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        max_epochs=config.trainer.max_epochs,
        deterministic=config.trainer.deterministic,

        plugins=DDPPlugin(find_unused_parameters=False),
        logger=logger,
        enable_model_summary=True,
        callbacks=[lr_callback, checkpoint_callback, lasted_checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    train(config)
