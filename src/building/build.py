import pytorch_lightning as pl

import datasets
from lightning import baseline, dann, logger as loggers
from run_baseline import _get_logdir


def build_transfer(
    source,
    target,
    percent_broken,
    pretrained_encoder_path,
    record_embeddings,
    domain_tradeoff,
    logger,
    gpu,
    seed,
):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/source_regression_loss"
    )
    trainer = build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=200 if pretrained_encoder_path is None else 10,
        val_interval=1.0 if pretrained_encoder_path is None else 0.1,
        gpu=gpu,
        seed=seed,
    )

    data = datasets.DomainAdaptionDataModule(
        fd_source=source,
        fd_target=target,
        batch_size=512,
        percent_broken=percent_broken,
    )

    model = dann.DANN(
        in_channels=14,
        seq_len=data.window_size,
        num_layers=6,
        kernel_size=3,
        base_filters=16,
        latent_dim=64,
        dropout=0.1,
        domain_trade_off=domain_tradeoff,
        domain_disc_dim=16,
        num_disc_layers=2,
        optim_type="adam",
        lr=0.01,
        record_embeddings=record_embeddings,
    )
    if pretrained_encoder_path is not None:
        model.load_encoder(pretrained_encoder_path, load_disc=True)
    add_hparams(model, data, seed)

    return trainer, data, model


def build_baseline(source, fails, pretrained_encoder_path, gpu, seed):
    logger = loggers.MLTBLogger(_get_logdir(), loggers.baseline_experiment_name(source))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val/regression_loss")
    trainer = build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=100,
        val_interval=1.0,
        gpu=gpu,
        seed=seed,
    )
    data = datasets.BaselineDataModule(
        fd_source=source, batch_size=512, window_size=30, percent_fail_runs=fails
    )
    model = baseline.Baseline(
        in_channels=14,
        seq_len=30,
        num_layers=4,
        kernel_size=3,
        base_filters=16,
        latent_dim=128,
        optim_type="adam",
        lr=0.01,
        record_embeddings=False,
    )
    if pretrained_encoder_path is not None:
        model.load_encoder(pretrained_encoder_path)
    add_hparams(model, data, seed)

    return trainer, data, model


def build_trainer(logger, checkpoint_callback, max_epochs, val_interval, gpu, seed):
    pl.trainer.seed_everything(seed)
    trainer = pl.Trainer(
        gpus=[gpu],
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=1.0,
        val_check_interval=val_interval,
    )

    return trainer


def add_hparams(model, data, seed):
    model.add_data_hparams(data)
    model.hparams.update({"seed": seed})
