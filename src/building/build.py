import os

import pytorch_lightning as pl

import datasets
from lightning import baseline, dann, loggers, pretraining


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
        max_epochs=200 if pretrained_encoder_path is None else 20,
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


def build_pretraining(
    source, target, domain_tradeoff, dropout, percent_broken, record_embeddings, gpu, seed
):
    pl.trainer.seed_everything(seed)
    logger = loggers.MLTBLogger(
        _get_logdir(),
        loggers.pretraining_experiment_name(source, target),
        tensorboard_struct={"pb": percent_broken, "dt": domain_tradeoff},
    )
    checkpoint_callback = loggers.MinEpochModelCheckpoint(
        monitor="val/checkpoint_score", min_epochs_before_saving=1
    )
    trainer = build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=100,
        val_interval=1.0,
        gpu=gpu,
        seed=seed,
    )
    truncate_val = not record_embeddings
    data = _build_datamodule(percent_broken, source, target, truncate_val)
    model = pretraining.UnsupervisedPretraining(
        in_channels=14,
        seq_len=data.window_size,
        num_layers=6,
        kernel_size=3,
        base_filters=16,
        latent_dim=64,
        dropout=dropout,
        domain_tradeoff=domain_tradeoff,
        domain_disc_dim=16,
        num_disc_layers=2,
        lr=0.01,
        weight_decay=0,
        record_embeddings=record_embeddings,
    )
    add_hparams(model, data, seed)

    return trainer, data, model


def build_trainer(
    logger,
    checkpoint_callback,
    max_epochs,
    val_interval,
    gpu,
    seed,
    callbacks=None,
    check_sanity=True,
):
    pl.trainer.seed_everything(seed)
    trainer = pl.Trainer(
        num_sanity_val_steps=2 if check_sanity else 0,
        gpus=[gpu],
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=1.0,
        val_check_interval=val_interval,
        callbacks=callbacks,
    )

    return trainer


def _build_datamodule(percent_broken, source, target, truncate_val):
    if target is None:
        return datasets.PretrainingBaselineDataModule(
            fd_source=source,
            num_samples=25000,
            batch_size=512,
            min_distance=1,
            percent_broken=percent_broken,
            truncate_val=truncate_val,
        )
    else:
        return datasets.PretrainingAdaptionDataModule(
            fd_source=source,
            fd_target=target,
            num_samples=50000,
            batch_size=512,
            min_distance=1,
            percent_broken=percent_broken,
            truncate_target_val=truncate_val,
        )


def add_hparams(model, data, seed):
    model.add_data_hparams(data)
    model.hparams.update({"seed": seed})


def _get_logdir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, ".."))

    return log_dir
