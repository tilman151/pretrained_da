import pytorch_lightning as pl

import datasets
from building.build_common import get_logdir, add_hparams, build_trainer
from lightning import autoencoder, baseline, dann, loggers, pretraining


def build_transfer(
    source,
    target,
    percent_broken,
    config,
    pretrained_encoder_path,
    record_embeddings,
    gpu,
    seed,
    version,
):
    logger = loggers.MLTBLogger(
        get_logdir(),
        loggers.transfer_experiment_name(source, target),
        tag=version,
        tensorboard_struct={"pb": percent_broken, "dt": config["domain_tradeoff"]},
    )
    checkpoint_callback = loggers.MinEpochModelCheckpoint(
        monitor="val/source_regression_loss",
        save_top_k=-1,
        min_epochs_before_saving=5,
    )
    trainer = build_trainer(
        logger,
        checkpoint_callback,
        max_epochs=200,
        val_interval=1.0,
        gpu=gpu,
        seed=seed,
        check_sanity=False,
    )

    data = datasets.DomainAdaptionDataModule(
        fd_source=source,
        fd_target=target,
        batch_size=config["batch_size"],
        percent_broken=percent_broken,
    )

    model = build_dann_from_config(
        config, data.window_size, pretrained_encoder_path, record_embeddings
    )
    add_hparams(model, data, seed)

    return trainer, data, model


def build_dann_from_config(config, seq_len, pretrained_encoder_path, record_embeddings):
    model = dann.DANN(
        in_channels=14,
        seq_len=seq_len,
        num_layers=config["num_layers"],
        kernel_size=3,
        base_filters=config["base_filters"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
        domain_trade_off=config["domain_tradeoff"],
        domain_disc_dim=config["latent_dim"],
        num_disc_layers=config["num_disc_layers"],
        optim_type="adam",
        lr=config["lr"],
        record_embeddings=record_embeddings,
    )
    if pretrained_encoder_path is not None:
        model.load_encoder(pretrained_encoder_path, load_disc=True)

    return model


def build_baseline(source, fails, config, pretrained_encoder_path, gpu, seed, version):
    logger = loggers.MLTBLogger(
        get_logdir(), loggers.baseline_experiment_name(source), tag=version
    )
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
        fd_source=source, batch_size=config["batch_size"], percent_fail_runs=fails
    )
    model = build_baseline_from_config(config, data.window_size, pretrained_encoder_path)
    add_hparams(model, data, seed)

    return trainer, data, model


def build_baseline_from_config(config, seq_len, pretrained_encoder_path):
    model = baseline.Baseline(
        in_channels=14,
        seq_len=seq_len,
        num_layers=config["num_layers"],
        kernel_size=3,
        base_filters=config["base_filters"],
        latent_dim=config["latent_dim"],
        optim_type="adam",
        lr=config["lr"],
        record_embeddings=False,
    )
    if pretrained_encoder_path is not None:
        model.load_encoder(pretrained_encoder_path)

    return model


def build_pretraining(
    source,
    target,
    percent_broken,
    arch_config,
    config,
    mode,
    record_embeddings,
    gpu,
    seed,
    version,
):
    logger = loggers.MLTBLogger(
        get_logdir(),
        loggers.pretraining_experiment_name(source, target),
        tag=version,
        tensorboard_struct={"pb": percent_broken, "dt": config["domain_tradeoff"]},
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
    data = _build_datamodule(
        source, target, percent_broken, config["batch_size"], truncate_val
    )
    use_adaption = target is not None
    if mode == "metric":
        model = build_pretraining_from_config(
            arch_config, config, data.window_size, record_embeddings, use_adaption
        )
    elif mode == "autoencoder":
        model = build_autoencoder_from_config(
            arch_config, config, data.window_size, record_embeddings, use_adaption
        )
    else:
        raise ValueError(f"Unrecognized pre-training mode {mode}.")

    add_hparams(model, data, seed)

    return trainer, data, model


def _build_datamodule(source, target, percent_broken, batch_size, truncate_val):
    if target is None:
        return datasets.PretrainingBaselineDataModule(
            fd_source=source,
            num_samples=25000,
            batch_size=batch_size,
            min_distance=1,
            percent_broken=percent_broken,
            truncate_val=truncate_val,
        )
    else:
        return datasets.PretrainingAdaptionDataModule(
            fd_source=source,
            fd_target=target,
            num_samples=50000,
            batch_size=batch_size,
            min_distance=1,
            percent_broken=percent_broken,
            truncate_target_val=truncate_val,
        )


def build_pretraining_from_config(
    arch_config, config, seq_len, record_embeddings, use_adaption
):
    model = pretraining.UnsupervisedPretraining(
        in_channels=14,
        seq_len=seq_len,
        num_layers=arch_config["num_layers"],
        kernel_size=3,
        base_filters=arch_config["base_filters"],
        latent_dim=arch_config["latent_dim"],
        dropout=config["dropout"],
        domain_tradeoff=config["domain_tradeoff"] if use_adaption else 0.0,
        domain_disc_dim=arch_config["latent_dim"],
        num_disc_layers=arch_config["num_disc_layers"],
        lr=config["lr"],
        weight_decay=0.0,
        record_embeddings=record_embeddings,
    )

    return model


def build_autoencoder_from_config(
    arch_config, config, seq_len, record_embeddings, use_adaption
):
    model = autoencoder.AutoencoderPretraining(
        in_channels=14,
        seq_len=seq_len,
        num_layers=arch_config["num_layers"],
        kernel_size=3,
        base_filters=arch_config["base_filters"],
        latent_dim=arch_config["latent_dim"],
        dropout=config["dropout"],
        domain_tradeoff=config["domain_tradeoff"] if use_adaption else 0.0,
        domain_disc_dim=arch_config["latent_dim"],
        num_disc_layers=arch_config["num_disc_layers"],
        lr=config["lr"],
        weight_decay=0.0,
        record_embeddings=record_embeddings,
    )

    return model
