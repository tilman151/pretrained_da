from unittest import mock


def mock_logging(lightning_module):
    trainer_patch = mock.patch.object(lightning_module, "trainer")
    trainer_patch.start()
    lightning_module._current_fx_name = "training_step"
