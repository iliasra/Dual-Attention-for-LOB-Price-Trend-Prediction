from __future__ import annotations

from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")

from configuration import load_config
from training import CheckpointCandidate, EpochResult, LobTrainer


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_complete_training_state_round_trip(artifact_dir: Path) -> None:
    tmp_path = artifact_dir
    config = load_config()
    config.training.device = "cpu"
    config.training.use_amp = False
    config.training.model_dir = str(tmp_path)
    trainer = LobTrainer(config.training)

    model = torch.nn.Linear(2, 3)
    optimizer = trainer._optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    trainer.top_checkpoint_candidates = [
        CheckpointCandidate(
            epoch=1,
            monitor_value=0.5,
            path=tmp_path / "checkpoints" / "epoch_0001.pth",
            validation_index=1,
            checkpoint_label="epoch_0001",
        )
    ]

    state_path = tmp_path / "training_state_latest.pth"
    trainer._save_training_state(
        state_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history=[EpochResult(train_loss=1.0, val_loss=0.5, epoch=1, validation_index=1)],
        next_epoch=2,
        next_batch_in_epoch=1,
        global_step=9,
        optimizer_step=9,
        validation_index=1,
        best_monitor_value=0.5,
        best_epoch=1,
        best_checkpoint_label="epoch_0001",
        validations_without_improvement=0,
        wandb_run_id="run-123",
    )

    restored_model = torch.nn.Linear(2, 3)
    restored_optimizer = trainer._optimizer(restored_model)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=3)
    state = trainer._load_training_state(
        state_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert state["next_epoch"] == 2
    assert state["global_step"] == 9
    assert state["optimizer_step"] == 9
    assert state["gradient_accumulation_steps"] == 1
    assert state["validation_index"] == 1
    assert state["wandb_run_id"] == "run-123"
    assert state["history"][0].val_loss == 0.5
    assert trainer.top_checkpoint_candidates[0].checkpoint_label == "epoch_0001"
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_training_state_rejects_incompatible_gradient_accumulation(artifact_dir: Path) -> None:
    tmp_path = artifact_dir
    config = load_config()
    config.training.device = "cpu"
    config.training.use_amp = False
    config.training.model_dir = str(tmp_path)
    trainer = LobTrainer(config.training)

    model = torch.nn.Linear(2, 3)
    optimizer = trainer._optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    state_path = tmp_path / "training_state_latest.pth"
    trainer._save_training_state(
        state_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history=[],
        next_epoch=1,
        next_batch_in_epoch=1,
        global_step=0,
        optimizer_step=0,
        validation_index=0,
        best_monitor_value=0.0,
        best_epoch=0,
        best_checkpoint_label="",
        validations_without_improvement=0,
        wandb_run_id=None,
    )

    config.training.gradient_accumulation_steps = 2
    incompatible_trainer = LobTrainer(config.training)
    restored_model = torch.nn.Linear(2, 3)
    restored_optimizer = incompatible_trainer._optimizer(restored_model)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=3)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        incompatible_trainer._load_training_state(
            state_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )

    legacy_state = torch.load(state_path, map_location="cpu", weights_only=False)
    legacy_state.pop("gradient_accumulation_steps")
    legacy_path = tmp_path / "legacy_training_state_latest.pth"
    torch.save(legacy_state, legacy_path)

    restored_model = torch.nn.Linear(2, 3)
    restored_optimizer = incompatible_trainer._optimizer(restored_model)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=3)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        incompatible_trainer._load_training_state(
            legacy_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )
