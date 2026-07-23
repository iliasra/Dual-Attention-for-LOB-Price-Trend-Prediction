from __future__ import annotations

from pathlib import Path
import io
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from action_value import (
    ACTION_VALUE_METRICS_SCHEMA_VERSION,
    ActionValueMetrics,
    action_value_metrics,
    action_value_policy_frontier,
    action_value_quantile_calibration,
    spearman_rank_correlation,
)
from action_value_training import (
    ActionValueEpochResult,
    ActionValueRegressionLoss,
    ActionValueTrainer,
    split_action_value_outputs,
)
from configuration import TrainingConfig, load_config


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_spearman_rank_correlation_handles_ties_and_perfect_order() -> None:
    assert spearman_rank_correlation([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0]) == pytest.approx(1.0)
    assert spearman_rank_correlation([1.0, 1.0], [0.0, 1.0]) == 0.0


def test_action_value_metrics_separate_ranking_from_realized_pnl() -> None:
    predictions = np.asarray(
        [
            [3.0, -1.0],
            [2.0, 0.0],
            [-1.0, 4.0],
            [0.1, 0.2],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            [1.0, -2.0],
            [-3.0, 1.0],
            [-2.0, 2.0],
            [-0.5, -0.25],
        ],
        dtype=np.float32,
    )

    metrics = action_value_metrics(
        predictions,
        targets,
        decision_threshold_ticks=0.0,
        fixed_rate=0.25,
    )

    assert metrics.n == 4
    assert metrics.decision_count == 4
    assert metrics.total_pnl_ticks == pytest.approx(-0.25)
    assert metrics.mean_pnl_ticks == pytest.approx(-0.0625)
    assert metrics.fixed_rate_count == 2
    assert metrics.fixed_rate_actual_rate == pytest.approx(0.5)
    assert metrics.fixed_rate_total_pnl_ticks == pytest.approx(3.0)
    assert metrics.fixed_rate_win_rate == pytest.approx(1.0)


def test_economic_oracle_can_abstain_when_both_actions_lose() -> None:
    predictions = np.zeros((2, 2), dtype=np.float32)
    targets = np.asarray([[-2.0, -1.0], [-4.0, -3.0]], dtype=np.float32)

    metrics = action_value_metrics(predictions, targets)

    assert metrics.oracle_mean_pnl_ticks == pytest.approx(0.0)
    assert metrics.forced_oracle_mean_pnl_ticks == pytest.approx(-2.0)


def test_economic_oracle_averages_no_trade_zeros_over_all_rows() -> None:
    predictions = np.zeros((3, 2), dtype=np.float32)
    targets = np.asarray([[2.0, -1.0], [-2.0, 1.0], [-3.0, -4.0]], dtype=np.float32)

    metrics = action_value_metrics(predictions, targets)

    assert metrics.oracle_mean_pnl_ticks == pytest.approx(1.0)
    assert metrics.forced_oracle_mean_pnl_ticks == pytest.approx(0.0)


def test_empty_action_value_split_has_zero_economic_and_forced_oracles() -> None:
    empty = np.empty((0, 2), dtype=np.float32)

    metrics = action_value_metrics(empty, empty)

    assert metrics.oracle_mean_pnl_ticks == 0.0
    assert metrics.forced_oracle_mean_pnl_ticks == 0.0


def test_binary_average_precision_groups_ties_and_is_row_order_invariant() -> None:
    from action_value import binary_average_precision

    scores = np.ones(5, dtype=np.float64)
    positives = np.asarray([True, False, False, True, False])

    first = binary_average_precision(scores, positives)
    permuted = binary_average_precision(scores[::-1], positives[::-1])

    assert first == pytest.approx(2.0 / 5.0)
    assert permuted == pytest.approx(first)


def test_action_value_metrics_serialization_marks_legacy_oracle_unambiguously() -> None:
    metrics = action_value_metrics(
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([[2.0, -1.0]], dtype=np.float32),
    )
    checkpoint = io.BytesIO()
    torch.save({"metrics_schema_version": ACTION_VALUE_METRICS_SCHEMA_VERSION, "metrics": metrics}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert restored["metrics_schema_version"] == 2
    assert restored["metrics"].to_dict()["oracle_mean_pnl_ticks"] == pytest.approx(2.0)
    assert restored["metrics"].to_dict()["forced_oracle_mean_pnl_ticks"] == pytest.approx(2.0)

    # Frozen slot dataclasses are restored by applying this ordered state. An
    # old checkpoint contains one fewer value because the forced-oracle field
    # did not exist when it was written.
    legacy_state = [getattr(metrics, field_name) for field_name in metrics.__dataclass_fields__]
    legacy_state = legacy_state[:-1]
    legacy_state[-1] = -1.25
    legacy_metrics = ActionValueMetrics.__new__(ActionValueMetrics)
    legacy_metrics.__setstate__(legacy_state)
    legacy_payload = legacy_metrics.to_dict()

    assert legacy_payload["oracle_mean_pnl_ticks"] is None
    assert legacy_payload["forced_oracle_mean_pnl_ticks"] == pytest.approx(-1.25)


def test_schema_v2_yaml_exposes_both_oracle_definitions() -> None:
    metrics = action_value_metrics(
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([[2.0, -1.0], [-4.0, -3.0]], dtype=np.float32),
    )
    epoch = ActionValueEpochResult(
        epoch=1,
        train_loss=0.5,
        validation_loss=0.4,
        validation_metrics=metrics,
        monitor_value=0.0,
    )
    serialized = yaml.safe_load(
        yaml.safe_dump(
            {
                "metrics_schema_version": ACTION_VALUE_METRICS_SCHEMA_VERSION,
                "epochs": [epoch.to_dict()],
            }
        )
    )

    assert serialized["metrics_schema_version"] == 2
    yaml_metrics = serialized["epochs"][0]["validation_metrics"]
    assert yaml_metrics["oracle_mean_pnl_ticks"] == pytest.approx(1.0)
    assert yaml_metrics["forced_oracle_mean_pnl_ticks"] == pytest.approx(-0.5)


def test_action_value_fixed_rate_never_trades_same_row_twice() -> None:
    predictions = np.asarray(
        [
            [0.99, 0.98],
            [0.90, 0.20],
            [0.10, 0.95],
            [0.20, 0.85],
        ]
    )
    targets = np.ones_like(predictions)

    metrics = action_value_metrics(predictions, targets, fixed_rate=0.5)

    assert metrics.fixed_rate_overlap_resolved_count == 1
    assert metrics.fixed_rate_count == 4
    assert metrics.fixed_rate_actual_rate == pytest.approx(1.0)
    assert metrics.fixed_rate_total_pnl_ticks == pytest.approx(4.0)


def test_policy_frontier_joins_profitable_pr_ranking_and_pnl() -> None:
    predictions = np.asarray([[4.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    targets = np.asarray([[2.0, -1.0], [-2.0, 1.0], [-1.0, 3.0], [1.0, -1.0]])

    frontier = action_value_policy_frontier(predictions, targets, coverages=(0.25, 0.5, 1.0))

    assert [row["trade_count"] for row in frontier] == [1, 2, 4]
    assert frontier[0]["profitable_precision"] == pytest.approx(1.0)
    assert frontier[0]["profitable_recall"] == pytest.approx(0.25)
    assert frontier[0]["mean_pnl_ticks"] == pytest.approx(2.0)
    assert frontier[1]["profitable_precision"] == pytest.approx(0.5)
    assert frontier[-1]["total_pnl_ticks"] == pytest.approx(2.0)
    assert frontier[-1]["profitable_recall"] == pytest.approx(0.5)
    assert 0.0 <= frontier[0]["policy_ap"] <= 1.0


def test_quantile_calibration_reports_empirical_coverage_and_crossing() -> None:
    targets = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    predictions = np.stack(
        [
            targets - 1.0,
            targets,
            targets + 1.0,
        ],
        axis=1,
    )

    calibration = action_value_quantile_calibration(predictions, targets, (0.1, 0.5, 0.9))

    assert calibration["n"] == 4
    assert calibration["row_crossing_rate"] == pytest.approx(0.0)
    assert calibration["central_interval"]["empirical_coverage"] == pytest.approx(1.0)
    assert calibration["per_action"]["long"]["0.5"]["empirical_cdf"] == pytest.approx(1.0)


def test_huber_quantile_loss_uses_separate_heads_and_penalizes_crossing() -> None:
    config = load_config().training
    config.objective.type = "action_value_regression"
    config.objective.loss = "huber_quantile"
    config.objective.quantiles = (0.1, 0.5, 0.9)
    config.objective.quantile_loss_weight = 0.25
    config.objective.quantile_crossing_weight = 0.1
    targets = torch.tensor([[1.0, -1.0]])
    ordered = torch.tensor([[1.0, -1.0, 0.0, -2.0, 1.0, -1.0, 2.0, 0.0]])
    crossed = ordered.clone()
    crossed[:, 2:4], crossed[:, 6:8] = ordered[:, 6:8].clone(), ordered[:, 2:4].clone()

    central, quantiles = split_action_value_outputs(ordered, config)
    criterion = ActionValueRegressionLoss(config)

    assert central.shape == (1, 2)
    assert quantiles is not None and quantiles.shape == (1, 3, 2)
    assert criterion(ordered, targets) < criterion(crossed, targets)


def test_action_value_trainer_evaluates_two_output_model() -> None:
    class LastTwoFeatures(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.zeros(2))

        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            values = x[..., :2] + self.offset
            return values if tokenwise else values[:, -1]

    config = load_config().training
    config.device = "cpu"
    config.use_amp = False
    config.objective.type = "action_value_regression"
    config.objective.loss = "huber"
    config.objective.fixed_rate = 0.5
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    trainer = ActionValueTrainer(config)
    targets = torch.tensor([[1.0, -1.0], [-2.0, 2.0]])
    features = torch.tensor([[[0.0, 0.0], [1.0, -1.0]], [[0.0, 0.0], [-2.0, 2.0]]])
    loader = [(features, torch.zeros((2, 2)), targets)]

    loss, metrics = trainer.evaluate(LastTwoFeatures(), loader, description="test")

    assert loss == pytest.approx(0.0)
    assert metrics.n == 2
    assert metrics.total_pnl_ticks == pytest.approx(3.0)


def test_action_value_trainer_rejects_scalar_classification_targets_before_reshape() -> None:
    class TokenwiseTwoOutputModel(nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            values = x[..., :2]
            return values if tokenwise else values[:, -1]

    config = load_config().training
    config.device = "cpu"
    config.use_amp = False
    trainer = ActionValueTrainer(config)
    features = torch.ones((2, 4, 2))
    scalar_targets = torch.ones((2, 4), dtype=torch.long)
    loss_mask = torch.tensor([[False, False, True, True], [False, False, True, True]])

    with pytest.raises(ValueError, match=r"\[batch, sequence, 2\].*stale classification targets"):
        trainer._supervised_values(
            TokenwiseTwoOutputModel(),
            features,
            torch.zeros((2, 4)),
            scalar_targets,
            loss_mask,
        )


def test_action_value_fp16_scaler_overflow_skips_step_instead_of_aborting() -> None:
    class OverflowGradientModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))

        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            values = x[..., :2] * self.weight
            return values if tokenwise else values[:, -1]

    class FakeOverflowScaler:
        def __init__(self) -> None:
            self.scale_value = 1024.0
            self.step_calls = 0

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
            return None

        def is_enabled(self) -> bool:
            return True

        def get_scale(self) -> float:
            return self.scale_value

        def step(self, _optimizer: torch.optim.Optimizer) -> None:
            self.step_calls += 1

        def update(self) -> None:
            self.scale_value /= 2.0

    config_path = Path(__file__).resolve().parents[2] / "configs" / "pipeline_config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))["training"]
    config = TrainingConfig.from_dict(payload)
    config.device = "cpu"
    config.use_amp = False
    config.gradient_accumulation_steps = 1
    trainer = ActionValueTrainer(config)
    fake_scaler = FakeOverflowScaler()
    trainer.scaler = fake_scaler  # type: ignore[assignment]
    model = OverflowGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    features = torch.ones((2, 4, 2))
    targets = torch.ones((2, 2))
    loader = [(features, torch.zeros((2, 4)), targets)]

    loss = trainer._run_train_epoch(
        model,
        loader,
        ActionValueRegressionLoss(config),
        optimizer,
        epoch=1,
    )

    assert np.isfinite(loss)
    assert fake_scaler.step_calls == 1
    assert fake_scaler.scale_value == 512.0


def test_action_value_batch_validation_reports_non_finite_features_and_tokens() -> None:
    with pytest.raises(FloatingPointError, match=r"invalid=\['features'\].*token_indices"):
        ActionValueTrainer._require_finite_batch(
            x=torch.tensor([[[1.0, float("nan")]]]),
            t=torch.zeros((1, 1)),
            targets=torch.zeros((1, 1, 2)),
            token_indices=torch.tensor([[123]]),
            epoch=1,
            batch_index=7,
        )


def test_action_value_trainer_collects_validation_quantile_outputs() -> None:
    class FixedQuantileModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            central = x[..., :2] + self.offset
            output = torch.cat([central, central - 1.0, central, central + 1.0], dim=-1)
            return output if tokenwise else output[:, -1]

    config = load_config().training
    config.device = "cpu"
    config.use_amp = False
    config.objective.type = "action_value_regression"
    config.objective.loss = "huber_quantile"
    config.objective.quantiles = (0.1, 0.5, 0.9)
    config.objective.fixed_rate = 0.5
    trainer = ActionValueTrainer(config)
    targets = torch.tensor([[1.0, -1.0], [-2.0, 2.0]])
    features = torch.tensor([[[0.0, 0.0], [1.0, -1.0]], [[0.0, 0.0], [-2.0, 2.0]]])

    loss, metrics = trainer.evaluate(
        FixedQuantileModel(),
        [(features, torch.zeros((2, 2)), targets)],
        description="quantile test",
    )

    assert loss > 0.0
    assert metrics.mae_long_ticks == pytest.approx(0.0)
    assert trainer.last_evaluation_outputs is not None
    assert trainer.last_evaluation_outputs["quantile_predictions"].shape == (2, 3, 2)
    np.testing.assert_allclose(trainer.last_evaluation_outputs["quantile_levels"], [0.1, 0.5, 0.9])


def test_action_value_training_state_resumes_at_next_epoch(artifact_dir: Path) -> None:
    tmp_path = artifact_dir
    class TinyActionValueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            values = self.head(x[..., :2])
            return values if tokenwise else values[:, -1]

    config = load_config().training
    config.device = "cpu"
    config.use_amp = False
    config.optimizer = "adamw"
    config.objective.type = "action_value_regression"
    config.objective.loss = "huber"
    config.objective.quantiles = ()
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.epochs = 1
    config.early_stopping_patience = 0
    config.validate_every_n_batches = "epoch"
    config.model_dir = str(tmp_path)
    features = torch.tensor([[[0.0, 0.0], [1.0, -1.0]], [[0.0, 0.0], [-2.0, 2.0]]])
    targets = torch.tensor([[1.5, -0.5], [-1.0, 1.0]])
    loader = [(features, torch.zeros((2, 2)), targets)]
    state_path = tmp_path / "training_state_latest.pth"

    first_trainer = ActionValueTrainer(config)
    _, first_history = first_trainer.fit(
        TinyActionValueModel(),
        loader,
        loader,
        training_state_path=state_path,
    )

    assert len(first_history) == 1
    assert state_path.exists()

    legacy_state = torch.load(state_path, map_location="cpu", weights_only=False)
    for key in (
        "next_batch_in_epoch",
        "global_step",
        "optimizer_step",
        "validation_index",
        "interval_loss_sum",
        "interval_rows",
        "wandb_run_id",
    ):
        legacy_state.pop(key, None)
    legacy_state["resume_signature"].pop("validate_every_n_batches", None)
    legacy_state["resume_signature"].pop("validate_at_epoch_end", None)
    torch.save(legacy_state, state_path)

    config.epochs = 2
    resumed_trainer = ActionValueTrainer(config)
    _, resumed_history = resumed_trainer.fit(
        TinyActionValueModel(),
        loader,
        loader,
        training_state_path=state_path,
        resume_checkpoint_path=state_path,
    )

    assert [result.epoch for result in resumed_history] == [1, 2]
    resumed_state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert resumed_state["next_epoch"] == 3
    assert resumed_trainer.selected_best_model_path is not None


def test_action_value_validates_and_checkpoints_at_batch_intervals(artifact_dir: Path) -> None:
    class TinyActionValueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            values = self.head(x[..., :2])
            return values if tokenwise else values[:, -1]

    config = load_config().training
    config.device = "cpu"
    config.use_amp = False
    config.optimizer = "adamw"
    config.objective.type = "action_value_regression"
    config.objective.loss = "huber"
    config.objective.quantiles = ()
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.epochs = 1
    config.gradient_accumulation_steps = 3
    config.validate_every_n_batches = 2
    config.validate_at_epoch_end = False
    config.early_stopping_patience = 0
    config.top_k_checkpoints = 2
    config.model_dir = str(artifact_dir)
    features = torch.tensor([[[0.0, 0.0], [1.0, -1.0]], [[0.0, 0.0], [-2.0, 2.0]]])
    targets = torch.tensor([[1.5, -0.5], [-1.0, 1.0]])
    batch = (features, torch.zeros((2, 2)), targets)
    train_loader = [batch for _ in range(5)]
    validation_loader = [batch]
    state_path = artifact_dir / "training_state_latest.pth"
    validation_steps: list[tuple[int, int]] = []
    training_steps: list[dict[str, object]] = []

    trainer = ActionValueTrainer(config)
    _, history = trainer.fit(
        TinyActionValueModel(),
        train_loader,
        validation_loader,
        training_state_path=state_path,
        validation_callback=lambda _result, global_step, optimizer_step: validation_steps.append(
            (global_step, optimizer_step)
        ),
        training_step_callback=training_steps.append,
    )

    assert validation_steps == [(2, 1), (4, 2)]
    assert [item.global_step for item in history] == [2, 4]
    assert [item.checkpoint_label for item in history] == [
        "epoch_0001_step_00000002",
        "epoch_0001_step_00000004",
    ]
    assert len(training_steps) == 5
    assert [step["optimizer_step_completed"] for step in training_steps] == [False, True, False, True, True]
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["next_epoch"] == 2
    assert state["next_batch_in_epoch"] == 1
    assert state["global_step"] == 5
    assert state["optimizer_step"] == 3
    assert state["validation_index"] == 2
    assert state["interval_rows"] == len(targets)
    assert 1 <= len(trainer.top_checkpoints) <= 2
    assert all(path.exists() for _value, _index, path in trainer.top_checkpoints)


def test_action_value_resume_skips_batches_before_last_validation(artifact_dir: Path) -> None:
    class TinyActionValueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            values = self.head(x[..., :2])
            return values if tokenwise else values[:, -1]

    config = load_config().training
    config.device = "cpu"
    config.use_amp = False
    config.optimizer = "adamw"
    config.objective.type = "action_value_regression"
    config.objective.loss = "huber"
    config.objective.quantiles = ()
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.epochs = 1
    config.gradient_accumulation_steps = 2
    config.validate_every_n_batches = 2
    config.validate_at_epoch_end = False
    config.early_stopping_patience = 0
    config.model_dir = str(artifact_dir)
    features = torch.tensor([[[0.0, 0.0], [1.0, -1.0]], [[0.0, 0.0], [-2.0, 2.0]]])
    targets = torch.tensor([[1.5, -0.5], [-1.0, 1.0]])
    batch = (features, torch.zeros((2, 2)), targets)
    loader = [batch for _ in range(5)]
    state_path = artifact_dir / "training_state_latest.pth"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        ActionValueTrainer(config).fit(
            TinyActionValueModel(),
            loader,
            [batch],
            training_state_path=state_path,
            wandb_run_id="wandb-action-run",
            validation_callback=lambda *_args: (_ for _ in ()).throw(RuntimeError("simulated interruption")),
        )

    interrupted = torch.load(state_path, map_location="cpu", weights_only=False)
    assert interrupted["next_epoch"] == 1
    assert interrupted["next_batch_in_epoch"] == 3
    assert interrupted["global_step"] == 2
    assert interrupted["wandb_run_id"] == "wandb-action-run"
    resumed_steps: list[dict[str, object]] = []
    _, history = ActionValueTrainer(config).fit(
        TinyActionValueModel(),
        loader,
        [batch],
        training_state_path=state_path,
        resume_checkpoint_path=state_path,
        wandb_run_id="wandb-action-run",
        training_step_callback=resumed_steps.append,
    )

    assert [step["batch_in_epoch"] for step in resumed_steps] == [3, 4, 5]
    assert [step["global_step"] for step in resumed_steps] == [3, 4, 5]
    assert [item.global_step for item in history] == [2, 4]


def test_action_value_interval_early_stopping_counts_validations(artifact_dir: Path) -> None:
    class ConstantActionValueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(2))

        def forward(self, x: torch.Tensor, t: torch.Tensor, *, tokenwise: bool = False) -> torch.Tensor:
            del t
            values = self.bias.expand(*x.shape[:-1], 2)
            return values if tokenwise else values[:, -1]

    config = load_config().training
    config.device = "cpu"
    config.use_amp = False
    config.optimizer = "adamw"
    config.learning_rate = 0.0
    config.objective.type = "action_value_regression"
    config.objective.loss = "huber"
    config.objective.quantiles = ()
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.epochs = 1
    config.gradient_accumulation_steps = 1
    config.validate_every_n_batches = 2
    config.validate_at_epoch_end = False
    config.early_stopping_patience = 1
    config.early_stopping_warmup = 0
    config.model_dir = str(artifact_dir)
    features = torch.ones((2, 2, 2))
    targets = torch.ones((2, 2))
    batch = (features, torch.zeros((2, 2)), targets)
    steps: list[dict[str, object]] = []

    _, history = ActionValueTrainer(config).fit(
        ConstantActionValueModel(),
        [batch for _ in range(8)],
        [batch],
        training_state_path=artifact_dir / "training_state_latest.pth",
        training_step_callback=steps.append,
    )

    assert len(history) == 2
    assert [item.global_step for item in history] == [2, 4]
    assert len(steps) == 4
