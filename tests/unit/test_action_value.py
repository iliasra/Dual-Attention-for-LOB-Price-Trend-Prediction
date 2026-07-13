from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from action_value import (
    action_value_metrics,
    action_value_policy_frontier,
    action_value_quantile_calibration,
    spearman_rank_correlation,
)
from action_value_training import ActionValueRegressionLoss, ActionValueTrainer, split_action_value_outputs
from configuration import load_config


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
    assert frontier[0]["mean_pnl_ticks"] == pytest.approx(2.0)
    assert frontier[1]["profitable_precision"] == pytest.approx(0.5)
    assert frontier[-1]["total_pnl_ticks"] == pytest.approx(2.0)
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
