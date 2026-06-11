from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from calibration import (
    apply_logit_calibration_to_outputs,
    calibrated_probabilities,
    apply_temperature_to_outputs,
    fit_logit_calibration,
    fit_temperature_scaling,
    save_temperature_scaling_artifact,
    temperature_scaled_probabilities,
)


def test_fit_temperature_scaling_uses_unweighted_validation_ce() -> None:
    logits = np.asarray(
        [
            [6.0, 0.0],
            [0.0, 6.0],
            [6.0, 0.0],
            [0.0, 6.0],
            [6.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 1, 0, 1, 1, 1], dtype=np.int64)

    result = fit_temperature_scaling(logits, targets, device="cpu", max_iter=30)

    assert result.temperature > 0.0
    assert result.validation_nll_after <= result.validation_nll_before
    assert result.n_samples == 6
    assert result.to_dict()["loss"] == "unweighted_cross_entropy"


def test_temperature_scaled_outputs_update_probabilities_and_predictions() -> None:
    outputs = {
        "sample_index": np.asarray([0, 1], dtype=np.int64),
        "targets": np.asarray([0, 1], dtype=np.int64),
        "predictions": np.asarray([0, 0], dtype=np.int64),
        "probabilities": np.asarray([[0.99, 0.01], [0.6, 0.4]], dtype=np.float32),
        "logits": np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32),
    }

    updated = apply_temperature_to_outputs(outputs, temperature=2.0)
    expected_probabilities = temperature_scaled_probabilities(outputs["logits"], 2.0)

    np.testing.assert_allclose(updated["probabilities"], expected_probabilities, rtol=1e-6)
    np.testing.assert_array_equal(updated["predictions"], np.asarray([0, 1], dtype=np.int64))
    assert updated["probabilities"].dtype == np.float32
    assert updated["temperature"] == pytest.approx(2.0)


def test_logit_calibration_with_class_bias_keeps_valid_probabilities() -> None:
    logits = np.asarray(
        [
            [3.0, 0.0, -1.0],
            [0.0, 2.0, -0.5],
            [-0.5, 0.0, 3.0],
            [2.0, 0.5, -1.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 1, 2, 0], dtype=np.int64)

    result = fit_logit_calibration(
        logits,
        targets,
        device="cpu",
        max_iter=20,
        class_bias_calibration=True,
    )
    probabilities = calibrated_probabilities(logits, result.temperature, result.class_biases)
    updated = apply_logit_calibration_to_outputs(
        {"logits": logits, "predictions": np.zeros(4, dtype=np.int64)},
        result.temperature,
        result.class_biases,
    )

    assert result.to_dict()["method"] == "temperature_scaling_with_class_bias"
    assert result.validation_nll_after <= result.validation_nll_before
    assert len(result.class_biases) == 3
    assert sum(result.class_biases) == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4), rtol=1e-6)
    np.testing.assert_array_equal(updated["predictions"], np.argmax(probabilities, axis=1))


def test_temperature_scaling_artifact_is_written(tmp_path: Path) -> None:
    target = tmp_path / "temperature_scaling.yaml"

    save_temperature_scaling_artifact({"enabled": True, "temperature": 1.5}, target)

    assert target.read_text(encoding="utf-8").startswith("enabled: true")
