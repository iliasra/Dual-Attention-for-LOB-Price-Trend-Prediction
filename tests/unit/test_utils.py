from __future__ import annotations

import random

import numpy as np
import torch

from utils import set_global_seed, torch_generator_from_seed


def test_set_global_seed_resets_python_numpy_and_torch_rngs() -> None:
    set_global_seed(123)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    set_global_seed(123)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_torch_generator_from_seed_is_reproducible() -> None:
    generator_a = torch_generator_from_seed(456)
    generator_b = torch_generator_from_seed(456)

    assert torch.equal(torch.rand(4, generator=generator_a), torch.rand(4, generator=generator_b))
