from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch


# This defines the anthropometric prior that we use in the iterative refining stage.
# Consists of a set of bone edges (pairs of joint indices), and the mean and stddev of the ratios of their lengths.
@dataclass
class AnthropometricPrior:
    bone_edges: list[tuple[int, int]]
    mean_ratios: np.ndarray
    std_ratios: np.ndarray


def load_anthropometric_prior(json_path: str | Path) -> AnthropometricPrior:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return AnthropometricPrior(
        bone_edges=[tuple(x) for x in data["bone_edges"]],
        mean_ratios=np.asarray(data["mean_ratios"], dtype=np.float32),
        std_ratios=np.asarray(data["std_ratios"], dtype=np.float32),
    )


def default_body_prior() -> AnthropometricPrior:
    edges = [(1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20), (17, 19), (19, 21), (1, 16), (2, 17)]
    mean_ratios = np.ones(len(edges), dtype=np.float32)
    std_ratios = np.full(len(edges), 0.15, dtype=np.float32)
    return AnthropometricPrior(edges, mean_ratios, std_ratios)


def prior_to_tensors(prior: AnthropometricPrior, device=None):
    return (
        torch.tensor(prior.bone_edges, device=device),
        torch.tensor(prior.mean_ratios, dtype=torch.float32, device=device),
        torch.tensor(prior.std_ratios, dtype=torch.float32, device=device),
    )
