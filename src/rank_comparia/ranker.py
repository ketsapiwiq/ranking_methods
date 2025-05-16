# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Base ranker class."""
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from random import choices

import numpy as np
import polars as pl
from tqdm import tqdm


class MatchScore(int, Enum):
    A = 2
    B = 0
    Draw = 1


@dataclass
class Match:
    model_a: str
    model_b: str
    score: MatchScore


class Ranker(ABC):
    def __init__(self, scale: int = 400, bootstrap_samples: int = 100):
        super().__init__()
        self.bootstrap_samples = bootstrap_samples
        self.scale = scale

    @abstractmethod
    def compute_scores(self, matches: list[Match]) -> dict[str, float]:
        raise NotImplementedError()

    @abstractmethod
    def get_scores(self) -> dict[str, float]:
        raise NotImplementedError()

    def compute_bootstrap_scores(self, matches: list[Match]) -> pl.DataFrame:
        rows = []
        for _ in tqdm(range(self.bootstrap_samples), desc="Processing bootstrap samples"):
            rows.append(self.compute_scores(choices(matches, k=len(matches))))

        aggregated = defaultdict(list)
        for d in rows:
            for key, value in d.items():
                aggregated[key].append(value)

        # compute boostrap confidence intervals
        results = pl.DataFrame(aggregated)

        bootstrap_column_names = [f"column_{index}" for index in range(self.bootstrap_samples)]
        return (
            results.transpose(include_header=True)
            .select(model="column", value_array=pl.concat_list(*bootstrap_column_names))
            .with_columns(
                [
                    pl.col("value_array")
                    .map_elements(lambda x: float(np.median(x)), return_dtype=pl.Float64)
                    .alias("median"),
                    pl.col("value_array")
                    .map_elements(lambda x: float(np.quantile(x, 0.025, method="nearest")), return_dtype=pl.Float64)
                    .alias("p2.5"),
                    pl.col("value_array")
                    .map_elements(lambda x: float(np.quantile(x, 0.975, method="nearest")), return_dtype=pl.Float64)
                    .alias("p97.5"),
                ]
            )
            .drop("value_array")
        )
