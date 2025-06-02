# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Base ranker class."""
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from random import choices

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
    id: str | None = None


class Ranker(ABC):
    def __init__(self, scale: int = 400, default_score: float = 1000.0, bootstrap_samples: int = 100):
        super().__init__()
        self.scale = scale
        self.default_score = default_score
        self.bootstrap_samples = bootstrap_samples

    @abstractmethod
    def compute_scores(self, matches: list[Match]) -> dict[str, float]:
        raise NotImplementedError()

    @abstractmethod
    def get_scores(self) -> dict[str, float]:
        raise NotImplementedError()

    def compute_bootstrap_scores(self, matches: list[Match]) -> pl.DataFrame:
        # TODO: proper logging
        print(f"Computing bootstrap scores from a sample of {len(matches)} matches.")
        rows = []
        all_keys = set()
        for _ in tqdm(range(self.bootstrap_samples), desc="Processing bootstrap samples"):
            scores = self.compute_scores(choices(matches, k=len(matches)))
            rows.append(scores)
            all_keys.update(scores.keys())

        # fill missing keys with default score
        for d in rows:
            for key in all_keys:
                d.setdefault(key, self.default_score)

        # aggregate
        aggregated = defaultdict(list)
        for d in rows:
            for key in all_keys:
                aggregated[key].append(d[key])

        # compute boostrap confidence intervals
        results = pl.DataFrame(aggregated)

        bootstrap_column_names = [f"column_{index}" for index in range(self.bootstrap_samples)]
        return (
            results.transpose(include_header=True)
            .select(model="column", value_array=pl.concat_list(*bootstrap_column_names))
            .with_columns(
                [
                    pl.col("value_array").list.median().alias("median"),
                    pl.col("value_array")
                    .list.eval(pl.element().quantile(0.025, interpolation="nearest"))
                    .list.first()
                    .alias("p2.5"),
                    pl.col("value_array")
                    .list.eval(pl.element().quantile(0.975, interpolation="nearest"))
                    .list.first()
                    .alias("p97.5"),
                ]
            )
            .drop("value_array")
            .sort("median", descending=True)
        )
