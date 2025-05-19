# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Ranking pipeline."""

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from rank_comparia.elo import ELORanker
from rank_comparia.maximum_likelihood import MaximumLikelihoodRanker
from rank_comparia.ranker import Match, Ranker


@dataclass
class RankingPipeline:

    method: Literal["elo_ordered", "elo_random", "ml"]
    include_votes: bool
    include_reactions: bool
    batch: bool
    export_graphs: bool
    ranker: Ranker = field(init=False)

    def __post_init__(self):
        if not (self.include_votes | self.include_reactions):
            raise ValueError("At least one of votes or reactions data must be used.")
        if self.method == "elo_random":
            self.ranker = ELORanker()
        elif self.method == "ml":
            self.ranker = MaximumLikelihoodRanker()
        else:
            raise NotImplementedError()

    def run(self) -> pl.DataFrame:
        matches = self.matches_from_data()
        return self.ranker.compute_bootstrap_scores(matches=matches)

    def matches_from_data(self) -> list[Match]:
        matches = []
        if self.include_votes:
            matches += self.matches_from_votes_data()
        if self.include_reactions:
            matches += self.matches_from_reactions_data()
        return matches

    def matches_from_votes_data(self) -> list[Match]:
        raise NotImplementedError()

    def matches_from_reactions_data(self) -> list[Match]:
        raise NotImplementedError()
