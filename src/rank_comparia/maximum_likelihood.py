# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Alternative maximum likelihood ranker."""

import math
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from rank_comparia.ranker import Match, Ranker


class MaximumLikelihoodRanker(Ranker):

    BASE = 10
    INIT_RATING = 1000

    def __init__(self, scale: int = 400, max_iter: int = 200):
        super().__init__(scale)
        self.max_iter = max_iter
        self.scores = {}

    @staticmethod
    def aggregate_matches(matches: Iterable[Match]) -> pl.DataFrame:
        # store match list in dataframe
        model_a = []
        model_b = []
        a_wins = []  # a wins
        b_wins = []
        draws = []
        for m in matches:
            model_a.append(m.model_a)
            model_b.append(m.model_b)
            a_wins.append(int(m.score.value / 2))
            b_wins.append(max(0, 1 - m.score.value))
            draws.append(1 - int(m.score.value / 2) - max(0, 1 - m.score.value))
        df = pl.DataFrame(
            {"model_a_name": model_a, "model_b_name": model_b, "a_wins": a_wins, "b_wins": b_wins, "draws": draws}
        )

        # filter out matches with identical a and b models
        df = df.filter(pl.col("model_a_name") != pl.col("model_b_name"))

        # aggregate match results
        counts = df.group_by(["model_a_name", "model_b_name"]).agg(
            pl.col("a_wins").sum(), pl.col("b_wins").sum(), pl.col("draws").sum()
        )
        reversed_counts = (
            df.group_by(["model_a_name", "model_b_name"])
            .agg(pl.col("a_wins").sum(), pl.col("b_wins").sum(), pl.col("draws").sum())
            .select(
                pl.col("model_b_name").alias("model_a_name"),
                pl.col("model_a_name").alias("model_b_name"),
                pl.col("b_wins").alias("a_wins"),
                pl.col("a_wins").alias("b_wins"),
                pl.col("draws"),
            )
        )
        all_counts = (
            pl.concat([counts, reversed_counts])
            .group_by(["model_a_name", "model_b_name"])
            .agg(pl.sum("a_wins"), pl.sum("b_wins"), pl.sum("draws"))
        )
        return all_counts

    def compute_scores(self, matches: Iterable[Match]) -> None:
        all_counts = self.aggregate_matches(matches=matches)
        # models list
        models = all_counts["model_a_name"].unique().to_list()
        models_df = pl.DataFrame({"idx": np.arange(len(models)), "model": models})

        # initialize matrices
        p = len(models)
        X = np.zeros([p * (p - 1) * 2, p])
        Y = np.zeros(p * (p - 1) * 2)
        cur_row = 0
        sample_weights = []

        for r1 in models_df.rows(named=True):
            model_a_idx = r1["idx"]
            model_a_name = r1["model"]
            for r2 in models_df.rows(named=True):
                model_b_idx = r2["idx"]
                model_b_name = r2["model"]

                # skip case where models are identical
                if model_a_idx == model_b_idx:
                    continue
                # if no row or no matches
                match_data = all_counts.filter(
                    pl.col("model_a_name") == model_a_name,
                    pl.col("model_b_name") == model_b_name,
                )
                if len(match_data) == 0:
                    continue
                X[cur_row, model_a_idx] = math.log(self.BASE)
                X[cur_row, model_b_idx] = -math.log(self.BASE)
                Y[cur_row] = 1.0
                sample_weights.append(match_data.row(0)[2] * 2 + match_data.row(0)[3])

                X[cur_row + 1, model_a_idx] = math.log(self.BASE)
                X[cur_row + 1, model_b_idx] = -math.log(self.BASE)
                Y[cur_row + 1] = 0.0
                sample_weights.append(match_data.row(0)[4] * 2 + match_data.row(0)[3])
                cur_row += 2

        X = X[:cur_row]
        Y = Y[:cur_row]

        lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6, max_iter=self.max_iter)  # type: ignore
        lr.fit(X, Y, sample_weight=sample_weights)
        scores = self.scale * lr.coef_[0] + self.INIT_RATING

        self.scores = {m: s for m, s in zip(models, scores)}
        return

    def get_scores(self) -> dict[str, float]:
        return {model: score for model, score in sorted(self.scores.items(), key=lambda x: -x[1])}
