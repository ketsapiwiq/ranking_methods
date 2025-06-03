# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Ranking pipeline."""

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from rank_comparia.elo import ELORanker
from rank_comparia.maximum_likelihood import MaximumLikelihoodRanker
from rank_comparia.ranker import Match, MatchScore, Ranker
from rank_comparia.utils import categories, load_comparia


@dataclass
class RankingPipeline:
    """
    Ranking pipeline class.
    """

    method: Literal["elo_ordered", "elo_random", "ml"]  # score computation method used
    include_votes: bool  # whether to include votes dataset in raw match data
    include_reactions: bool  # whether to include reactions dataset in raw match data
    bootstrap_samples: int  # number of bootstrap samples
    batch: bool  # whether or not to batch matches together before computing score
    export_graphs: bool  # whether or not to export graphs
    ranker: Ranker = field(init=False)  # ranker

    def __post_init__(self):
        if not (self.include_votes | self.include_reactions):
            raise ValueError("At least one of votes or reactions data must be used.")
        if self.method == "elo_random":
            self.ranker = ELORanker(bootstrap_samples=self.bootstrap_samples)
        elif self.method == "ml":
            self.ranker = MaximumLikelihoodRanker(bootstrap_samples=self.bootstrap_samples)
        else:
            raise NotImplementedError()
        # matches
        self.matches = self._process_data()

    def run(self) -> pl.DataFrame:
        """
        Run bootstrap score computation.

        Returns:
            pl.DataFrame: Bootstrap scores.
        """
        matches = self.match_list()
        return self.ranker.compute_bootstrap_scores(matches=matches)

    def run_category(self, category: str) -> pl.DataFrame:
        """
        Run bootstrap score computation for a specific conversation topic.

        Args:
            category (str): Category.

        Returns:
            pl.DataFrame: Bootstrap scores for the provided category.
        """
        # filter matches
        matches = self.match_list(category=category)
        return self.ranker.compute_bootstrap_scores(matches=matches)

    def run_all_categories(self, min_matches: int = 5000) -> dict[str, pl.DataFrame]:
        """
        Run bootstrap score computation for a all conversation topics with
        more than `min_matches` matches.

        Args:
            min_matches (int): Threshold on the number of matches to compute scores.

        Returns:
            dict[str, pl.DataFrame]: Bootstrap scores by category.
        """
        results = {}
        for category in categories:
            matches = self.match_list(category=category)
            if len(matches) < min_matches:
                print(f"Skipping {category} which has less than 1000 matches.")
                continue
            results[category] = self.ranker.compute_bootstrap_scores(matches=matches)
        return results

    def match_list(self, category: str | None = None) -> list[Match]:
        """
        Return all matches, or matches for the provided category, as a list of `Match` objects.

        Args:
            category (str | None): Optional category.

        Returns:
            list[Match]: List of matches.
        """
        if category is None:
            matches = self.matches
        else:
            if category not in categories:
                raise ValueError(f"Category {category} does not exist in data.")
            # filter
            matches = self.matches.filter(pl.col("categories").list.contains(category))
        # return list of Matches
        return [
            Match(d["model_a"], d["model_b"], MatchScore(d["score"]), d["conversation_pair_id"])
            for d in matches.select(["model_a", "model_b", "score", "conversation_pair_id"]).to_dicts()
        ]

    def _process_data(self) -> pl.DataFrame:
        """
        Process raw data.

        Returns:
            pl.DataFrame: Formatted data.
        """
        matches = []
        if self.include_votes:
            matches.append(self._process_votes_data())
        if self.include_reactions:
            matches.append(self._process_reactions_data())
        return pl.concat(matches, how="vertical")

    def _process_votes_data(self) -> pl.DataFrame:
        """
        Process raw votes data.

        Returns:
            pl.DataFrame: Formatted votes data.
        """
        data = load_comparia("ministere-culture/comparia-votes")
        # drop duplicates
        data = data.unique(subset="conversation_pair_id", keep="first")
        # remove if equal is None and chosen is None
        data = data.filter(~((pl.col("both_equal").is_null()) & (pl.col("chosen_model_name").is_null())))

        # get score
        data = data.with_columns(
            pl.when(pl.col("both_equal"))
            .then(MatchScore.Draw)
            .when(pl.col("chosen_model_name") == pl.col("model_a_name"))
            .then(MatchScore.A)
            .when(pl.col("chosen_model_name") == pl.col("model_b_name"))
            .then(MatchScore.B)
            .otherwise(pl.lit(None))
            .alias("score")
        )
        # remove null scores
        data = data.filter(pl.col("score").is_not_null())

        print(f"Final votes dataset contains {len(data)} conversations pairs.")
        return data.select(
            pl.col("model_a_name").alias("model_a"),
            pl.col("model_b_name").alias("model_b"),
            "score",
            "conversation_pair_id",
            "categories",
        )

    def _process_reactions_data(self) -> pl.DataFrame:
        """
        Process raw reactions data.

        Returns:
            pl.DataFrame: Formatted reactions data.
        """
        # load data
        data = load_comparia("ministere-culture/comparia-reactions")

        # aggregate data by conversation pair (~ session)
        data = data.group_by("conversation_pair_id").agg(
            [
                pl.first("model_a_name").alias("model_a"),
                pl.first("model_b_name").alias("model_b"),
                pl.col("model_pos").alias("positions"),
                pl.col("liked").alias("likes"),
                pl.col("msg_rank").alias("ranks"),
                pl.first("categories"),
            ]
        )

        # list columns
        pos = pl.col("positions")
        liked = pl.col("likes")
        ranks = pl.col("ranks")

        # safe expressions
        pos_0 = pos.list.get(0, null_on_oob=True)
        liked_0 = liked.list.get(0, null_on_oob=True)
        rank_0 = ranks.list.get(0, null_on_oob=True)
        pos_1 = pos.list.get(1, null_on_oob=True)
        liked_1 = liked.list.get(1, null_on_oob=True)
        rank_1 = ranks.list.get(1, null_on_oob=True)

        # match logic
        data = data.with_columns(
            pl.col("model_a"),
            pl.col("model_b"),
            # 1 reaction case
            pl.when(pos.list.len() == 1).then(
                pl.when(liked_0)
                .then(pl.when(pos_0 == "a").then(MatchScore.A).otherwise(MatchScore.B))
                .otherwise(pl.when(pos_0 == "a").then(MatchScore.B).otherwise(MatchScore.A))
            )
            # 2 reactions
            .when(pos.list.len() == 2).then(
                # 2 likes
                pl.when(liked_0 & liked_1)
                .then(
                    pl.when(pos_0 == pos_1)
                    .then(pl.when(pos_0 == "a").then(MatchScore.A).otherwise(MatchScore.B))
                    .when(rank_0 == rank_1)
                    .then(MatchScore.Draw)
                    .otherwise(
                        pl.when(rank_0 < rank_1)
                        .then(pl.when(pos_0 == "a").then(MatchScore.A).otherwise(MatchScore.B))
                        .otherwise(pl.when(pos_1 == "a").then(MatchScore.A).otherwise(MatchScore.B))
                    )
                )
                # 2 dislikes
                .when(~liked_0 & ~liked_1)
                .then(
                    pl.when(pos_0 == pos_1)
                    .then(pl.when(pos_0 == "a").then(MatchScore.B).otherwise(MatchScore.A))
                    .when(rank_0 == rank_1)
                    .then(MatchScore.Draw)
                    .otherwise(
                        pl.when(rank_0 < rank_1)
                        .then(pl.when(pos_0 == "a").then(MatchScore.B).otherwise(MatchScore.A))
                        .otherwise(pl.when(pos_1 == "a").then(MatchScore.B).otherwise(MatchScore.A))
                    )
                )
                # 1 like, 1 dislike
                .otherwise(
                    pl.when(pos_0 != pos_1)
                    .then(
                        pl.when(liked_0)
                        .then(pl.when(pos_0 == "a").then(MatchScore.A).otherwise(MatchScore.B))
                        .otherwise(pl.when(pos_1 == "a").then(MatchScore.A).otherwise(MatchScore.B))
                    )
                    .otherwise(
                        # same model, if like first then the model with reactions wins
                        # if dislike first it loses
                        pl.when(pl.when(liked_0).then(rank_0 < rank_1).otherwise(rank_1 < rank_0))
                        .then(pl.when(pos_0 == "a").then(MatchScore.A).otherwise(MatchScore.B))
                        .otherwise(pl.when(pos_0 == "a").then(MatchScore.B).otherwise(MatchScore.A))
                    )
                )
            )
            # for now 3 reactions is not considered
            .otherwise(pl.lit(None)).alias("score"),
        )

        # remove null scores
        print(f"Reactions data originally contains {len(data)} conversations pairs.")
        data = data.filter(pl.col("score").is_not_null())
        print(f"Final reactions dataset contains {len(data)} conversations pairs.")

        return data.select(["model_a", "model_b", "score", "conversation_pair_id", "categories"])
