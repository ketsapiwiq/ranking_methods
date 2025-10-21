import os
import operator
import polars as pl

from rank_comparia.utils import load_comparia

POSITIVE_REACTIONS = [
    "useful",
    "creative",
    "complete",
    "clear_formatting",
]
NEGATIVE_REACTIONS = ["incorrect", "superficial", "instructions_not_followed"]


def compute_total_and_ratio(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        total_prefs=pl.fold(
            acc=pl.lit(0),
            function=operator.add,
            exprs=pl.col(*POSITIVE_REACTIONS, *NEGATIVE_REACTIONS),
        )
    )
    data = data.with_columns(
        positive_prefs_ratio=pl.fold(
            acc=pl.lit(0),
            function=operator.add,
            exprs=pl.col(*POSITIVE_REACTIONS),
        )
        / pl.col("total_prefs")
    )

    return data


def get_votes_preferences(data: pl.DataFrame | None = None) -> pl.DataFrame:
    if data is None:
        data = load_comparia("ministere-culture/comparia-votes", token=None)

    data = (
        pl.concat(
            [
                data.select(
                    model_name=pl.col("model_a_name"),
                    **{
                        **{reaction: f"conv_{reaction}_a" for reaction in POSITIVE_REACTIONS},
                        **{reaction: f"conv_{reaction}_a" for reaction in NEGATIVE_REACTIONS},
                    },
                ),
                data.select(
                    model_name=pl.col("model_b_name"),
                    **{
                        **{reaction: f"conv_{reaction}_b" for reaction in POSITIVE_REACTIONS},
                        **{reaction: f"conv_{reaction}_b" for reaction in NEGATIVE_REACTIONS},
                    },
                ),
            ]
        )
        .group_by("model_name")
        .sum()
        .sort(by="model_name")
        .drop_nulls()
    )

    return compute_total_and_ratio(data)


def get_reactions_preferences(data: pl.DataFrame | None = None) -> pl.DataFrame:
    if data is None:
        data = load_comparia("ministere-culture/comparia-reactions", token=None)

    data = (
        data.select(
            model_name=pl.col("refers_to_model"),
            **{
                **{reaction: pl.col("liked") & pl.col(reaction) for reaction in POSITIVE_REACTIONS},
                **{reaction: pl.col("disliked") & pl.col(reaction) for reaction in NEGATIVE_REACTIONS},
            },
        )
        .group_by("model_name")
        .sum()
        .sort(by="model_name")
        .drop_nulls()
    )

    return compute_total_and_ratio(data)


def get_preferences_data(votes_data: pl.DataFrame | None = None, reactions_data: pl.DataFrame | None = None):
    votes_preferences = get_votes_preferences(votes_data)
    reactions_preferences = get_reactions_preferences(reactions_data)

    return compute_total_and_ratio(
        pl.concat([votes_preferences, reactions_preferences]).group_by("model_name").sum().sort(by="model_name")
    )
