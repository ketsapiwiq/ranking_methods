# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

import altair as alt
import polars as pl


def calculate_frugality_score(conversations: pl.DataFrame, n_match: pl.DataFrame, mean: bool) -> pl.DataFrame:
    """
    Calculate a frugality score by model from conversations data.

    Args:
        conversations (pl.DataFrame): Conversations data.
        n_match (pl.DataFrame): Data with the number of matches by model.
        mean (bool): If True, compute mean energy consumption per conversation.

    Returns:
        pl.DataFrame: DataFrame with frugality scores.
    """
    frugal_score = (
        pl.concat(
            [
                conversations.select(model_name=pl.col("model_a_name"), conso_all_conv=pl.col("total_conv_a_kwh")),
                conversations.select(model_name=pl.col("model_b_name"), conso_all_conv=pl.col("total_conv_b_kwh")),
            ]
        )
        .group_by("model_name")
        .sum()
        .sort(by="conso_all_conv", descending=True)
        .drop_nulls()
    )

    frugal_score = frugal_score.join(n_match, on="model_name")

    if mean:
        frugal_score = frugal_score.with_columns(pl.col("conso_all_conv").truediv("len").alias("mean_conso"))

    return frugal_score


def draw_chart(frugality_infos: pl.DataFrame, title: str, log: bool = False, mean: bool = False) -> alt.Chart:
    """
    Draw chart displaying Elo/BT scores against frugality scores.

    Args:
        frugality_infos (pl.DataFrame): DataFrame with frugality scores.
        title (str): Chart title.
        log (bool): Whether or not to use a log scale.
        mean (bool): Whether or not to display total consumption or consumption per conversation.

    Returns:
        alt.Chart: Chart displaying Elo/BT scores against frugality scores.
    """
    x_column = "mean_conso" if mean else "conso_all_conv"

    return (
        alt.Chart(frugality_infos, title=title)
        .mark_point()
        .encode(
            alt.Y("elo_score:Q").scale(zero=False),
            alt.X(f"{x_column}:Q").scale(type="log" if log else "linear"),
            color="organization",
            tooltip=["model_name", "organization", "license", "elo_score", "conso_all_conv"],
        )
        .properties(height=300, width=500)
    )
