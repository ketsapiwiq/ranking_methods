# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT
"""
Plot functions.
"""

import altair as alt
import polars as pl


def plot_scores_with_confidence(scores: pl.DataFrame) -> alt.LayerChart:
    """
    Plot models scores.

    Args:
        scores (pl.DataFrame): Scores DataFrame with columns "model", "median", "p2.5" and "p97.5".
    Returns:
        alt.Chart: Plot.
    """
    scores = scores.sort("median", descending=True)
    model_order = scores["model"].to_list()
    df = scores.to_pandas()

    error_bars = (
        alt.Chart(df)
        .mark_errorbar(extent="ci", color="lightblue", thickness=4)
        .encode(x=alt.X("model:N", sort=model_order, title="Model"), y=alt.Y("p2.5:Q", title="Score"), y2="p97.5:Q")
    )
    medians = (
        alt.Chart(df).mark_point(color="black", thickness=1).encode(x=alt.X("model:N", sort=model_order), y="median:Q")
    )

    # Combine
    return (error_bars + medians).properties(
        width=800,
        height=500,
    )


def format_matches_for_heatmap(matches: pl.DataFrame) -> pl.DataFrame:
    """
    From a DataFrame of matches with columns "score, "model_a" and "model_b"
    where score is 2 when B wins, 0 when A wins and 1 if draw,
    returned aggregated match data to plot in heatmaps..

    Args:
        matches (pl.DataFrame): Matches.
    Returns:
        pl.DataFrame: Heatmap.
    """
    df = matches.with_columns(
        a_wins=pl.when(pl.col("score") == 2).then(1).otherwise(0),
        b_wins=pl.when(pl.col("score") == 1).then(1).otherwise(0),
        draws=pl.when(pl.col("score") == 0).then(1).otherwise(0),
    ).select("model_a", "model_b", "a_wins", "b_wins", "draws")

    # count of wins by pair of model
    counts = df.group_by(["model_a", "model_b"]).agg(
        pl.col("a_wins").sum(), pl.col("b_wins").sum(), pl.col("draws").sum()
    )
    # reversed counts since we want to add match statistics of a vs. b and b vs. a
    reversed_counts = counts.select(
        model_a="model_b",
        model_b="model_a",
        a_wins="b_wins",
        b_wins="a_wins",
        draws="draws",
    )
    # aggregate counts
    return (
        (
            pl.concat([counts, reversed_counts])
            .group_by(["model_a", "model_b"])
            .agg(pl.sum("a_wins"), pl.sum("b_wins"), pl.sum("draws"))
        )
        .with_columns((pl.col("a_wins") + pl.col("b_wins") + pl.col("draws")).alias("count"))
        .with_columns((pl.col("a_wins") / (pl.col("a_wins") + pl.col("b_wins"))).round(2).alias("a_win_ratio"))
    )


def plot_match_counts(heatmap_data: pl.DataFrame) -> alt.LayerChart:
    """
    From aggregated data with columns "model_a", "model_b" and "count"
    plot a heatmap of match counts.

    Args:
        heatmap_data (pl.DataFrame): Matches data.
    Returns:
        alt.Chart: Heatmap.
    """
    # plot
    base = (
        alt.Chart(heatmap_data.to_pandas())
        .encode(
            x=alt.X("model_b:N", title="Modèle B", sort=alt.EncodingSortField(field="model_b", order="descending")),
            y=alt.Y("model_a:N", title="Modèle A", sort=alt.EncodingSortField(field="model_a")),
        )
        .properties(
            width=1000,
            height=1000,
        )
    )
    heatmap = base.mark_rect(opacity=0.7).encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="viridis"), legend=None)
    )
    text = base.mark_text(color="black", fontSize=7).encode(text=alt.Text("count:Q"))

    final_chart = heatmap + text
    # rotate x-axis labels 45 degrees
    return final_chart.configure_axisX(labelAngle=45)


def plot_winrate_heatmap(heatmap_data: pl.DataFrame) -> alt.LayerChart:
    """
    From aggregated data with columns "model_a", "model_b" and "a_win_ratio",
    plot a heatmap of winrates by confrontation.

    Args:
        heatmap_data (pl.DataFrame): Matches data.
    Returns:
        alt.Chart: Heatmap.
    """
    # plot
    base = (
        alt.Chart(heatmap_data.to_pandas()[["model_a", "model_b", "a_win_ratio"]].dropna())  # type: ignore
        .encode(
            x=alt.X(
                "model_b:N", title="Modèle B: Perdant", sort=alt.EncodingSortField(field="model_b", order="descending")
            ),
            y=alt.Y("model_a:N", title="Modèle A: Gagnant", sort=alt.EncodingSortField(field="model_a")),
        )
        .properties(
            width=1000,
            height=1000,
        )
    )
    heatmap = base.mark_rect(opacity=0.7).encode(
        color=alt.Color("a_win_ratio:Q", scale=alt.Scale(scheme="redblue"), legend=None)
    )
    text = base.mark_text(color="black", fontSize=7).encode(text=alt.Text("a_win_ratio:Q"))

    final_chart = heatmap + text
    # rotate x-axis labels 45 degrees
    return final_chart.configure_axisX(labelAngle=45)
