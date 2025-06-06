# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Literal

import datasets
import polars as pl

from rank_comparia.frugality import draw_chart


def save_chart(data: pl.DataFrame, title: str, log: bool, save_path: Path) -> None:
    """
    Save a specific altair chart as an html file.
    TODO: make this function generic.

    Args:
        data (pl.DataFrame): DataFrame. For now this has to be 'frugality' data.
        title (str): File name.
        log (bool): Whether or not to use a log scale.
        save_path (Path): Repository.
    """
    chart = draw_chart(data, title=title, log=log)
    file_friendly_title = "_".join(title.split())
    chart.save(fp=save_path / f"{file_friendly_title}.html", format="html")


def save_data(data: pl.DataFrame, title: str, save_path: Path) -> None:
    """
    Save polars DataFrame as a csv file.

    Args:
        data (pl.DataFrame): DataFrame with all infos calculated.
        title (str): File name.
        save_path (Path): Repository.
    """
    data.write_csv(file=save_path / f"{title}.csv", separator=";")


def load_comparia(
    repository: Literal[
        "ministere-culture/comparia-reactions",
        "ministere-culture/comparia-votes",
    ],
    **kwargs,
) -> pl.DataFrame:
    """
    Load `comparia-reactions` or `comparia-votes` as a polars DataFrame
    with a category field coming from `comparia-conversations`.
    Extra keyword arguments will be forwarded to `datasets.load_dataset`.

    Args:
        repository (Literal[
            "ministere-culture/comparia-reactions",
            "ministere-culture/comparia-votes",
        ]): HF repository name.

    Returns:
        pl.DataFrame: Dataset.
    """
    # environment variable HF_HOME must be set
    # and authentication to the hub is necessary
    data = datasets.load_dataset(
        repository,
        split="train",
        **kwargs,
    ).to_polars()  # type: ignore
    # add categories column
    conversations = (
        datasets.load_dataset("ministere-culture/comparia-conversations", split="train", **kwargs)
        .to_polars()  # type: ignore
        .select(["conversation_pair_id", "categories", "total_conv_a_kwh", "total_conv_b_kwh"])  # type: ignore
    )
    data = data.join(conversations, on="conversation_pair_id")  # type: ignore
    return data  # type: ignore


# List of categories in the `comparia-conversation` dataset
categories: list[str] = [
    "Education",
    "Arts",
    "Entertainment & Travel & Hobby",
    "Culture & Cultural geography",
    "Politics & Government",
    "Food & Drink & Cooking",
    "Law & Justice",
    "Natural Science & Formal Science & Technology",
    "Business & Economics & Finance",
    "Society & Social Issues & Human Rights",
    "Other",
    "Personal Development & Human Resources & Career",
    "Environment",
    "Health & Wellness & Medicine",
    "Shopping & Commodity",
    "Daily Life & Home & Lifestyle",
    "Religion & Spirituality",
    "Sports",
    "History",
    "Real Estate",
    "Philosophy",
    "International",
    "Psychology",
    "Security",
    "Philosophy & Spirituality",
    "Fashion",
    "Music",
    "Marketing",
    "Ethics & Debate",
    "Philosophy & logic",
    "Philosophy & Ethics",
    "Industry",
    "Robotics",
    "Travel",
    "Technology",
    "Travel & Hobby",
    "Philosophy and Ethics",
    "Theology",
    "Anthropology",
    "Philosophy & Religion",
    "Urban Planning",
    "Agriculture",
    "Linguistics",
    "Philosophy & Metaphysics",
    "Psychology & Mental Health",
    "Sociology",
    "Architecture and construction",
    "Industry and artisanat",
    "Biotechnology",
    "Marketing & Sales",
    "Mathematics",
    "Engineering",
    "Ethics",
]
