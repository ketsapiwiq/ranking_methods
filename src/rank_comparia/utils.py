# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Literal

import datasets
import polars as pl

from rank_comparia.frugality import draw_chart


def save_chart(data: pl.DataFrame, title: str, log: bool, save_path: Path):
    chart = draw_chart(data, title=title, log=log)
    file_friendly_title = "_".join(title.split())
    chart.save(fp=save_path / f"{file_friendly_title}.html", format="html")


def save_data(data: pl.DataFrame, title: str, save_path: Path):
    data.write_csv(file=save_path / f"{title}.csv", separator=";")


def load_comparia(
    repository: Literal[
        "ministere-culture/comparia-reactions",
        "ministere-culture/comparia-votes",
    ]
) -> pl.DataFrame:
    # environment variable HF_DATASETS_CACHE must be set
    # and authentication to the hub is necessary
    data = datasets.load_dataset(
        repository,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split="train",
    ).to_polars()  # type: ignore
    # add categories column
    conversations = (
        datasets.load_dataset(
            "ministere-culture/comparia-conversations", cache_dir=os.environ["HF_DATASETS_CACHE"], split="train"
        )
        .to_polars()  # type: ignore
        .select(["conversation_pair_id", "categories"])  # type: ignore
    )
    data = data.join(conversations, on="conversation_pair_id")  # type: ignore
    return data  # type: ignore


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
