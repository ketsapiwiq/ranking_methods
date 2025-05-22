# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import polars as pl

from rank_comparia.frugality import draw_chart


def save_chart(data: pl.DataFrame, title: str, log: bool, save_path: Path):
    chart = draw_chart(data, title=title, log=log)
    file_friendly_title = "_".join(title.split())
    chart.save(fp=save_path / f"{file_friendly_title}.html", format="html")


def save_data(data: pl.DataFrame, title: str, save_path: Path):
    data.write_csv(file=save_path / f"{title}.csv", separator=";")
