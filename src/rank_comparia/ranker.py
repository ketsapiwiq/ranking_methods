# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Base ranker class."""
from abc import ABC, abstractmethod
from typing import Iterable

from rank_comparia.match import Match


class Ranker(ABC):

    def __init__(self, scale: int = 400):
        super().__init__()
        self.scale = scale

    @abstractmethod
    def compute_ranks(self, matches: Iterable[Match]):
        raise NotImplementedError()
