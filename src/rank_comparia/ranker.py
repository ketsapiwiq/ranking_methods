# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Base ranker class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class MatchScore(int, Enum):
    A = 2
    B = 0
    Draw = 1


@dataclass
class Match:
    model_a: str
    model_b: str
    score: MatchScore


class Ranker(ABC):

    def __init__(self, scale: int = 400):
        super().__init__()
        self.scale = scale

    @abstractmethod
    def compute_scores(self, matches: Iterable[Match]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_scores(self) -> dict[str, float]:
        raise NotImplementedError()
