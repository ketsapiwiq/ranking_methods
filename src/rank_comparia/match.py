# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum


class MatchScore(int, Enum):
    A = 2
    B = 0
    Draw = 1


@dataclass
class Match:
    model_a: str
    model_b: str
    score: MatchScore
