# SPDX-FileCopyrightText: 2025 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

from typing import Iterable, Literal


def reciprocal_function(score_difference: float):
    return 1 / (1 + 10 ** (-score_difference / 400))


class RankerELO:

    def __init__(self, K=40):
        self.players = {}
        self.played_parties = {}
        self.K = K

    def add_player(self, player_name: str, score: float = 1000):
        self.players |= {player_name: score}
        self.played_parties |= {player_name: 0}

    def add_players(self, player_names):
        for player_name in player_names:
            self.add_player(player_name)

    def get_all_rankings(self):
        for player, ranking in sorted(self.players.items(), key=lambda x: -x[1]):
            print(f"{player} : {ranking}")

    def player_ranking(self, name: str) -> float:
        return self.players[name]

    def player_number_of_matches(self, name: str) -> int:
        return self.played_parties[name]

    def add_match(self, player_a: str, player_b: str, score: Literal[0, 1, 2], K: float = 40):
        """score : 0 -> b wins, 2 -> a wins, 1 -> draw"""
        W = score / 2.0
        D = min(max(self.player_ranking(player_a) - self.player_ranking(player_b), -400), 400)
        pd = reciprocal_function(D)
        self.players[player_a] += K * (W - pd)
        self.players[player_b] += K * (pd - W)
        self.played_parties[player_a] += 1
        self.played_parties[player_b] += 1

    def _add_match(self, model_a_name: str, model_b_name: str, score: Literal[0, 1, 2]):
        min_number_of_parties = min(
            self.player_number_of_matches(model_a_name),
            self.player_number_of_matches(model_b_name),
        )
        max_elo = max(self.player_ranking(model_a_name), self.player_ranking(model_b_name))
        if max_elo > 2400:
            K = self.K / 4
        elif min_number_of_parties > 30:
            K = self.K / 2
        else:
            K = self.K
        self.add_match(model_a_name, model_b_name, score=score, K=K)

    def compute_ranks(self, matches_list: Iterable[tuple[str, str, float, float]]):
        for model_a_name, model_b_name, score_a, score_b in matches_list:
            final_score = score_b - score_a
            if final_score > 0:
                score = 0
            elif final_score < 0:
                score = 2
            else:
                score = 1
            self._add_match(model_a_name, model_b_name, score=score)
