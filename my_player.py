"""
    Team:
        Maxime Gazzé  - 2454017
        Samuel Lajoie - 2447739
"""
import heapq
import math
from typing import Optional
from itertools import count

from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from game_state_quoridor import GameStateQuoridor, Orientation, Wall
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.player.player import Player

class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name)


    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        actions = self.generate_possible_stateless_actions(current_state);

        if not actions:
            raise RuntimeError("No legal action available.")

        best_action = None
        best_value = float('inf')

        for action in actions:
            game_state = current_state.apply_action(action)
            value = self.minimax(game_state)

            if value < best_value:
                best_action = action
                best_value = value

        if best_action is None:
            raise RuntimeError('No best action was selected')

        return best_action


    def minimax(
        self,
        game_state: GameStateQuoridor,
        depth: int = 1,
        alpha: int = -math.inf,
        beta: int = math.inf,
        maximizing_player: bool = True
    ) -> float:
        if depth == 0:
            return self.score_game_state(game_state)

        actions = self.generate_possible_stateless_actions(game_state);

        if maximizing_player:
            value = -math.inf
            
            for action in actions:
                new_game_state = game_state.apply_action(action)
                value = max(value, self.minimax(new_game_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break

            return value
        else:
            value = math.inf

            for action in actions:
                new_game_state = game_state.apply_action(action)
                value = min(value, self.minimax(new_game_state, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break

            return value


    def score_game_state(self, game_state: GameStateQuoridor) -> float:
        player0_path_len = self.a_star_shortest_path(game_state, game_state.players[0])
        player1_path_len = self.a_star_shortest_path(game_state, game_state.players[1])

        if player0_path_len is None:
            raise RuntimeError('No valid path for player 0')

        if player1_path_len is None:
            raise RuntimeError('No valid path for player 1')

        value = player0_path_len - player1_path_len

        if self.piece_type == 'B':
            value *= -1



        return value

    def a_star_shortest_path(self, game_state: GameStateQuoridor, player: PlayerQuoridor) -> Optional[int]:
        start_pos = game_state.rep.pawn_positions[player.id]
        goal_row = player.get_goal_row()

        # The heuristic function is simply the distance between the position and the player's goal row.
        def heuristic(position) -> int:
            return abs(position[0] - goal_row)

        # Counter is used in case the first two elements are equal. We do not want to compare the position to pick an element.
        counter = count()

        # heap entries : (estimated_distance, actual_distance, counter, position)
        open_set = [(abs(start_pos[0] - goal_row), 0, next(counter), start_pos)]

        actual_distance = {start_pos: 0}

        while open_set:
            _, current_distance, _, position = heapq.heappop(open_set)

            if current_distance != actual_distance[position]:
                continue

            if position[0] == goal_row:
                return current_distance

            for neighbour_pos in game_state._reachable_neighbours(position):
                temp_distance = current_distance + 1

                if temp_distance < actual_distance.get(neighbour_pos, float("inf")):
                    actual_distance[neighbour_pos] = temp_distance

                    estimated_distance = temp_distance + abs(neighbour_pos[0] - goal_row)
                    heapq.heappush(open_set, (estimated_distance, temp_distance, next(counter), neighbour_pos))

        return None


    def generate_possible_stateless_actions(self, game_state: GameStateQuoridor) -> [Action]:
        """
            Adding some improvements on the existing generate_possible_stateless_actions method of GameStateQuoridor:

            - removing the external wall as they serves no purpose:
        """
        legal_moves = game_state._legal_moves()

        actions = [*legal_moves]
        dimensions = range(game_state.rep.dimension)
        for row in dimensions:
            for col in dimensions:

                # Removing unusable top walls
                h_wall = Wall(row, col, Orientation.HORIZONTAL)
                if row != dimensions[0] and game_state._is_wall_legal(h_wall):
                    actions.append(
                        StatelessAction({"type": "horizontal", "destination": (h_wall.row, h_wall.col)}))

                # Removing unusable left walls
                v_wall = Wall(row, col, Orientation.VERTICAL)
                if col != dimensions[0] and game_state._is_wall_legal(v_wall):
                    actions.append(
                        StatelessAction({"type": "vertical", "destination": (v_wall.row, v_wall.col)}))
                    
        return actions
