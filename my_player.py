"""
    Team:
        Maxime Gazzé  - 2454017
        Samuel Lajoie - 2447739
"""
import heapq
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
from itertools import count
import random

from actions_quoridor import MoveAction
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from game_state_quoridor import GameStateQuoridor, Orientation, Wall
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.player.player import Player

opening_book: dict[int, StatelessAction] = {
    # moves for white
    3385439458495528269:  StatelessAction({"type": "move", "destination": (7, 4)}),
    8457294491704433364:  StatelessAction({"type": "move", "destination": (6, 4)}),
    12314458424982501481: StatelessAction({"type": "move", "destination": (5, 4)}),
    12509703932543820695: StatelessAction({"type": "horizontal", "destination": (6, 4)}),

    # moves for black
    13231600766163051132: StatelessAction({"type": "move", "destination": (1, 4)}), # first move W moved forward -> B will move forward
    11548358507912285508: StatelessAction({"type": "move", "destination": (2, 4)}),
    7829530064862643382:  StatelessAction({"type": "move", "destination": (3, 4)}),
    4481792325934018976:  StatelessAction({"type": "horizontal", "destination": (3, 3)}),
}

# under which remaining threshold the time should switch
depth_remaining_time_threshold = {
    1: 4,      # 4s
    2: 3 * 60, # 3min
    # 3: 
}

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
        self._zobrist_hasher = ZobristQuoridorHasher()
        self._computated_states: dict[int, TTEntry] = {}

    
    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        position_hash = self._zobrist_hasher.hash(current_state)
        
        opening_book_action = opening_book.get(position_hash)
        if not opening_book_action is None:
            return opening_book_action


        actions = self.generate_possible_stateless_actions(current_state)

        if not actions:
            raise RuntimeError("No legal action available.")

        depth = 3
        if remaining_time < depth_remaining_time_threshold[2]:
            depth = 2
        elif remaining_time < depth_remaining_time_threshold[1]:
            depth = 1

        return self.iterative_deepened_negamax(current_state, remaining_time, depth)


    def iterative_deepened_negamax(
        self,
        game_state: GameStateQuoridor,
        remaining_time: float,
        depth: int = 2,
        color: int = 1,
        alpha: float = -math.inf,
        beta: float = math.inf,
    ) -> Optional[Action]:
        actions = self.generate_possible_stateless_actions(game_state)

        if not actions:
            return None

        best_action = actions[0]

        for current_depth in range(1, depth + 1):
            iteration_scores = []
            iteration_best_value = -math.inf
            iteration_best_action = None
            local_alpha = alpha

            for action in actions:
                start = time.time()
                new_game_state = game_state.apply_action(action)
                
                value, _ = self.negamax(new_game_state, depth=current_depth - 1, color=-color, alpha=-beta, beta=-local_alpha)
                value = -value

                iteration_scores.append((value, action))

                if value > iteration_best_value:
                    iteration_best_value = value
                    iteration_best_action = action

                end = time.time()
                elapsed_time = end - start
                remaining_time -= elapsed_time
                if remaining_time < depth_remaining_time_threshold[1]:
                   return best_action 
                
                local_alpha = max(local_alpha, iteration_best_value)
                if local_alpha >= beta:
                    break

            if iteration_best_action is not None:
                best_action = iteration_best_action

            searched_actions = [action for _, action in iteration_scores]
            unsearched_actions = list(filter(lambda action: action not in searched_actions, actions))

            # order the actions by their score and append all pruned actions
            actions = [action for _, action in sorted(iteration_scores, key=lambda x: x[0], reverse=True)] + unsearched_actions

        return best_action


    # inspired from https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning_and_transposition_tables
    def negamax(
        self,
        game_state: GameStateQuoridor,
        depth: int = 2,
        color: int = 1,
        alpha: float = -math.inf,
        beta: float = math.inf
    ) -> Tuple[float, Optional[Action]]:
        key = self._zobrist_hasher.hash(game_state)
        ttEntry: Optional[ttEntry] = self._computated_states.get(key)

        # If state is visited 
        if ttEntry is not None and ttEntry.depth >= depth:
            match (ttEntry.bound):
                case Bound.EXACT:
                    return ttEntry.value, ttEntry.action
                case Bound.LOWER:
                    alpha = max(alpha, ttEntry.value)
                case Bound.UPPER:
                    beta = min(beta, ttEntry.value)
            if alpha >= beta:
                return ttEntry.value, ttEntry.action

        def game_score():
            value = color * self.score_game_state(game_state)

            self._computated_states[key] = TTEntry(
                depth=depth,
                value=value,
                action=None,
                bound=Bound.EXACT,
            )

            return value, None

        # TODO return more explicit result when the game is actually done
        if depth == 0 or game_state.is_done(): 
            return game_score()

        actions = self.generate_possible_stateless_actions(game_state)
        if not actions:
            return game_score()

        original_alpha = alpha
        original_beta = beta
        best_value = -math.inf
        best_action = None

        for action in actions:
            new_game_state = game_state.apply_action(action)

            value, _ = self.negamax(new_game_state, depth - 1, -color, -beta, -alpha)
            value = -value

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

            if alpha >= beta:
                break

        bound = Bound.EXACT
        if best_value <= original_alpha:
            bound = Bound.UPPER
        elif best_value >= original_beta:
            bound = Bound.LOWER

        self._computated_states[key] = TTEntry(
            depth=depth,
            value=best_value,
            action=best_action,
            bound=bound,
        )

        return best_value, best_action

    def score_game_state(
        self,
        game_state: GameStateQuoridor,
    ) -> float:
        p1_shortest_path_len = self.a_star_shortest_path(game_state, game_state.players[0])
        p2_shortest_path_len = self.a_star_shortest_path(game_state, game_state.players[1])

        if p1_shortest_path_len is None:
            raise RuntimeError('No valid path for player 1')

        if p2_shortest_path_len is None:
            raise RuntimeError('No valid path for player 2')

        value = p1_shortest_path_len - p2_shortest_path_len

        if self.piece_type == 'W':
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
            # for neighbour_pos in self.reachable_neighbours(game_state, position):
                temp_distance = current_distance + 1

                if temp_distance < actual_distance.get(neighbour_pos, float("inf")):
                    actual_distance[neighbour_pos] = temp_distance

                    estimated_distance = temp_distance + abs(neighbour_pos[0] - goal_row)
                    heapq.heappush(open_set, (estimated_distance, temp_distance, next(counter), neighbour_pos))

        return None


    def generate_possible_stateless_actions(self, game_state: GameStateQuoridor) -> list[StatelessAction]:
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




class Bound(Enum):
    EXACT = auto()
    LOWER = auto()
    UPPER = auto()

@dataclass
class TTEntry:
    depth: int
    value: float
    action: Optional[Action]
    bound: Bound

class ZobristQuoridorHasher:
    def __init__(self, height: int = 9, width: int = 9) -> None:
        random.seed(123)

        self.height = height
        self.width = width

        self.player_tiles = [[self.generate_random_number() for _ in range(height * width)] for _ in range(2)]
        self.wall_counts = [[self.generate_random_number() for _ in range(11)] for _ in range(2)]
        self.horizontal_walls = [self.generate_random_number() for _ in range(height * (width - 1))]
        self.vertical_walls = [self.generate_random_number() for _ in range((height - 1) * width)]
        self.player_turn = self.generate_random_number()


    def generate_random_number(self) -> int:
        return random.getrandbits(64)


    def hash(self, game_state: GameStateQuoridor) -> int:
        h = 0

        player1_pawn_pos = game_state.rep.pawn_positions[game_state.players[0].id]
        player2_pawn_pos = game_state.rep.pawn_positions[game_state.players[1].id]

        h ^= self.player_tiles[0][player1_pawn_pos[0] * (self.width - 1) + player1_pawn_pos[1]]
        h ^= self.player_tiles[1][player2_pawn_pos[0] * (self.width - 1) + player2_pawn_pos[1]]

        h ^= self.wall_counts[0][game_state.rep.remaining_walls[game_state.players[0].id]]
        h ^= self.wall_counts[1][game_state.rep.remaining_walls[game_state.players[1].id]]

        for wall in game_state.rep.walls:
            if wall.orientation == Orientation.HORIZONTAL:
                h ^= self.horizontal_walls[wall.row * (self.width - 1) + wall.col]
            else:
                h ^= self.vertical_walls[wall.row * (self.width - 1) + wall.col]

        if game_state.active_player.id == 0:
            h ^= self.player_turn

        return h
