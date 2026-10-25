from my_player import ZobristQuoridorHasher
from board_quoridor import BoardQuoridor
from player_quoridor import PlayerQuoridor
from game_state_quoridor import GameStateQuoridor, Orientation, Wall
from seahorse.game.stateless_action import StatelessAction
from enum import Enum

# NOTE
# - peu ajouter différents mouvements par hash pour confuse les adversaires,
#   par exemple si on sait que l'aversaire utilise une table d'ouverture
#   ont peu faire un mouvement inconventionnel pour le forcer à process son prochain move pendant qu'on
#   prend une approche naive (pour les premiers moves)
# - pour une meme distance couverte, prioriser switch de row plus tôt

class Color(Enum):
    WHITE = 1
    BLACK = 2

class OpeningMove:
    def __init__(self, color: Color, hash: int, action: StatelessAction):
        self.color = color
        self.hash = hash
        self.action = action

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    # NOTE add diag

def moving(pos: (int, int), direction: Direction):
    match (direction):
        case Direction.UP:
            return (pos[0] - 1, pos[1])
        case Direction.DOWN:
            return (pos[0] + 1, pos[1])
        case Direction.LEFT:
            return (pos[0], pos[1] - 1)
        case Direction.RIGHT:
            return (pos[0], pos[1] + 1)


def create_mock_game() -> tuple[
    PlayerQuoridor,
    PlayerQuoridor,
    GameStateQuoridor
]:
    # Create players
    player1 = PlayerQuoridor("W", goal_row=0, name="player1")
    player2 = PlayerQuoridor("B", goal_row=8, name="player2")

    players = [player1, player2]

    # Create the initial board
    board = BoardQuoridor(
        {
            player1.get_id(): (8, 4),
            player2.get_id(): (0, 4),
        },
        frozenset(),
        {
            player1.get_id(): 10,
            player2.get_id(): 10,
        },
    )

    # Create the initial game state
    game_state = GameStateQuoridor(
        {
            player1.get_id(): 0,
            player2.get_id(): 0,
        },
        player1,
        players=players,
        rep=board,
        step=0,
    )

    return player1, player2, game_state


player1, player2, game_state = create_mock_game()
zh = ZobristQuoridorHasher()

position_hash = zh.hash(game_state)

w_opening_table = {}
b_opening_table = {}

white_pos = (8,4)
black_pos = (0,4)

def add_opening_move(opening_table, game_state, player_position, action: Direction | Wall):
    position_hash = zh.hash(game_state)
    new_position = player_position

    new_action = None
    if isinstance(action, Wall):
        orientation = "horizontal" if action.orientation == Orientation.HORIZONTAL else "vertical"
        new_action = StatelessAction({"type": orientation, "destination": (action.row, action.col)})
    else:
        new_position = moving(player_position, action)
        new_action = StatelessAction({"type": "move", "destination": new_position})

    new_game_state = game_state.apply_action(new_action)

    # adding the action to the opening table
    # NOTE pass different opening table depending on the color
    # if position_hash in opening_table:
    #     opening_table[position_hash].append(new_action)
    # else:
    #     opening_table[position_hash] = [new_action]
        
    opening_table[position_hash] = new_action

    return new_game_state, new_position  

# Common beginning
# white 1er move
new_game_state, white_pos = add_opening_move(w_opening_table, game_state, white_pos, Direction.UP)      # white: (8,4), wall: []
# black 1er move
new_game_state, black_pos = add_opening_move(b_opening_table, new_game_state, black_pos, Direction.DOWN)# black: (1, 4), wall: [] 
# white 2e move
new_game_state, white_pos = add_opening_move(w_opening_table, new_game_state, white_pos, Direction.UP)  # white: (7,4), wall: []
# black 2e move
new_game_state, black_pos = add_opening_move(b_opening_table, new_game_state, black_pos, Direction.DOWN)# black: (2, 4), wall: []
# white 3e move
new_game_state, white_pos = add_opening_move(w_opening_table, new_game_state, white_pos, Direction.UP)  # white: (7,4), wall: []
# black 3e move
b3_new_game_state, black_pos = add_opening_move(b_opening_table, new_game_state, black_pos, Direction.DOWN)# black: (3, 4), wall: []

# Aoi (WHITE) VS sneaky_pete (BLACK) https://playquor.org/replay/6a5071773a50a56156f4aad8
new_white_pos = white_pos
new_black_pos = black_pos
w4_new_game_state, new_white_pos = add_opening_move(w_opening_table, b3_new_game_state, new_white_pos, Wall(6, 4, Orientation.HORIZONTAL))
b4_new_game_state = w4_new_game_state.apply_action(StatelessAction({"type": "horizontal", "destination": (3, 3)}))

w5_new_game_state, new_white_pos = add_opening_move(w_opening_table, b3_new_game_state, new_white_pos, Wall(3, 6, Orientation.VERTICAL))

# Aoi (BLACK) VS IronBarrier(WHITE) https://playquor.org/replay/6a4f62bf3a50a56156f4a83f
new_white_pos = white_pos
new_black_pos = black_pos
w4_new_game_state = b3_new_game_state.apply_action(StatelessAction({"type": "horizontal", "destination": (6, 5)}))
b4_new_game_state, new_black_pos = add_opening_move(b_opening_table, w4_new_game_state, new_black_pos, Wall(4, 6, Orientation.VERTICAL))

# autre cas generaux
# white 4e move
game_state_last_conventional_move, white_pos_conventional_move = add_opening_move(w_opening_table, b3_new_game_state, white_pos, Wall(6, 4, Orientation.HORIZONTAL))  # white: (7,4), wall: [('h', (6,4))]
game_state_w_wall_right = b3_new_game_state.apply_action(StatelessAction({"type": "horizontal", "destination": (6, 3)}))

# black 4e move
_new_game_state, black_pos = add_opening_move(b_opening_table, game_state_last_conventional_move, black_pos, Wall(3, 3, Orientation.HORIZONTAL))# black: (3, 4), wall: [('h', (3,3))]
# wall left side
_new_game_state, black_pos = add_opening_move(b_opening_table, game_state_w_wall_right, black_pos, Wall(3, 3, Orientation.HORIZONTAL))# black: (3, 4), wall: [('h', (3,3))]

# printing the result as a dictionnary
print("opening_book: dict[int, StatelessAction] = {\n    # moves for white")
for hash, move in w_opening_table.items():
    action = str(move).removeprefix("StatelessAction: ")
    print("    " + str(hash)+":\t", "StatelessAction("+action +"),")

print("    # moves for black")
for hash, move in b_opening_table.items():
    action = str(move).removeprefix("StatelessAction: ")
    print("    " + str(hash)+":\t", "StatelessAction("+action +"),")

print("}")
