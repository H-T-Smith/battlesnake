import typing
import copy
import math
import sys

class TreeNode:
    def __init__(self, state=None, eval=None, children=None, last_move=None):
        self.state = state
        self.eval = eval
        self.children = children
        self.last_move = last_move
    

# calculates basic distance from one point to another
def calculate_distance(pos1, pos2):
    return abs(pos1['x'] - pos2['x']) + abs(pos1['y'] - pos2['y'])

# calculates coordinates of all possible moves
def get_possible_moves(head):
    return {'up': {'x': head['x'], 'y': head['y'] + 1}, 'down': {'x': head['x'], 'y': head['y'] - 1},
            'left': {'x': head['x'] - 1, 'y': head['y']}, 'right': {'x': head['x'] + 1, 'y': head['y']}}

# checks for immediate collisions
def is_position_safe(next_pos, game_state):
    # Check for boundary conditions
    if not (0 <= next_pos['x'] < game_state['board']['width']) or not (0 <= next_pos['y'] < game_state['board']['height']):
        return False
    # Check for collisions with snakes (including self)
    for snake in game_state['board']['snakes']:
        if next_pos in snake['body'][:-1]:  # Exclude the tail since it will move
            return False
    return True

# returns a list moves that don't immediately result in collision
def determine_safe_moves(game_state):
    head = game_state['you']['head']
    safe_moves = []
    for move in get_possible_moves(head).items():
        if is_position_safe(move[1], game_state):
            safe_moves.append(move)
    return safe_moves

# determines the last move based on the neck position
def infer_last_move(my_body):
    if len(my_body) < 2:
        return None
    head, neck = my_body[0], my_body[1]
    if head['x'] > neck['x']:
        return 'right'
    elif head['x'] < neck['x']:
        return 'left'
    elif head['y'] > neck['y']:
        return 'down'
    else:
        return 'up'

# calculates the nearest food position
def find_nearest_food(my_head, food):
    if not food:
        return None
    return min(food, key=lambda x: calculate_distance(my_head, x))

# returns a move that goes toward the target
def move_towards_target(my_head, target, safe_moves):
    if not target:
        return None
    moves = get_possible_moves(my_head)
    return min(moves, key=calculate_distance(moves.get, target))

# TODO: use all snakes & obstacles instead of just self
# calculates the density of the snake's body around a given position within a specified radius
def density_score(game_state, radius=2):
    my_head = game_state['you']['head']
    body_segments_nearby = sum(1 for segment in game_state['you']['body']
                               if abs(segment['x'] - my_head['x']) <= radius and
                               abs(segment['y'] - my_head['y']) <= radius)
    area = (2 * radius + 1) ** 2
    return body_segments_nearby / area

# food score based on distance to foods
def food_score(game_state):
    food_positions = game_state['board']['food']
    if not food_positions: return 0
    sum = 0
    for food_position in food_positions:
        sum += math.sqrt(calculate_distance(game_state['you']['head'], food_position))
    return sum / len(food_positions)



# objective function considering nearby body density, closeness to food, and health
def objective_function(game_state): # minimize the output
    # arbitrary weights for now, can be trained
    return 12 * food_score(game_state) + 96 * density_score(game_state) + 1 * (100 - game_state['you']['health'])

# calculates next game state from given move + a bool if food was eaten
def calculate_next_game_state(game_state, move):
    next_game_state = copy.deepcopy(game_state)
    
    # find our snake
    for snake in next_game_state['board']['snakes']:
        if snake['id'] == next_game_state['you']['id']:
            # decrement health
            snake['health'] -= 1
            next_game_state['you']['health'] -= 1
            # remove food & refill health & add tail if necessary
            for food in next_game_state['board']['food']:
                # if food was eaten
                if food == move:
                    next_game_state['board']['food'].remove(food)
                    snake['health'] = 100
                    next_game_state['you']['health'] = 100
                    # add tail
                    snake['body'].append(snake['body'][-1])
                    snake['length'] += 1
                    next_game_state['you']['body'].append(snake['body'][-1])
                    next_game_state['you']['length'] += 1
            # update our snake position
            snake['head'] = move
            snake['body'].insert(0, move)
            snake['body'].pop()

    # update our snake position
    next_game_state['you']['head'] = move
    next_game_state['you']['body'].insert(0, move)
    next_game_state['you']['body'].pop()

    return next_game_state
    

# builds the decision tree to analyze with 
def build_tree(node: TreeNode, depth, max_depth, num_children):
    # determine "good" moves
    good_moves = determine_safe_moves(node.state)
    good_moves.sort(key=lambda x: objective_function(calculate_next_game_state(node.state, x[1])))
    good_moves = good_moves[:num_children] 
    # if no safe moves
    if not good_moves:
        node.eval = float('inf')
    if depth < max_depth or not good_moves:
        # find children
        node.children = [TreeNode(state=calculate_next_game_state(node.state, move[1]), eval = objective_function(calculate_next_game_state(node.state, move[1])), last_move=move[0]) for move in good_moves]
        if len(node.children) > num_children:
            node.children = node.children[:num_children]
        for child in node.children:
            # TODO: here is where we could implement minimax as we build the tree
            build_tree(child, depth + 1, max_depth, num_children)
    

def move(game_state: typing.Dict) -> typing.Dict:
    root = TreeNode(state = game_state)
    build_tree(root, 0, 3, 2)
    best_child = min(root.children, key=lambda x: x.eval)
    return {"move": best_child.last_move}


def info() -> typing.Dict:
    return {"apiversion": "1", "author": "Test", "color": "#888888", "head": "default", "tail": "default"}

def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")
    return {}

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    port = "8000"
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == '--port':
            port = sys.argv[i+1]
        elif sys.argv[i] == '--seed':
            random_seed = int(sys.argv[i+1])
    run_server({"info": info, "start": start, "move": move, "end": end, "port": port})