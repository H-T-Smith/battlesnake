import typing
import copy
import math
import sys
import json

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
        if next_pos in snake['body']:  # Exclude the tail since it will move, but not if that snake just ate
            if next_pos == snake['body'][-1] and (snake['length'] == 1 or snake['body'][-2] != snake['body'][-1]):
                return True
            return False
    return True

# returns a list moves that don't immediately result in collision
def determine_safe_moves(game_state, hero):
    head = game_state['you']['head']
    if not hero:
        villain_snake_index = 1 if game_state['board']['snakes'][0]['id'] == game_state['you']['id'] else 0
        head = game_state['board']['snakes'][villain_snake_index]['head']
    safe_moves = []
    for move in get_possible_moves(head).items():
        if is_position_safe(move[1], game_state):
            safe_moves.append(move)
    return safe_moves

# calculates the density of the snake's body around a given position within a specified radius
def density_score(game_state, position, radius=2):
    sum = 0
    for snake in game_state['board']['snakes']:
        # count snake segments in radius
        for segment in snake['body']:
            if abs(segment['x'] - position['x']) <= radius and abs(segment['y'] - position['y']) <= radius:
                sum += 1
    # count out-of-bounds squares in radius
    if game_state['board']['width'] - position['x'] <= radius:
        sum += (2 * radius + 1) * (radius - (game_state['board']['width'] - position['x']) + 1)
    if game_state['board']['height'] - position['y'] <= radius:
        sum += (2 * radius + 1) * (radius - (game_state['board']['height'] - position['y']) + 1)
    if position['x'] <= radius:
        sum += (2 * radius + 1) * (radius - position['x'] + 1)
    if position['y'] <= radius:
        sum += (2 * radius + 1) * (radius - position['y'] + 1)
    return -sum

# food score based on distance to foods
def food_score(game_state, position):
    food_positions = game_state['board']['food']
    if not food_positions: return 0
    sum = 0
    for food_position in food_positions:
        sum += math.sqrt(calculate_distance(position, food_position))
    return -sum / len(food_positions)

def head_on_score(game_state):
    villain_snake_index = 1 if game_state['board']['snakes'][0]['id'] == game_state['you']['id'] else 0
    villain_snake = game_state['board']['snakes'][villain_snake_index]
    if (calculate_distance(game_state['you']['head'], villain_snake['head']) > 1
        or game_state['you']['length'] == villain_snake['length']): return 0
    if game_state['you']['length'] > villain_snake['length']: return 1
    else: return -1

def length_score(game_state):
    hero_length = game_state['you']['length']
    villain_snake_index = 1 if game_state['board']['snakes'][0]['id'] == game_state['you']['id'] else 0
    villain_length = game_state['board']['snakes'][villain_snake_index]['length']
    if hero_length > villain_length: return 1
    if hero_length < villain_length: return -1
    else: return 0


def out_of_health(game_state, hero):
    if hero:
        return game_state['you']['health'] == 0
    if not hero:
        villain_snake_index = 1 if game_state['board']['snakes'][0]['id'] == game_state['you']['id'] else 0
        return game_state['board']['snakes'][villain_snake_index]['health'] == 0

# objective function considering nearby body density, closeness to food, and health
        # weights: (food, density, health, head-on, length)
def objective_function(game_state, weights): # minimize the output
    villain_snake_index = 1 if game_state['board']['snakes'][0]['id'] == game_state['you']['id'] else 0
    villain_head = game_state['board']['snakes'][villain_snake_index]['head']
    villain_health = game_state['board']['snakes'][villain_snake_index]['health']
    # arbitrary weights for now, can be trained
    eval = (weights[0] * (food_score(game_state, game_state['you']['head']) - food_score(game_state, villain_head)) 
        + weights[1] * (density_score(game_state, game_state['you']['head']) - density_score(game_state, villain_head)) 
        + weights[2] * (game_state['you']['health'] - villain_health)
        + weights[3] * (head_on_score(game_state))
        + weights[4] * (length_score(game_state)))
    return eval

# calculates next game state from given move
def calculate_next_game_state(game_state, move, hero):
    next_game_state = copy.deepcopy(game_state)
    # find snake
    hero_snake_index = 0 if game_state['board']['snakes'][0]['id'] == game_state['you']['id'] else 1
    villain_snake_index = 0 if hero_snake_index == 1 else 1
    snake = next_game_state['board']['snakes'][hero_snake_index] if hero else next_game_state['board']['snakes'][villain_snake_index]
    # decrement health
    snake['health'] -= 1
    if hero:
        next_game_state['you']['health'] -= 1
    # update our snake position
    snake['head'] = move
    snake['body'].insert(0, move)
    snake['body'].pop()
    if hero:
        # update 'you' snake position
        next_game_state['you']['head'] = move
        next_game_state['you']['body'].insert(0, move)
        next_game_state['you']['body'].pop()
    # remove food & refill health & add tail if necessary
    for food in next_game_state['board']['food']:
        # if food was eaten
        if food == move:
            next_game_state['board']['food'].remove(food)
            snake['health'] = 100
            if hero:
                next_game_state['you']['health'] = 100
            # add tail (at -2 because the tail hasn't been deleted yet)
            snake['body'].append(snake['body'][-1])
            snake['length'] += 1
            if hero:
                next_game_state['you']['body'].append(snake['body'][-1])
                next_game_state['you']['length'] += 1

    return next_game_state
    
def miniMax(game_state, depth, hero, weights):
    if depth == 0:
        return (objective_function(game_state, weights), None)
    safe_moves = determine_safe_moves(game_state, hero)
    if out_of_health(game_state, hero):
        return (float('-inf'), None) if hero else (float('inf'), None)
    if hero:
        value = float('-inf')
        best_move = None
        for move in safe_moves:
            new_state = calculate_next_game_state(game_state, move[1], True)
            score = miniMax(new_state, depth-1, False, weights)[0]
            if score > value:
                value = score
                best_move = move
        return (value, best_move)
    else:
        value = float('inf')
        best_move = None
        for move in safe_moves:
            new_state = calculate_next_game_state(game_state, move[1], False)
            score = miniMax(new_state, depth-1, True, weights)[0]
            if score < value:
                value = score
                best_move = move
        return (value, best_move)

        

def move(game_state: typing.Dict) -> typing.Dict:
    with open('weights.json', 'r') as f:
        weights = json.load(f)
    res = miniMax(game_state, 7, True, weights)
    print(res)
    return {"move": res[1][0]}


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