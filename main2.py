import typing
import copy 
import sys
import json

DEPTH = 8

def info() -> typing.Dict:
    return {"apiversion": "1", "author": "Test", "color": "#888888", "head": "default", "tail": "default"}

def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")
    return {}

# calculates Manhattan distance from one point to another
def calculate_distance(pos1, pos2):
    return abs(pos1['x'] - pos2['x']) + abs(pos1['y'] - pos2['y'])

# finds the coordinates of all possible moves
def get_possible_moves(head):
    return {'up': {'x': head['x'], 'y': head['y'] + 1}, 
            'down': {'x': head['x'], 'y': head['y'] - 1},
            'left': {'x': head['x'] - 1, 'y': head['y']}, 
            'right': {'x': head['x'] + 1, 'y': head['y']}}

# checks for immediate collisions
def is_position_safe(next_pos, game_state):
    width, height = game_state['board']['width'], game_state['board']['height']
    x, y = next_pos['x'], next_pos['y']

    # Check for boundary conditions
    if not (0 <= x < width) or not (0 <= y < height):
        return False
    
    # Check for collisions with snakes (including self)
    for snake in game_state['board']['snakes']:
        if next_pos in snake['body']:
            if next_pos == snake['body'][-1] and (snake['length'] == 1 or snake['body'][-2] != snake['body'][-1]):
                return True # no collision with tail
            return False # collision with snake body
    return True # no collision

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

# snake moves toward open space
def density_score(game_state, position, radius=2):
    width = game_state['board']['width']
    height = game_state['board']['height']
    sum = 0

    # count snake segments in radius
    for snake in game_state['board']['snakes']:
        for segment in snake['body']:
            if abs(segment['x'] - position['x']) <= radius and abs(segment['y'] - position['y']) <= radius:
                sum += 1

    diameter = 2 * radius + 1

    # count out-of-bounds squares in radius
    if width - position['x'] <= radius:
        sum += diameter * (radius - (width - position['x']) + 1)
    if height - position['y'] <= radius:
        sum += diameter * (radius - (height - position['y']) + 1)
    if position['x'] <= radius:
        sum += diameter * (radius - position['x'] + 1)
    if position['y'] <= radius:
        sum += diameter * (radius - position['y'] + 1) 

    return -sum

# snake moves toward food
def food_score(game_state, position):
    food_positions = game_state['board']['food']

    if not food_positions: 
        return 0
    
    # calculate the distances to all food positions
    distances = [calculate_distance(position, food_pos) for food_pos in food_positions]

    # sum the reciprocals of the distances, return the negation
    total_score = sum(1 / distance for distance in distances)
    return -total_score

# snake moves toward opponent's head if longer than opponent
def head_on_score(game_state):
    hero_snake = game_state['you']
    hero_head = hero_snake['head']
    hero_length = hero_snake['length']
    villain_snake_index = 1 if game_state['board']['snakes'][0]['id'] == hero_snake['id'] else 0
    villain_snake = game_state['board']['snakes'][villain_snake_index]
    villain_length = villain_snake['length']
    villain_head = villain_snake['head']

    if calculate_distance(hero_head, villain_head) > 1 or hero_length == villain_length:
        return 0
    elif hero_length > villain_length:
        return 1
    else:
        return -1

# heuristic for the minimax algorithm
def objective_function(game_state, weights):
    hero_head = game_state['you']['head']
    hero_health = game_state['you']['health']
    hero_length = game_state['you']['length']
    all_snakes = game_state['board']['snakes']
    villain_snake_index = 1 if all_snakes[0]['id'] == game_state['you']['id'] else 0
    villain_snake = all_snakes[villain_snake_index]
    villain_head = villain_snake['head']
    villain_health = villain_snake['health']
    villain_length = villain_snake['length']

    food_diff = food_score(game_state, hero_head) - food_score(game_state, villain_head)
    density_diff = density_score(game_state, hero_head) - density_score(game_state, villain_head)
    health_diff = hero_health - villain_health
    length_diff = hero_length - villain_length
    head_on = head_on_score(game_state)

    eval = (weights[0] * food_diff +
            weights[1] * density_diff +
            weights[2] * health_diff +
            weights[3] * length_diff +
            weights[4] * head_on)
    
    return eval

def out_of_health(game_state, hero):
    if hero:
        return game_state['you']['health'] == 0
    else:
        all_snakes = game_state['board']['snakes']
        villain_snake_index = 1 if all_snakes[0]['id'] == game_state['you']['id'] else 0
        return all_snakes[villain_snake_index]['health'] == 0

# calculates what the next game state will be after a given move
def calculate_next_game_state(game_state, move, hero):
    next_game_state = copy.deepcopy(game_state)

    hero_snake_index = 0 if game_state['board']['snakes'][0]['id'] == game_state['you']['id'] else 1
    villain_snake_index = 1 - hero_snake_index

    # choose the current snake
    snake = next_game_state['board']['snakes'][hero_snake_index] if hero else next_game_state['board']['snakes'][villain_snake_index]
    
    # decrement health
    snake['health'] -= 1
    if hero:
        next_game_state['you']['health'] -= 1

    # update snake position
    snake['head'] = move
    snake['body'].insert(0, move)
    snake['body'].pop()

    if hero:
        # update 'you' snake position
        next_game_state['you']['head'] = move
        next_game_state['you']['body'].insert(0, move)
        next_game_state['you']['body'].pop()

    # check for food and update health, length, body
    for food in next_game_state['board']['food']:
        if food == move:
            next_game_state['board']['food'].remove(food)
            snake['health'] = 100
            snake['body'].append(snake['body'][-1])
            snake['length'] += 1
            if hero:
                next_game_state['you']['health'] = 100
                next_game_state['you']['body'].append(snake['body'][-1])
                next_game_state['you']['length'] += 1
            break

    return next_game_state
  
# minimax algorithm to calculate the best move
def miniMax(game_state, depth, hero, weights, alpha=float('-inf'), beta=float('inf')):
    if depth == 0:
        return (objective_function(game_state, weights), None)
    
    safe_moves = determine_safe_moves(game_state, hero)

    if not safe_moves or out_of_health(game_state, hero):
        return (float('-inf'), None) if hero else (float('inf'), None)
    
    if hero:
        best_score = float('-inf')
        best_move = None
        for move in safe_moves:
            new_state = calculate_next_game_state(game_state, move[1], hero)
            score, _ = miniMax(new_state, depth - 1, not hero, weights, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break # prune
        return (best_score, best_move)
    else:
        best_score = float('inf')
        best_move = None
        for move in safe_moves:
            new_state = calculate_next_game_state(game_state, move[1], hero)
            score, _ = miniMax(new_state, depth - 1, not hero, weights, alpha, beta)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # prune
        return (best_score, best_move)
        
        
def move(game_state: typing.Dict) -> typing.Dict:
    # check for game over
    if len(game_state['board']['snakes']) == 1:
        return {"move": "up"}
    
    with open('snake2_weights.json', 'r') as f:
        weights = json.load(f)

    print(game_state['you']['latency'])

    return {"move": miniMax(game_state, DEPTH, True, weights)[1][0]}

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