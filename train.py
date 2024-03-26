import subprocess
import math
import json
import random

# Define file paths
snake1_weights_file = 'snake1_weights.json'
snake2_weights_file = 'snake2_weights.json'
current_weights_file = 'current_weights.json'

def run_games(num_games=10):
    win_count = 0
    loss_count = 0
    draw_count = 0

    for _ in range(num_games):
        # Construct command with weights
        command = "./battlesnake play -W 11 -H 11 --name 'snake1' --url http://127.0.0.1:8000 --name 'snake2' --url http://127.0.0.1:8001"

        try:
            # Execute command
            output = subprocess.run(command, capture_output=True).stderr

            # Parse output to extract game outcome
            outcome = extract_game_outcome(output)

            # Update win/loss/draw counts
            if outcome == 1:
                win_count += 1
            elif outcome == -1:
                loss_count += 1
            elif outcome == 0:
                draw_count += 1
        except subprocess.CalledProcessError as e:
            # Print error output
            print("Error:", e)
    return win_count, loss_count, draw_count



def extract_game_outcome(output):
    # Decode the bytes object to a string
    output_str = output.decode('utf-8')

    # Find the line indicating game completion and outcome
    lines = output_str.strip().split('\n')
    for line in reversed(lines):
        if "Game completed" in line:
            outcome_line = line
            break
    print(outcome_line)
    
    # Determine if the game was a win, loss, or draw
    if "snake1" in outcome_line:
        return 1  # Win
    elif "snake2" in outcome_line:
        return -1  # Loss
    elif "draw" in outcome_line:
        return 0  # Draw
    else:
        return None  # Unable to determine outcome

def calculate_win_loss_ratio(win_count, loss_count, draw_count):
    if win_count + loss_count == 0:
        return 0  # Avoid division by zero
    return (win_count + (draw_count / 2)) / (win_count + loss_count + draw_count)

def save_weights(weights, file):
    with open(file, 'w') as f:
        json.dump(weights, f)

def load_weights(file):
    try:
        with open(file, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        weights = None
    return weights


def hill_climbing(num_iterations=100, step_size=50, num_games_per_iteration=10):
    current_weights = load_weights(current_weights_file)

    for _ in range(num_iterations):
        # Perturb weights
        snake1_weights = [weight + random.uniform(-step_size, step_size) for weight in current_weights]
        snake2_weights = [weight + random.uniform(-step_size, step_size) for weight in current_weights]
        save_weights(snake1_weights, snake1_weights_file)
        save_weights(snake2_weights, snake2_weights_file)

        # Run games with new weights
        win_count, loss_count, draw_count = run_games(num_games=num_games_per_iteration)
        win_count += draw_count / 2
        loss_count += draw_count / 2

        if not (win_count == 0 and loss_count == 0):
            snake1_score = win_count**2 / (win_count**2 + loss_count**2)
            snake2_score = 1 - snake1_score
            # for each weight
            for i in range(5):
                current_weights[i] = ((snake1_weights[i] * snake1_score + snake2_weights[i] * snake2_score) + current_weights[i]) / 2

        save_weights(current_weights, current_weights_file)
        # Print results
        print(snake1_score)
        print(f"Iteration {_+1}/{num_iterations}: Last Score: {win_count} - {loss_count}, Current Weights: {current_weights}")

# Perform hill climbing
hill_climbing()