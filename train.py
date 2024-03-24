import subprocess
import time
import json
import random

# Define file paths
weights_file = 'weights.json'

def run_games(weights, num_games=10):
    win_count = 0
    loss_count = 0
    draw_count = 0

    for _ in range(num_games):
        # Construct command with weights
        command = "./battlesnake play -W 11 -H 11 --name 'snake1' --url http://127.0.0.1:8000 --name 'snake2' --url http://127.0.0.1:8001 > output.txt 2>&1"

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
    return win_count + (draw_count / 2) / (win_count + loss_count + draw_count)

def save_weights(weights):
    with open(weights_file, 'w') as f:
        json.dump(weights, f)

def load_weights():
    try:
        with open(weights_file, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        weights = None
    return weights

def hill_climbing(num_iterations=10, step_size=3, num_games_per_iteration=25):
    current_weights = load_weights()
    best_score = float('-inf')
    best_weights = current_weights.copy()

    for _ in range(num_iterations):
        # Perturb weights
        new_weights = [weight + random.uniform(-step_size, step_size) for weight in current_weights]

        # Run games with new weights
        win_count, loss_count, draw_count = run_games(new_weights, num_games=num_games_per_iteration)

        # Calculate win/loss ratio
        win_loss_ratio = calculate_win_loss_ratio(win_count, loss_count, draw_count)

        # Update weights if score improved
        if win_loss_ratio > best_score:
            best_score = win_loss_ratio
            best_weights = new_weights

        # Save best weights to file
        save_weights(best_weights)

        # Print results
        print(f"Iteration {_+1}/{num_iterations}: Best Score: {best_score}, Best Weights: {best_weights}")

        # Update current weights
        current_weights = best_weights.copy()

# Perform hill climbing
hill_climbing()