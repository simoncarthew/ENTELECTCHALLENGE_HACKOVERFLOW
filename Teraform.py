import torch
import os
import argparse

def save_q_table(q_table, filename="q_table.pt"):
    torch.save(q_table, filename)
    print(f"Q-table saved to {filename}")

def load_q_table(filename="q_table.pt"):
    if os.path.exists(filename):
        q_table = torch.load(filename)
        print(f"Q-table loaded from {filename}")
        return q_table
    else:
        print(f"No Q-table found at {filename}")
        return None
    
def save_best_path(path, cumulative_reward, filename="best_path.txt"):
    with open(filename, "w") as f:
        f.write(f"Best cumulative reward: {cumulative_reward}\n\n")
        f.write("Best path:\n")
        for i, (state, action) in enumerate(path, 1):
            f.write(f"Step {i}:\n")
            f.write(f"  State: (x={state[0]}, y={state[1]}, points={state[2]}, quality={state[3]}, days_remaining={state[4]})\n")
            f.write(f"  Action: {action_to_string(action)}\n\n")
    print(f"Best path saved to {filename}")

def action_to_string(action):
    # Convert action number to a string representation
    actions = ["North", "South", "East", "West"]
    return actions[action]

def action_to_string(action):
    # Convert action number to a string representation
    actions = ["North", "South", "East", "West"]
    return actions[action]
    
def main():
    # Set the hyperparameters
    num_episodes = 5000
    alpha = 0.1
    steps_per_episode = 90
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    decay = 0.001
    discount = 0.99

    best_cumulative_reward = float('-inf')
    best_path = []

    # Initialize the Q-table with zeros
    q_table = torch.zeros(22, 22, 1, 1, 1, 4)  # (x, y) , points, quality, days remaining, action #TODO

    def reward_function( state, next_state, done, South_Pole):
        reward = -1
        if done:
            if South_Pole:
                reward += 1000              # large reward for reaching the South pole
            else:
                reward -= 5000              # severe penalty for not reaching south pole
        else :
            reward = next_state[1] * next_state[2] * (next_state[3] - 1)
        return reward

    for episode in range(num_episodes):
        gridworld.newEpoch() 
        state = (0,11) + (0 , 0, 37)            #TODO Level 1 , change for each level
        done = False 
        cumulative_reward = 0 
        current_path = []
        reached_south_pole = False

        for step in range(steps_per_episode):
            #epsilon-greedy strategy
            if torch.rand(1).item() < exploration_rate:
                action = torch.randint(0, 4, (1,)).item()
            else:
                action = torch.argmax(q_table[state[0], state[1], state[2]]).item()

            # take action
            valid = False 
            while not valid:
                valid, next_pos, points, quality, days_remaining, done, South_Pole = gridworld.takeAction(action)
            next_state = next_pos + (points, quality, days_remaining)

            # Add the current state and action to the path
            current_path.append((state, action))

            # reward 
            reward = reward_function(state, next_state, done, South_Pole)
            cumulative_reward += reward

            #update Q-table

            q_table[state[0],state[1], state[2], state[3], action] = q_table[state[0],state[1],state[2],state[3], action]*(1-alpha) + alpha * (reward + discount*torch.max(q_table[next_state[0],next_state[1],next_state[2],next_state[3]]))

            state = next_state

            if South_Pole:
                reached_south_pole = True

            if done :
                break
        
        # After each episode, check if this is the best path so far
        # Only consider it the best if it reached the South Pole
        if reached_south_pole and cumulative_reward > best_cumulative_reward:
            best_cumulative_reward = cumulative_reward
            best_path = current_path
            best_q_table = q_table.clone()  # Store a copy of the current Q-table

        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) *torch.exp(torch.tensor(-decay * episode))

    # After all episodes, print and save the best path
    print(f"Best cumulative reward: {best_cumulative_reward}")
    print("Best path:")
    for state, action in best_path:
        print(f"State: {state}, Action: {action}")

    # Save the best path to a file
    if best_path:
        save_best_path(best_path, best_cumulative_reward)
    else:
        print("No valid path found that reaches the South Pole.")

    if best_q_table is not None:
        save_q_table(best_q_table, "best_q_table.pt")
    else:
        print("No valid path found that reaches the South Pole.")
                

if __name__ == "__main__":
    main()