import numpy as np
import random
from FourRooms import FourRooms
from matplotlib import pyplot as plt

# Initialize the FourRooms environment for Scenario 1
fourRoomsObj = FourRooms('simple')

# Parameters for Q-learning
epsilon = 0.1  # Exploration rate
alpha = 0.5    # Learning rate
gamma = 0.9    # Discount factor

# Initialize the Q-table
state_space_size = 11 * 11
action_space_size = 4
Q = np.zeros((state_space_size, action_space_size))

# Function to get the state index from position
def get_state_index(position):
    return position[0] * 11 + position[1]

# Main function for Q-learning in Scenario 1
def main():
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Start a new epoch
        fourRoomsObj.newEpoch()
        
        # Get the initial position of the agent
        current_pos = fourRoomsObj.getPosition()
        state_index = get_state_index(current_pos)
        
        while not fourRoomsObj.isTerminal():
            # Choose an action based on epsilon-greedy policy
            if random.random() < epsilon:
                action = random.choice([FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT])
            else:
                action = np.argmax(Q[state_index])

            # Take the action and observe the outcome
            grid_cell, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            new_state_index = get_state_index(new_pos)

            # Compute the reward
            reward = 1 if packages_remaining == 0 else -0.01
            
            # Update Q-table using the Q-learning formula
            Q[state_index, action] += alpha * (reward + gamma * np.max(Q[new_state_index]) - Q[state_index, action])

            # Move to the next state
            state_index = new_state_index
        
        # Optionally print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} completed.")
    
    # After training, show the final path taken by the agent
    fourRoomsObj.showPath(-1, savefig='final_path_scenario1.png')

if __name__ == "__main__":
    main()
