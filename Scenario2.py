import numpy as np
import random
from FourRooms import FourRooms
from matplotlib import pyplot as plt

# Initialize the FourRooms environment for Scenario 2 (Multiple Package Collection)
fourRoomsObj = FourRooms('multi')

# Parameters for Q-learning
epsilon = 0.1  # Exploration rate
alpha = 0.5    # Learning rate
gamma = 0.9    # Discount factor

# Initialize the Q-table
state_space_size = 11 * 11  # As the environment grid is 11x11
action_space_size = 4       # Four possible actions: UP, DOWN, LEFT, RIGHT
Q = np.zeros((state_space_size, action_space_size))

# Function to get the state index from position
def get_state_index(position):
    x, y = position
    return (x - 1) * 11 + (y - 1)  # Adjusting for zero indexing

# Main function for Q-learning in Scenario 2
def main():
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Start a new epoch
        fourRoomsObj.newEpoch()
        
        # Get the initial position of the agent
        current_pos = fourRoomsObj.getPosition()
        state_index = get_state_index(current_pos)
        initial_packages_remaining = fourRoomsObj.getPackagesRemaining()

        while not fourRoomsObj.isTerminal():
            # Choose an action based on epsilon-greedy policy
            if random.random() < epsilon:
                action = random.choice([FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT])
            else:
                action = np.argmax(Q[state_index])

            # Take the action and observe the outcome
            _, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action)
            new_state_index = get_state_index(new_pos)

            # Compute the reward
            reward = 10 if packages_remaining < initial_packages_remaining else -0.01
            initial_packages_remaining = packages_remaining  # Update remaining packages

            # Update Q-table using the Q-learning formula
            Q[state_index, action] += alpha * (reward + gamma * np.max(Q[new_state_index]) - Q[state_index, action])

            # Debug information for each step
            print(f"Epoch: {epoch+1}, Step: From {current_pos} to {new_pos}, Action: {action}, Reward: {reward}, Remaining Packages: {packages_remaining}")

            # Update the current position
            current_pos = new_pos
            state_index = new_state_index

        # Optionally print completion of each epoch
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")

    # After training, show the final path taken by the agent
    fourRoomsObj.showPath(-1, savefig='final_path_scenario2.png')

if __name__ == "__main__":
    main()
