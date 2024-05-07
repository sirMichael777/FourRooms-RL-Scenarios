from FourRooms import FourRooms

# Initialize the FourRooms environment for testing
test_env = FourRooms('simple')

# Simulate a series of actions
actions = [FourRooms.UP, FourRooms.LEFT, FourRooms.RIGHT, FourRooms.DOWN]

# Initial position
print(f"Initial Position: {test_env.getPosition()}")

# Test each action
for action in actions:
    grid_cell, new_pos, packages_remaining, is_terminal = test_env.takeAction(action)
    print(f"Action: {action}, New Position: {new_pos}, Packages Remaining: {packages_remaining}, Is Terminal: {is_terminal}")

# Check if the environment resets correctly
test_env.newEpoch()
print(f"Position after reset: {test_env.getPosition()}")
