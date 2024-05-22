import numpy as np
import matplotlib.pyplot as plt
import copy
bomb_location = (1,3)
gold_location = (0,3)
terminal_states = [bomb_location, gold_location]

class GridWorld:
    ## Initialise starting data
    def __init__(self):
    # Set information about the gridworld
        self.height = 10
        self.width = 10
        self.grid = np.zeros((self.height, self.width)) - 1

        # Set random start location for the agent
        # self.current_location = (np.random.randint(0,9), 1 )

        # Set locations for the bomb and the gold
        self.bomb_location = (1,3)
        self.gold_location = (0,3)
        self.terminal_states = [self.bomb_location, self.gold_location]

        # Set grid rewards for special cells
        self.grid[self.bomb_location[0], self.bomb_location[1]] = -100
        self.grid[self.gold_location[0], self.gold_location[1]] = 100

        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        # Set obstacles at specific locations (representing narrow passages/doors)
        self.obstacle_locations = [(i, 5) for i in range(self.height) if i != 4]
        for location in self.obstacle_locations:
            self.grid[location[0], location[1]] = -101  # Assign a high negative reward for obstacles
        
        self.obstacle_locations = [(i, 2) for i in range(self.height) if i != 2]
        for location in self.obstacle_locations:
            self.grid[location[0], location[1]] = -101  # Assign a high negative reward for obstacles

        self.obstacle_locations = [(i, 7) for i in range(self.height) if i != 3]
        for location in self.obstacle_locations:
            self.grid[location[0], location[1]] = -101  # Assign a high negative reward for obstacles
            
        self.grid[7, 3] = -101
        self.grid[4, 3] = -101
        self.grid[6, 0] = -101
        self.grid[6, 0] = -101
        self.grid[2, 9] = -101
        self.grid[8, 8] = -101
        
        while True:
            self.current_location = (np.random.randint(0,9), np.random.randint(0,9))
            if self.grid[self.current_location[0], self.current_location[1] ] != -1:
                print("self.current_location")
                print(self.current_location)
                continue
            break
        
        
    ## Put methods here:
    def get_available_actions(self):
        """Returns possible actions"""
        return self.actions

    def agent_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid
    def set_current_location(self, newvalue) :
        self.current_location = (newvalue[0], newvalue[1])
        
    def get_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.grid[ new_location[0], new_location[1]]

    def print_grid(self):
        print(self.grid)
        
    def make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.current_location

        # UP
        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)

        # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)

        return reward

    def check_state(self):
        """Check if the agent is in a terminal state (gold or bomb), if so return 'TERMINAL'"""
        if self.current_location in self.terminal_states:
            return 'TERMINAL'


class ValueIterationAgent():
    # Choose a random action
    def __init__(self, learning_rate) -> None:
        self.values = np.full((10, 10, 4), 0.01)  # Q(s, a)
        self.learning_rate = learning_rate
    def learn(self, old_state, reward, new_state, action):
        action_index = ['UP', 'DOWN', 'LEFT', 'RIGHT'].index(action)

        # Calculate the maximum future reward
        max_future_reward = np.max(self.values[new_state[0], new_state[1]])

        # Calculate the new value
        new_v = reward + max_future_reward
        old_v = self.values[old_state[0], old_state[1], action_index]

        # Update the Q-value for the old state and action pair
        self.values[old_state[0], old_state[1], action_index] += self.learning_rate * (new_v - old_v)

    
    def choose_action(self, available_actions, environment):
        # Get the action values for the current state
        action_values = self.values[environment.current_location[0], environment.current_location[1]]

        # Choose the action with the highest value
        best_action_index = np.argmax(action_values)
        best_action = available_actions[best_action_index]

        return best_action
    
    
    
def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log

    for trial in range(trials): # Run trials
        
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        game_over = False
        print("------------")
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            
                
            old_state = environment.current_location
            action = agent.choose_action(environment.actions, environment)
            reward = environment.make_step(action)
            new_state = environment.current_location
            if trial > 298:
                
                print(old_state)
                print(action)
            if learn == True: # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True
                break
            
        print(cumulative_reward)
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
    
    
    return reward_per_episode # Return performance log