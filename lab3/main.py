import matplotlib.pyplot as plt

from GridWorld import *
environment = GridWorld()
agent = ValueIterationAgent(0.1)

# reward_per_episode = play(environment, agent, trials=500, learn=True)
# reward_per_episode = play(environment, agent, trials=500, learn=True)
alphas = [0.1, 0.5, 0.9]  # Різні значення α для експериментів

for alpha in alphas:
    agent = ValueIterationAgent(learning_rate=alpha)
    rewards = play(environment, agent, trials=300, max_steps_per_episode=1000, learn=True)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.show()

environment.print_grid()
# Simple learning curve
