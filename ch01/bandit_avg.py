import numpy as np
from bandit import Bandit
from bandit import Agent

runs = 200
steps = 1000
epsiron = 0.1
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsiron)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action,reward)
        total_reward += reward
        rates.append(total_reward / (step+1))

    all_rates[run] = rates
    # こんな書き方出来るの！？？！？！？！！？
    # いや…まじで，pythonよしなにしてくれるところ多すぎだろ…

avg_rates = np.average(all_rates, axis=0)

import matplotlib.pyplot as plt
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.savefig('./bandit_avg.png')


