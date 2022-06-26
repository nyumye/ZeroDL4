import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # for importing the parent dirs
from collections import defaultdict
import numpy as np

from common.gridworld import GridWorld
# from ..common.gridworld import GridWorld


class McAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4
        self.epsilon = 0.1
        self.alpha = 0.1

        random_choice = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: defaultdict = defaultdict(lambda: random_choice)
        self.Q: defaultdict = defaultdict(lambda: 0)
        # self.cnts: defaultdict = defaultdict(lambda: 0)
        self.memory: list[tuple[int, int, int]] = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = np.random.choice(actions, p=probs)
        return action

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()


    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = reward + self.gamma*G
            key = (state, action)
            # self.cnts[key] += 1
            # self.Q[key] += (G - self.Q[key]) / self.cnts[key]
            self.Q[key] += (G - self.Q[key]) * self.alpha

            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

# 状態sにおける各行動の取られる確率を返す．
def greedy_probs(Q, state, epsilon: float = 0., action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)
