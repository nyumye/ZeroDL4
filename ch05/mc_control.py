from collections import defaultdict
import numpy as np


class McAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4

        random_choice = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: defaultdict = defaultdict(lambda: random_choice)
        self.Q: defaultdict = defaultdict(lambda: 0)
        self.cnts: defaultdict = defaultdict(lambda: 0)
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

    def resert(self):
        self.memory.clear()


    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G += reward + self.gamma*G
            key = (state, action)
            self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) / self.cnts[key]

            self.pi[state] = greedy_probs(self.Q, state)

def greedy_probs(Q, state, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    action_probs = {action: 0.0 for action in range(action_size)}
    action_probs[max_action] = 1
    return action_probs
