import imp
from re import A


import numpy as np

# デフォルト10本腕の盗賊
class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)
    
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
    
    # 真の平均を知りたいからそれ返す用のメソッド
    def getTrueQValue(self, arm):
        return self.rates[arm]


# # 0番目のスロットに対してだけ，標本平均を出してみる．
# bandit = Bandit()
# Q = 0

# for n in range(1,10):
#     reward = bandit.play(0)
#     Q += (reward - Q) /n
#     print(Q)
# print(f"the true q value is : {bandit.getTrueQValue(0)}")


# # 0~10のスロットにおける標本平均を出す
# bandit = Bandit()
# Qs = np.zeros(10)
# ns = np.zeros(10)

# for n in range(10):
#     action = np.random.randint(1,10)
#     reward = bandit.play(action)

#     ns[action] += 1
#     Qs[action] = (reward - Qs[action]) / ns[action]
#     print("---------------")
#     print(Qs)


# Agentの実装
class Agent:
    def __init__(self, epsiron, action_size=10):
        self.epsiron = epsiron
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsiron:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)

        

# 動かしてみる．
import matplotlib.pyplot as plt

steps = 1000
epsiron = 0.1

bandit = Bandit()
agent = Agent(epsiron=epsiron)
total_reward = 0
total_rewards = []
rates = []

reward = 0

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))

print(total_rewards)
print("--------------")
print(rates)

# ステップごとに累積される報酬を描画(docker上で表示するの面倒そうだったので保存)
plt.xlabel("step")
plt.ylabel("total_reward")
plt.plot(total_rewards)
plt.show()
plt.savefig('./total_rewards.png')

# 新規描画開始？？
plt.figure()

# rateを描画
plt.xlabel("step")
plt.ylabel("rate")
plt.plot(rates)
plt.show()
plt.savefig('./rates.png')

