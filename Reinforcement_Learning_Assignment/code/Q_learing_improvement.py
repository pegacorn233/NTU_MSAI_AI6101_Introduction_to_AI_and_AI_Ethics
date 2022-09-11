from environment import CliffBoxPushingBase
from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class QAgent(object):
    def __init__(self):
        self.action_space = [1,2,3,4]
#         self.V = []
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.discount_factor=0.99       # ori = 0.99
        self.alpha=0.9      # ori = 0.5
        self.epsilon=0.01   # ori = 0.01

    def take_action(self, state):
        if epsilon_decay1 == True:
            self.epsilon = 0.005
        if epsilon_decay2 == True:
            self.epsilon = 0.001
        if epsilon_decay3 == True:
            self.epsilon = 0.0005
        if epsilon_decay4 == True:
            self.epsilon = 0.00001
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = self.action_space[np.argmax(self.Q[state])]
        return action

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        '''q-learning algorithm implementation'''
        q_predict = self.Q[state][action-1]
        q_target = reward + self.discount_factor * max(self.Q[next_state])
        self.Q[state][action - 1] += self.alpha * (q_target - q_predict)

if __name__ == '__main__':
    env = CliffBoxPushingBase()
    # you can implement other algorithms
    agent = QAgent()
    teminated = False
    rewards = []
    time_step = 0
    num_iterations = 1000
    #all_episodes = np.zeros(num_iterations)
    all_episodes = []
    all_rewards = []
    epsilon_decay1 = False
    epsilon_decay2 = False
    epsilon_decay3 = False
    epsilon_decay4 = False
    for i in range(num_iterations):
        if i > 200:
            epsilon_decay1 = True
        if i > 400:
            epsilon_decay2 = True
        if i > 600:
            epsilon_decay3 = True
        if i > 800:
            epsilon_decay4 = True
        env.reset()
        while not teminated:
            state = env.get_state()
            action = agent.take_action(state)
            #print(action)
            reward, teminated, _ = env.step([action])
            next_state = env.get_state()
            rewards.append(reward)
            print(f'step: {time_step}, actions: {action}, reward: {reward}')
            time_step += 1
            agent.train(state, action, next_state, reward)

        print(f'episode: {i}, rewards: {sum(rewards)}')
        print(f'print the historical actions: {env.episode_actions}')
        #all_episodes[i] += i
        #sum_rewards = sum(rewards)
        all_episodes.append(i)
        all_rewards.append(sum(rewards))
        teminated = False
        rewards = []

    print("all_episode: ", all_episodes)
    print("all_rewards: ", all_rewards)
    print('\r\nQ-Table:\n', agent.Q)

    # save q table into csv
    df_col = ['(Agent), (Box)', '[up, down, left, right]']
    df = pd.DataFrame(agent.Q.items(), columns = df_col).to_csv("final_q_table_improvement.csv", index=None)

    plt.figure()
    plt.plot(all_episodes, all_rewards)
    #plt.xlim((-1, num_iterations))
    #plt.ylim((-2000, 0))
    plt.xlabel('Episode', fontdict={'size' : 16})
    plt.ylabel('Rewards', fontdict={'size' : 16})
    plt.title('Episode Rewards vs. Episodes', fontdict={'size' : 20})
    plt.show()




