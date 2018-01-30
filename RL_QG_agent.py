from MCTS import *
from UseNNet import UseNNet
import gym
import tensorflow as tf

class RL_QG_agent(object):
    def __init__(self):

        self.env1 = copy.deepcopy(gym.make('Reversi8x8-v0'))
        self.env1.reset()
        self.env2 = copy.deepcopy(self.env1)
        self.neural_network = UseNNet()
        self.neural_network.load_model(folder='Reversi')

        self.mcts = MCTS(self.neural_network, self.env1)

        self.mcts.Simulation_time = 100

    def place(self, state, enable, player):
        # c = int(player == 0)
        if player == 1:#load白棋最佳模型
            self.neural_network.load_model(folder="Reversi")
        elif player == 0:#load 黑棋最佳模型
            self.neural_network.load_model(folder="Reversi2")
        action = self.mcts.place_by_Tree(state, enable, player)

        self.env1.reset()
        self.env1.state = state
        return action

    def check_model(self, agent, agent_tmp, N=40):
        self.env2.reset()
        win1, win2 = self.pit(agent, agent_tmp, N)
        win3, win4 = self.pit(agent_tmp, agent, N)
        flag = (win1 + win4) > (win2 + win3)
        print("判断网络是否在进步", "新网络（黑）", win1, "旧网络（白）", win2, "新网络（白）", win4, "旧网络（黑）", win3, "进步否", flag)
        return flag

    def pit(self, agent1, agent2, N):

        agent1_win = 0
        agent2_win = 0
        for i_episode in range(int(N / 2)):
            observation = self.env2.reset()
            for t in range(100):
                action = [1, 2]
                enables = self.env2.possible_actions
                if len(enables) == 0:
                    action_ = self.env2.board_size ** 2 + 1
                else:
                    action_ = agent1.place(observation, enables, 0)  # 0 表示黑棋
                action[0] = action_
                action[1] = 0  # 黑棋 B  为 0
                observation, reward, done, info = self.env2.step(action)
                enables = self.env2.possible_actions
                if len(enables) == 0:
                    action_ = self.env2.board_size ** 2 + 1  # pass
                else:
                    action_ = agent2.place(observation, enables, 1)  # 调用自己训练的模型
                    if action_ not in enables:  # 如果输出的结果 非法，直接判断另一方赢
                        agent1_win += 1
                        break
                action[0] = action_
                action[1] = 1  # 白棋 W 为 1
                observation, reward, done, info = self.env2.step(action)
                if done:
                    black_score = len(np.where(self.env2.state[0, :, :] == 1)[0])
                    white_score = len(np.where(self.env2.state[1, :, :] == 1)[0])
                    if black_score > white_score:
                        agent1_win += 1
                    elif black_score < white_score:
                        agent2_win += 1
                    else:
                        pass
                    break
        return agent1_win, agent2_win


