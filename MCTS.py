
import math
import numpy as np
from Game import *
import copy
import gym
import random
#需要define一个game来判断可行动作和done
class MCTS():

    def __init__(self, nnet, env):
        self.nnet = nnet
        self.count_dum = 0
        self.count_65 = 0
        tem_copy = copy.deepcopy(env)
        self.env = tem_copy

        self.Simulation_time = 20
        self.constant = 1
        #创建字典存放树的所有信息
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        #认为动作有65个，当action=64时代表是pass，与env交互时将其改成65
        self.n_actions = 65
        self.depth_max = 70

    def search(self, state, reverse_state, current_player):


        self.depth_count += 1
        if self.depth_count > (self.depth_max/2):
            # print("We are dangerous since the search depth is", self.depth_count)
            # print("State", reverse_state, "Player", current_player)
            pass
        #将状态变成str方便dict索引
        s = stringRepresentation(reverse_state)
        if s not in self.Es:
            #如果节点没有被存储，就存储该节点
            self.Es[s] = End_Reward(copy.deepcopy(state), copy.deepcopy(current_player))
        if self.Es[s] != 0:
            #游戏结束返回真实的reward
            return -self.Es[s]

        if s not in self.Ps:
            temp_state2 = np.reshape(reverse_state, [8,8])
            self.Ps[s], v = self.nnet.Predict(np.array(temp_state2))#用网络估计P(s,a)
            valid_mul_hot = get_enables_mul_hot(copy.deepcopy(state), self.color_change(copy.deepcopy(current_player)))#计算当前状态的可行动作，采用multi hot 编码从而mask生成的P(s,a)

            self.Ps[s] = self.Ps[s]*valid_mul_hot
            tt = np.sum(self.Ps[s])
            if not np.isclose(tt,0):
                self.Ps[s] /= np.sum(self.Ps[s])  # normalize变成概率
            else:
                # print("sum of Ps is too small and the sum is ", t)
                pass
            self.Vs[s] = valid_mul_hot#存储该状态下的可行动作
            self.Ns[s] = 0#初始化state被访问的次数
            return -v
        if len(self.Vs[s]) == 0:
            print('error'*100)
        enable_actions = self.Vs[s]
        best_now = -float('inf')
        best_action = -100
        #对于所有可行的action计算UCB，并选择best action
        for action in range(self.n_actions):
            #action 0~64
            if enable_actions[action]:
                #If actions are enable
                if (s,action) in self.Qsa:
                    UCB = self.Qsa[(s, action)] + \
                          self.constant*self.Ps[s][action]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s, action)])
                else:
                    UCB = self.constant * self.Ps[s][action] * math.sqrt(self.Ns[s])
                if UCB > best_now:
                    best_now = UCB
                    best_action = action
        action_choose = best_action
        if self.depth_count >= self.depth_max:
            return 0
        self.env.reset()
        self.env.state = copy.deepcopy(state)
        color = self.color_change(current_player)
        if action_choose != 64:
            next_state, _, _, _ = self.env._step([action_choose, color])
        else:
            next_state, _, _, _ = self.env._step([65, color])
        next_player = -current_player
        next_reverse_state = get_reverse_state(transfer(copy.deepcopy(next_state)), copy.deepcopy(next_player))
        v = self.search(copy.deepcopy(next_state), copy.deepcopy(next_reverse_state), copy.deepcopy(next_player))
        if (s,action_choose) in self.Qsa:
            self.Qsa[(s, action_choose)] = (self.Nsa[(s, action_choose)] * self.Qsa[(s, action_choose)] + v)/(self.Nsa[(s, action_choose)] + 1)
            self.Nsa[(s, action_choose)] += 1
        else:
            self.Qsa[(s, action_choose)] = v
            self.Nsa[(s, action_choose)] = 1
        self.Ns[s] += 1
        self.depth_count = self.depth_count-1
        return -v


    def Simulation(self, state, current_player, T=1):
        self.env.reset()
        self.env.state = copy.deepcopy(state)

        reverse_state = get_reverse_state(transfer(state), current_player)
        for i in range(self.Simulation_time):
            self.depth_count = 0
            self.search(copy.deepcopy(state), copy.deepcopy(reverse_state), copy.deepcopy(current_player))

        s = stringRepresentation(get_reverse_state(transfer(copy.deepcopy(state)), copy.deepcopy(current_player)))
        count_nmber = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.n_actions)]
        if T == 0:
            max_value = np.argmax(count_nmber)
            prob = [0] * len(count_nmber)
            prob[max_value] = 1
            return prob

        count_modefy = [s**(1./T) + 1e-10  for s in count_nmber]
        sum_all = sum(count_modefy)
        prob = [x/float(sum_all) for x in count_modefy]

        return prob

    def color_change(self, color_in):

        if color_in == 1:
            return 0
        elif color_in == -1:
            return 1

    def place_by_Tree(self, state, enables, player):
        """
        Make decision on action to take on by MC Tree.
        :param state: The current state, sized 3 * 8 * 8, Black pieces 1, White pieces -1.
        :param enables: The actions allowed.
        :param player: Black (1) or White (-1).
        :return: Action to take on. A number ranged 0~63, 64 means no available actions.
        """
        if enables == [65]:
            return 65  # 表示此时无子可走.
        if player == 0:
            color = 1
        elif player == 1:
            color = -1
        pi_prob = self.Simulation(state, color, T=1)

        index = np.argsort(pi_prob)
        for action in reversed(index):
            if action in enables:
                return action

    def onehot2num(self, count):
        tets = []
        for i in range(65):
            if count[i] != 0:
                tets.append(i)
        return tets


