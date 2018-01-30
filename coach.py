from collections import deque
from MCTS import *
import numpy as np
import os
import copy

class Coach():
    """
    This class executes the self-play + learning.
    """
    def __init__(self, env, NNet):
        self.tempThreshold = 15
        self.numEps = 25
        self.updateThreshold = 0.6
        self.memory_length = 200000
        self.cpuct = 1
        self.numIters = 20
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        self.env = copy.deepcopy(env)
        self.board = self.env.reset() # 初始化棋盘
        self.NNet = NNet

        self.mcts = MCTS(self.NNet, self.env)
        self.NNet.load_model()

    def executeEpisode(self):
        train_memory = []
        # 棋盘的初始情况
        self.state = self.env.reset()
        # 黑棋先下
        self.curent_Player = 1
        episodeStep = 0
        temp_env = copy.deepcopy(self.env)
        temp_state = temp_env.reset()
        temp_env.reset()
        while True:
            episodeStep += 1
            reverse_state = self.get_reverse_state(self.transfer(copy.deepcopy(temp_state)), self.curent_Player)

            copy_state = copy.deepcopy(temp_state)
            if episodeStep > self.tempThreshold:
                ttt = 0
            else:
                ttt = 1
            pi = self.mcts.Simulation(copy_state, copy.deepcopy(self.curent_Player), T=ttt)

            # action = 65
            train_memory.append([np.reshape(reverse_state, [8,8]), self.curent_Player, pi, None])
            color = self.color_change(self.curent_Player)
            temp_env.reset()
            temp_env.state = copy.deepcopy(temp_state)
            enables_index = temp_env.possible_actions
            action = np.random.choice(len(pi), p=pi)
            if enables_index == [65]:
                action = 65
            temp_state, reward, done, _ = temp_env._step([action, color])
            self.curent_Player = -self.curent_Player
            r = self.reward_change(reward, self.curent_Player)
            if r != 0:
                return [(examples[0], examples[2], r*((-1)**(examples[1] != self.curent_Player))) for examples in train_memory]


    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train memory (which has a maximium length of maxlenofQueue).
        """
        train_memory = deque([], maxlen=self.memory_length)
        for i in range(self.numIters):
            print('------ITER ' + str(i+1) + '------')

            for eps in range(self.numEps):
                self.mcts = MCTS(self.NNet, self.env)   # reset search tree

                train_memory += self.executeEpisode()
            self.NNet.Train(train_memory)
            self.NNet.save_model()
        self.NNet.save_model()

    def transfer(self, state_qipan):
        state = dict()
        for i in range(8):
            for j in range(8):
                state[(i, j)] = 0
                if (state_qipan[2][i][j] == 0):
                    if (state_qipan[1][i][j] == 1):
                        state[(i, j)] = -1  # 白棋
                    elif (state_qipan[0][i][j] == 1):
                        state[(i, j)] = 1  # 黑棋
        return state

    def get_reverse_state(self, state, player_color):
        reverse_state = [state[(i, j)] * player_color for i in range(8) for j in range(8)]
        return reverse_state

    def color_change(self, color_in):

        if color_in == 1:
            return 0
        elif color_in == -1:
            return 1

    def reward_change(self, reward, current_player):
        if current_player == 1:
            return reward
        elif current_player == -1:
            return -reward

    def action_to_coordinate(self, board, action):
        return action // board.shape[-1], action % board.shape[-1]

    def make_place(self, board, action, player_color):
        coords = self.action_to_coordinate(board, action)

        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while (board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if (n > 0 and board[player_color, nx, ny] == 1):
                    nx = pos_x + dx
                    ny = pos_y + dy
                    while (board[opponent_color, nx, ny] == 1):
                        board[2, nx, ny] = 0
                        board[player_color, nx, ny] = 1
                        board[opponent_color, nx, ny] = 0
                        nx += dx
                        ny += dy
                    board[2, pos_x, pos_y] = 0
                    board[player_color, pos_x, pos_y] = 1
                    board[opponent_color, pos_x, pos_y] = 0
        return board




