import gym
import random
import numpy as np

from RL_QG_agent import RL_QG_agent
import alpha_beta as ab

env = gym.make('Reversi8x8-v0')

max_epochs = 20
player = 1
w_win = 0
b_win = 0

for i_episode in range(max_epochs):
    observation = env.reset()
    agent = RL_QG_agent()
    # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
    for t in range(100):
        print("回合", t)
        action = [1, 2]
        # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）

        ################### 黑棋 B ############################### 0表示黑棋
        #  这部分 黑棋 是 用 alpha-beta搜索
        enables = env.possible_actions
        if len(enables) == 0:
            action_ = env.board_size ** 2 + 1
        else:
            action_ =ab.place(observation, enables, 0)  # 0 表示黑棋
        action[0] = action_

        action[1] = 0  # 黑棋 B  为 0
        #env.render()
        observation, reward, done, info = env.step(action)
        ################### 白棋  W ############################### 1表示白棋
        # print(action,enables, done, reward)
        #env.render()
        enables = env.possible_actions
        # if nothing to do ,select pass
        if len(enables) == 0:
            action_ = env.board_size ** 2 + 1  # pass
        else:
            action_ = agent.place(observation, enables, player)  # 调用自己训练的模型
            #action_ = random.choice(enables)
            if action_ not in enables:  # 如果输出的结果 非法，直接判断另一方赢
                print(action_, "AI", enables)
                print("黑棋赢了！")
                b_win += 1
                break

        action[0] = action_
        action[1] = 1  # 白棋 W 为 1
        observation, reward, done, info = env.step(action)
        print(action, enables)
        print(done)
        if done:  # 游戏 结束
            black_score = len(np.where(env.state[0, :, :] == 1)[0])
            white_score = len(np.where(env.state[1, :, :] == 1)[0])
            if black_score > white_score:
                print("黑棋赢了！")
                b_win += 1

            elif black_score < white_score:
                print("白棋赢了！")
                w_win += 1
            else:
                print("平局")
            print(black_score)
            break
    print("黑棋：", b_win, "  白棋 ", w_win)
