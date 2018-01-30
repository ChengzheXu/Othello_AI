



import numpy as np


def get_enables_mul_hot(state, color):
    """
    利用reversi里面的enables将其转变成多活跃编码，即只有enables所在的行才是1
    :param state:
    :param color:
    :return:
    """
    n_actions = 65
    enables = get_possible_actions(state, color)
    enables_mul_hot = [0] * n_actions
    if len(enables) == 0:
        enables_mul_hot[-1] = 1
        return np.array(enables_mul_hot)
    for i in enables:
        enables_mul_hot[i] = 1
    return np.array(enables_mul_hot)


    pass

#这段代码来自环境reversi
def get_possible_actions(board, player_color):
    #Use the code in the reversi.py to get enable actions and then use it to determnie whether it has_legal_actions
    #But we modefy this code a liitile bit so that len(actions) can be zero
    actions = set()
    d = board.shape[-1]
    opponent_color = 1 - player_color
    for pos_x in range(d):
        for pos_y in range(d):
            if (board[2, pos_x, pos_y] == 0):
                continue
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
                        actions.add(pos_x * 8 + pos_y)
    actions = list(actions)
    return actions


def have_enable_action(state, player_color):
    #改变state的编码方式从1,-1 改成 0,1
    color = 0
    if player_color == 1:
        color = 0
    elif player_color == -1:
        color = 1
    actions = get_possible_actions(state, color)
    return len(actions) != 0

def End_Reward(state, player):
    # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
    # player = 1
    if have_enable_action(state, player):
        return 0
    if have_enable_action(state, -player):
        return 0

    return game_reward(state, player)


def game_reward(state, player_color):
    #根据输入的player_color 输出对应的1，-1
    black_score = len(np.where(state[0, :, :] == 1)[0])
    white_score = len(np.where(state[1, :, :] == 1)[0])
    if black_score > white_score:
        # print("黑棋赢了！")
        if player_color == 1:
            return 1
        elif player_color == -1:
            return -1

    elif black_score < white_score:
        # print("白棋赢了！")
        if player_color == 1:
            return -1
        elif player_color == -1:
            return 1
    else:
        return 1e-10#和棋的时候返回一个很小的非零值，从而保证MCTS能够return

def stringRepresentation(state):

    # return np.array_str(state)
    return str(state)






def transfer(state_qipan):
    state = dict()
    for i in range(8):
        for j in range(8):
            if(state_qipan[2][i][j] == 0):
                if(state_qipan[1][i][j] == 1):
                    state[(i, j)] = -1 # 白棋
                elif( state_qipan[0][i][j]==1 ):
                    state[(i, j)] =  1  # 黑棋
            # else:
            #     state[i][j] = 0

    return state






def transfer(state_qipan):
    state = dict()
    for i in range(8):
        for j in range(8):
            state[(i, j)] = 0
            if(state_qipan[2][i][j] == 0):
                if(state_qipan[1][i][j] == 1):
                    state[(i, j)] = -1 # 白棋
                elif( state_qipan[0][i][j]==1 ):
                    state[(i, j)] =  1  # 黑棋
            # else:
            #     state[i][j] = 0

    return state

def get_reverse_state(state, player_color):
    reverse_state = [state[(i, j)] * player_color for i in range(8) for j in range(8)]
    return reverse_state