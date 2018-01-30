import copy

weights1 = [[99,-8,8,6,6,8,-8,99],
           [-8,-24,-4,-3,-3,-4,-24,-8],
           [8,-4,7,4,4,7,-4,8],
           [6,-3,4,1,1,4,-3,6],
           [6,-3,4,1,1,4,-3,6],
           [8,-4,7,4,4,7,-4,8],
           [-8,-24,-4,-3,-3,-4,-24,-8],
           [99,-8,8,6,6,8,-8,99]]

weights = [
    [500, -100, 100, 50, 50, 100, -100, 500],
   [ -100, -200, -50, -50, -50, -50, -200, -100],
  [  100, -50,60, 4, 4, 60, -50, 100],
  [  50, -50,4, 2, 2, 4, -50, 50],
   [ 50, -50, 4, 2, 2, 4, -50, 50],
   [ 100, -50, 60, 4, 4, 60, -50, 100],
    [-100, -200, -50, -50, -50, -50, -200, -100],
    [500, -100, 100, 50, 50, 100, -100, 500]]





def getMoves(player,state):# 输入的是 棋盘的状态，和 玩家 当前 可以下的点。
    moves = dict()
    for i in range(0,8):
        for j in range(0,8):
            if (i,j) not in state:
                if i+1<8 and j+1<8 and ((i+1,j+1) in state) and state[(i+1,j+1)] == player*-1:#location 1
                    p,q = i,j
                    while p+1<8 and q+1<8 and ((p+1,q+1) in state) and state[(p+1,q+1)] == player*-1:
                        p,q = p+1,q+1
                    if p+1<8 and q+1<8 and ((p+1,q+1) in state) and state[(p+1,q+1)] == player:
                        addToMoves(moves,i,j,1)
                if i+1<8 and ((i+1,j) in state) and state[(i+1,j)] == player*-1:#location 2
                    p,q = i,j
                    while p+1<8 and ((p+1,q) in state) and state[(p+1,q)] == player*-1:
                        p = p+1
                    if p+1<8 and ((p+1,q) in state) and state[(p+1,q)] == player:
                        addToMoves(moves,i,j,2)
                if i+1<8 and j-1>=0 and ((i+1,j-1) in state) and state[(i+1,j-1)] == player*-1:#location 3
                    p,q = i,j
                    while p+1<8 and q-1>=0 and ((p+1,q-1) in state) and state[(p+1,q-1)] == player*-1:
                        p,q = p+1,q-1
                    if p+1<8 and q-1>=0 and ((p+1,q-1) in state) and state[(p+1,q-1)] == player:
                        addToMoves(moves,i,j,3)
                if j+1<8 and ((i,j+1) in state) and state[(i,j+1)] == player*-1:#location 4
                    p,q = i,j
                    while q+1<8 and ((p,q+1) in state) and state[(p,q+1)] == player*-1:
                        q = q+1
                    if q+1<8 and ((p,q+1) in state) and state[(p,q+1)] == player:
                        addToMoves(moves,i,j,4)
                if j-1>=0 and ((i,j-1) in state) and state[(i,j-1)] == player*-1:#location 5
                    p,q = i,j
                    while q-1>=0 and ((p,q-1) in state) and state[(p,q-1)] == player*-1:
                        q = q-1
                    if q-1>=0 and ((p,q-1) in state) and state[(p,q-1)] == player:
                        addToMoves(moves,i,j,5)
                if i-1>=0 and j+1<8 and ((i-1,j+1) in state) and state[(i-1,j+1)] == player*-1:#location 6
                    p,q = i,j
                    while p-1>=0 and q+1<8 and ((p-1,q+1) in state) and state[(p-1,q+1)] == player*-1:
                        p,q = p-1,q+1
                    if p-1>=0 and q+1<8 and ((p-1,q+1) in state) and state[(p-1,q+1)] == player:
                        addToMoves(moves,i,j,6)
                if i-1>=0 and ((i-1,j) in state) and state[(i-1,j)] == player*-1:#location 7
                    p,q = i,j
                    while p-1>=0 and ((p-1,q) in state) and state[(p-1,q)] == player*-1:
                        p = p-1
                    if p-1>=0 and ((p-1,q) in state) and state[(p-1,q)] == player:
                        addToMoves(moves,i,j,7)
                if i-1>=0 and j-1>=0 and ((i-1,j-1) in state) and state[(i-1,j-1)] == player*-1:#location 8
                    p,q = i,j
                    while p-1>=0 and q-1>=0 and ((p-1,q-1) in state) and state[(p-1,q-1)] == player*-1:
                        p,q = p-1,q-1
                    if p-1>=0 and q-1>=0 and ((p-1,q-1) in state) and state[(p-1,q-1)] == player:
                        addToMoves(moves,i,j,8)
    return moves

def addToMoves(moves,i,j,direction):
    if (i,j) not in moves:
        moves[(i,j)] = [direction]
    else:
        moves[(i,j)].append(direction)
    return moves

def evaluateState(player,state):
    val = 0
    if(len(state) < 32):
        for pos,s in list(state.items()):
            val += s*weights[pos[0]][pos[1]]
        return val*player
    else:
        for pos,s in list(state.items()):
            val += s*weights1[pos[0]][pos[1]]/2
        return val*player

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
pl  = 1

def place(state_qipan ,enables,player):
    if player ==0 :
        player =1
    else:
        player = -1
    pl = player
    action = 65
    depth = 6 # 最大搜索深度
    state = transfer(state_qipan)
    state_pre = state
    (val, resState) = play('null', 'root', player, 0, depth, state, player, -10000, 10000)
    for i in range(8):
        for j in range(8):
            if( (not ((i,j) in state_pre )) and ((i,j) in resState)):
                action = i*8 +j
    return  action

def placePiece(player, move, directions, state):  # 这里就是 进行下棋操作。并更新棋盘
    res = copy.copy(state)
    res[(move[0], move[1])] = player
    for direct in directions:
        i, j = move[0], move[1]
        if direct == 1:
            while i + 1 < 8 and j + 1 < 8 and ((i + 1, j + 1) in state) and state[(i + 1, j + 1)] == player * -1:
                i, j = i + 1, j + 1
                res[(i, j)] = player
        elif direct == 2:
            while i + 1 < 8 and ((i + 1, j) in state) and state[(i + 1, j)] == player * -1:
                i = i + 1
                res[(i, j)] = player
        elif direct == 3:
            while i + 1 < 8 and j - 1 >= 0 and ((i + 1, j - 1) in state) and state[(i + 1, j - 1)] == player * -1:
                i, j = i + 1, j - 1
                res[(i, j)] = player
        elif direct == 4:
            while j + 1 < 8 and ((i, j + 1) in state) and state[(i, j + 1)] == player * -1:
                j = j + 1
                res[(i, j)] = player
        elif direct == 5:
            while j - 1 >= 0 and ((i, j - 1) in state) and state[(i, j - 1)] == player * -1:
                j = j - 1
                res[(i, j)] = player
        elif direct == 6:
            while i - 1 >= 0 and j + 1 < 8 and ((i - 1, j + 1) in state) and state[(i - 1, j + 1)] == player * -1:
                i, j = i - 1, j + 1
                res[(i, j)] = player
        elif direct == 7:
            while i - 1 >= 0 and ((i - 1, j) in state) and state[(i - 1, j)] == player * -1:
                i = i - 1
                res[(i, j)] = player
        elif direct == 8:
            while i - 1 >= 0 and j - 1 >= 0 and ((i - 1, j - 1) in state) and state[(i - 1, j - 1)] == player * -1:
                i, j = i - 1, j - 1
                res[(i, j)] = player
    return res

def getLocation(move):
    col = ['a','b','c','d','e','f','g','h']
    return col[move[1]] + str(move[0]+1)

def play(dad,node,player,depth,maxDepth,state,minmax,alpha,beta): # 
    #
    val = minmax*-6000
    if depth == maxDepth:
        val = evaluateState(pl,state)
        return (val,state)
    res = copy.copy(state)
    moves = getMoves(player,state)
    if dad == 'pass' and len(moves) == 0:
        val = evaluateState(pl ,state)
        return (val,state)
    if len(moves) == 0:
        (result,st) = play(node,'pass',player*-1,depth+1,maxDepth,copy.copy(state),minmax*-1,alpha,beta)
        if minmax == 1:
            val = max(val,result)
            if val >= beta:
                return (val,res)
            alpha = max(alpha,val)
        elif minmax == -1:
            val = min(val,result)
            if val<= alpha:

                return (val,res)
            beta = min(beta,val)

    else:
        for move,directions in sorted(moves.items()):
            cur = placePiece(player,move,directions,state)
            (result,st) = play(node,getLocation(move),player*-1,depth+1,maxDepth,copy.copy(cur),minmax*-1,alpha,beta)
            if minmax == 1:
                if val < result:
                    val,res = result, copy.copy(cur)
                if val >= beta:
                    return (val,res)
                alpha = max(alpha,val)
            elif minmax == -1:
                if val > result:
                    val,res = result, copy.copy(cur)
                if val <= alpha:

                    return (val,res)
                beta = min(beta,val)

    return (val,res) 
