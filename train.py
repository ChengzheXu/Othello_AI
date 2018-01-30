from coach import *
import gym
from UseNNet import *
import sys
sys.setrecursionlimit(10**6)

if __name__=="__main__":

    env = gym.make('Reversi8x8-v0')
    env.reset()
    nnet = UseNNet()


    c = Coach(env, nnet)
    c.learn()
