import tensorflow as tf
import os
import numpy as np
import random
from collections import deque
from tensorflow.python import pywrap_tensorflow

class RL_QG_agent:
    def __init__(self):
        tf.reset_default_graph()
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi2")

        #定义输入有关的参数
        self.n_action = 64
        self.state_size = 8 * 8 * 3
        #定义学习有关的参数
        self.batch_size = 32
        self.memory_size = 1000
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.epsilon = 0.9
        self.loss_now = 0.0
        self.replace_iteration = 500
        self.learn_step_counter = 0
        self.init_model()
        self.count = 0
        #replay memory
        self.memory = deque(maxlen=self.memory_size)
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_Net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Evaluate_Net')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def init_model(self):
        # 定义自己的 网络
        self.s = tf.placeholder(tf.float32, [None, 3, 8, 8])
        self.s_ = tf.placeholder(tf.float32, [None, 3, 8, 8])
        input_s = tf.reshape(self.s, [-1, 8, 8, 3])
        input_s_ = tf.reshape(self.s_, [-1, 8, 8, 3])
        #Fully connected layer
        with tf.variable_scope('Evaluate_Net'):
            weight_c1 = tf.Variable(tf.truncated_normal([3, 3, 3, 60], mean=1, stddev=0.1))
            biases_c1 = tf.Variable(tf.zeros([60]) + 1)
            conv_r1 = tf.nn.relu(tf.nn.conv2d(input=input_s, filter=weight_c1, strides=[1, 1, 1, 1],
                                              padding="VALID") + biases_c1)
            w1 = tf.Variable(tf.truncated_normal([6 * 6 * 60, 100], stddev=0.1))
            b1 = tf.Variable(tf.constant(0.1))
            out1 = tf.nn.relu(tf.matmul(tf.reshape(conv_r1, shape=[-1, 6 * 6 * 60]), w1) + b1)

            w2 = tf.Variable(tf.truncated_normal([100, self.n_action], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1))
            self.y = tf.matmul(out1, w2) + b2

        with tf.variable_scope('Target_Net'):
            weight_c2 = tf.Variable(tf.truncated_normal([3, 3, 3, 60], mean=1, stddev=0.1))
            biases_c2 = tf.Variable(tf.zeros([60]) + 1)
            conv_r2 = tf.nn.relu(tf.nn.conv2d(input=input_s_, filter=weight_c2, strides=[1, 1, 1, 1],
                                              padding="VALID") + biases_c2)
            w21 = tf.Variable(tf.truncated_normal([6 * 6 * 60, 100], stddev=0.1))
            b21 = tf.Variable(tf.constant(0.1))
            out21 = tf.nn.relu(tf.matmul(tf.reshape(conv_r2, shape=[-1, 6 * 6 * 60]), w21) + b21)

            w22 = tf.Variable(tf.truncated_normal([100, self.n_action], stddev=0.1))
            b22 = tf.Variable(tf.constant(0.1))
            self.y2 = tf.matmul(out21, w22) + b22

        with tf.variable_scope('loss'):
            #loss function
            self.y_ = tf.placeholder(tf.float32, [None, self.n_action])
            tf.stop_gradient(self.y_)
            self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
        with tf.variable_scope('Training'):
            #Training
            self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        #self.load_model()


        # 补全代码


    def place(self,state,enables,player):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        #测试的时候应该是根据确定的Q得出一个action，训练时采用epsilon-greedy 方法

        self.saver.save(self.sess,
                        os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi1"), 'parameter.ckpt'))
        if enables == [65]:
            return 65
        Q_values = self.get_Q(state)
        index = np.argsort(Q_values)
        for action in reversed(index):
            if action in enables:
                break
        return action

            #action = random.choice(enables)


    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    # 定义自己需要的函数
    def get_Q(self, state):
        #input: state
        #output: Q(s,a) for all actions
        return self.sess.run(self.y, feed_dict={self.s: [state]})[0]

    def get_Q_target(self, state):
        #input: state
        #output: Q(s,a) for all actions

        return self.sess.run(self.y2, feed_dict={self.s_: [state]})[0]

    def choose_actions(self, state, enables, player):
        #choose actions based on the epsilon-greedy method
        if np.random.rand() >= self.epsilon:
            return random.choice(enables)
        else:
            return self.place(state, enables, player)


    def store_memory(self, state, enables, action, reward, state_next, enables_next, done):

        self.memory.append((state, enables, action, reward, state_next, enables_next, done))

    def replay_memory(self):
        batch_state = []
        batch_y_ = []
        if self.learn_step_counter % self.replace_iteration == 0:
            self.sess.run(self.target_replace_op)
            print('target_params_replaced')
        #sample a batch
        batch_size = min(len(self.memory), self.batch_size)
        batch_index = np.random.randint(0, len(self.memory), batch_size)

        for i in batch_index:
            state_i, enables_i, action_i, reward_i, state_i_1, enables_i_1, done = self.memory[i]
            y_i = self.get_Q_target(state_i)

            if done:
                y_i[action_i] = reward_i
            else:
                qvalue, action = self.choose_qvalues_actions(state_i_1, enables_i_1)
                y_i[action_i] = reward_i + self.gamma * qvalue
            batch_state.append(state_i)
            #print(batch_state)
            batch_y_.append(y_i)
        self.learn_step_counter += 1

        self.sess.run(self.train, feed_dict={self.s:batch_state, self.y_:batch_y_})
        self.loss_now = self.sess.run(self.loss, feed_dict={self.s:batch_state, self.y_:batch_y_})



    def choose_qvalues_actions(self, state, enables):
        Q_values = self.get_Q(state)
        index = np.argsort(Q_values)
        for action in reversed(index):
            if action in enables:
                break
        qvalues = Q_values[action]
        return qvalues, action


    def save_model2(self,name="Reversi"):  # 保存 模型
        self.saver.save(self.sess, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), 'parameter.ckpt'))

    def load_model2(self, name="Reversi"):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), 'parameter.ckpt'))
