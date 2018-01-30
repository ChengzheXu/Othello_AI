import tensorflow as tf
import numpy as np
from NeuralNetwork import NeuralNetwork as NNet
import os
from Game import *
class UseNNet(object):
    """
    Use the neural network, to train or predict.
    """
    def __init__(self):
        tf.reset_default_graph()
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        self.board_side = 8
        self.epochs = 100
        self.batch_size = 64

        self.NNet = NNet()
        self.NNet.dropout_prob = 0.0
        self.sess = tf.Session(graph=self.NNet.graph)
        self.action_size = self.board_side * self.board_side + 1 # 65

        # self.dropout_prob = 0.3

        # 开始建图.
        # self.sess = tf.Session(graph=self.NNet.graph)


        # 全局初始化
        with tf.Session() as global_sess:
            global_sess.run(tf.global_variables_initializer())

        # 初始化NNet图中的Variable.
        self.sess.run(tf.variables_initializer(self.NNet.graph.get_collection('variables')))

        self.saver = None

    def Train(self, train_set):
        """
        Train the network.
        :param batch: List of examples, each example is of form (state, pi, v)
        :return: None.
        """
        self.NNet.dropout_prob = 0.5
        pi_losses = [None]*self.epochs
        v_losses = [None]*self.epochs

        for epoch in range(self.epochs):
            print('epoch: ', (epoch+1))

            pi_losses[epoch] = []
            v_losses[epoch] = []

            for batch_id in range(int(len(train_set)/self.batch_size)):

                sample_ids = np.random.randint(len(train_set), size=self.batch_size)
                train_states, train_pis, train_vs = list(zip(*[train_set[i] for i in sample_ids]))

                feed_dict = \
                    {self.NNet.input_state: train_states, self.NNet.pi: train_pis, self.NNet.v: train_vs,
                              self.NNet.ifTrain: True}

                [_, pi_loss, v_loss] = self.sess.run([self.NNet.train_op, self.NNet.loss_pi, self.NNet.loss_v],
                                                     feed_dict=feed_dict)

                # self.saver.save(self.sess, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TMP"),
                #                                         'parameter.ckpt'), global_step=2)
                pi_losses[epoch].append(pi_loss)
                v_losses[epoch].append(v_loss)

                print("({batch}/{size}) | Loss_pi: {loss_pi:.4f} | Loss_v: {loss_v:.3f}".format(
                    batch=batch_id+1,
                    size=int(len(train_set)/self.batch_size),
                    loss_pi=np.mean(pi_loss),
                    loss_v=np.mean(v_loss)))
            self.save_model()#每个周期保存一次
        self.save_model()
        print("End Training.")
        self.NNet.dropout_prob = 0.0

        return None

    def Predict(self, state):
        """
        Predict pi and V according to board.
        :param state: numpy array with board
        :return: probability of all actions and V of the state.
        """
        self.NNet.dropout_prob = 0.0
        pi_prob, v_out = self.sess.run([self.NNet.pi_prob, self.NNet.v_out],
                                       feed_dict={self.NNet.input_state: np.reshape(state, [-1, 8, 8]),
                                                    self.NNet.ifTrain: False})
        # 1 * self.action_size, 1 x 1
        self.NNet.dropout_prob = 0.0
        return pi_prob[0], v_out[0]

    # def save_model(self):  # 保存 模型
    #     self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))
    #
    # def load_model(self):  # 重新导入模型
    #     self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))



    def save_model(self, folder='Reversi', filename='checkpoint.pth.tar'):
        """
        Save the model.
        :param folder: folder. default == 'Model'
        :param filename: filename. default == 'checkpoint.pth.tar'
        :return: None
        """
        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)

        if not self.saver:
            self.saver = tf.train.Saver(self.NNet.graph.get_collection('variables'))
        with self.NNet.graph.as_default():
            self.saver.save(self.sess, filepath)

        return None

    def load_model(self, folder='Reversi', filename='checkpoint.pth.tar'):
        """
        Load the model.
        :param folder: folder. default == 'Rversi'
        :param filename: filename. default == 'checkpoint.pth.tar'
        :return: None
        """

        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath+'.meta'):
            raise("NotExitError: No model in path {}".format(filepath))
        with self.NNet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)
        print("load model Now***********************")
        return None

    def place(self, state, enables, player):
        """
        Make decision on action to take on.
        :param state: The current state, sized 8 * 8, Black pieces 1, White pieces -1.
        :param player: Black (1) or White (-1).
        :return: Action to take on. A number ranged 0~63, 64 means no available actions and hence a pss action 65 is taken.
        """
        if (enables) == 1 and enables == [65]:
            return 65  # 表示此时无子可走.
        if player == 0:
            color = 1
        elif player == 1:
            color = -1
        state = np.reshape(get_reverse_state(transfer(state), color), [8,8])
        pi_prob, v_out = self.Predict(state)

        index = np.argsort(pi_prob)
        for action in reversed(index):
            if action in enables:
                return action
        else:
            return 65