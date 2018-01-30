import tensorflow as tf
import os
class NeuralNetwork(object):
    """
    Construct the neural network for MCTS.
    """
    def __init__(self):
        tf.reset_default_graph()
        # 对tensorflow库中的函数重命名.
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")

        relu = tf.nn.relu
        tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        dropout = tf.nn.dropout
        dense = tf.layers.dense

        # 定义网络超参数：
        self.learning_rate = 0.0001
        self.conv_channels = [512, 512, 256]  # 卷积层通道数，层数固定.
        self.conv_kernel = [3, 3]
        self.fc_neural = [1024, 512]  # 全连接层大小，层数固定
        self.dropout_prob = 0.0

        # 棋盘的边长为 8.
        board_side = 8
        # 动作向量维度，default==65. 其中 0~63 均是正常合法输出动作，64表示无子可走.
        self.action_num = board_side * board_side + 1 # 65

        # 构建神经网络的流图.
        self.graph = tf.Graph()

        with self.graph.as_default():
            # 输入棋盘状态.
            self.input_state = tf.placeholder(tf.float32, shape=[None, board_side, board_side])
            # self.dropout_prob = tf.placeholder(tf.float32)
            self.ifTrain = tf.placeholder(tf.bool, name="if_train")

            input_state_image = tf.reshape(self.input_state, [-1, board_side, board_side, 1])

            h_conv1 = relu(BatchNormalization(self.conv2d(input_state_image, self.conv_channels[0], 'same'),
                                              axis=3, training=self.ifTrain))
            # batch_size * board_side * board_side * conv_channels[0]
            h_conv2 = relu(BatchNormalization(self.conv2d(h_conv1, self.conv_channels[1], 'same'),
                                              axis=3, training=self.ifTrain))
            # batch_size * board_side * board_side * conv_channels[1]
            h_conv3 = relu(BatchNormalization(self.conv2d(h_conv2, self.conv_channels[2], 'same'),
                                              axis=3, training=self.ifTrain))
            # batch_size * board_side * board_side * conv_channels[2]

            h_conv3_flat = tf.reshape(h_conv3, [-1, self.conv_channels[2]*board_side*board_side])

            h_fc1 = dropout(relu(BatchNormalization(dense(h_conv3_flat, self.fc_neural[0]), axis=1,
                                                    training=self.ifTrain)), keep_prob=1-self.dropout_prob)
            # batch_size * fc_neural[0]
            h_fc2 = dropout(relu(BatchNormalization(dense(h_fc1, self.fc_neural[1]), axis=1,
                                                    training=self.ifTrain)), keep_prob=1-self.dropout_prob)
            # batch_size * fc_neural[1]

            # output
            self.pi_out = dense(h_fc2, self.action_num)
            self.pi_prob = tf.nn.softmax(self.pi_out)
            # batch_size * self.action_num

            self.v_out = tanh(dense(h_fc2, 1))
            # batch_size x 1

            self.calculate_loss()

    def conv2d(self, input, out_channels, padding):
        """
        The convolution kernel is 3*3.
        """
        return tf.layers.conv2d(input, out_channels, kernel_size=self.conv_kernel, padding=padding)

    def calculate_loss(self):
        """
        Compute the loss of MCTS's neural network and update the network.
        """
        self.pi = tf.placeholder(tf.float32, shape=[None, self.action_num])
        self.v = tf.placeholder(tf.float32, shape=[None])

        self.loss_pi = tf.losses.softmax_cross_entropy(self.pi, self.pi_out)
        self.loss_v = tf.losses.mean_squared_error(self.v, tf.reshape(self.v_out, shape=[-1, ]))

        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

        return None

