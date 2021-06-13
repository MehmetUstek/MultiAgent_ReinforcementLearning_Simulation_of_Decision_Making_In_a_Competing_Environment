
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Concatenate
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSprop
from keras.optimizers import *
import tensorflow as tf
from keras import backend as K
import datetime
from tensorflow.python.keras.callbacks import TensorBoard

HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)


class Brain(object):

    def __init__(self, state_size, action_size, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.optimizer_model = arguments['optimizer']
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        # with strategy.scope():
        #     self.model = self._build_model()
        #     self.model_ = self._build_model()
        self.model = self._build_model()
        self.model_ = self._build_model()

    def _build_model(self):
        print("state_size",self.state_size)
        x = Input(shape=(self.state_size,))

        # a series of fully connected layer for estimating V(s)

        y11 = Dense(self.num_nodes, activation='relu')(x)
        y12 = Dense(self.num_nodes, activation='relu')(y11)
        y13 = Dense(1, activation="linear")(y12)

        # a series of fully connected layer for estimating A(s,a)

        y21 = Dense(self.num_nodes, activation='relu')(x)
        y22 = Dense(self.num_nodes, activation='relu')(y21)
        y23 = Dense(self.action_size, activation="linear")(y22)

        w = Concatenate(axis=-1)([y13, y23])

        # combine V(s) and A(s,a) to get Q(s,a)
        z = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(self.action_size,))(w)
        print(z)
        model = Model(inputs=x, outputs=z)
        model.summary()

        if self.optimizer_model == 'Adam':
            optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(learning_rate=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')

        model.compile(loss=huber_loss, optimizer=optimizer)

        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model.load_weights(self.weight_backup)

        return model

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):
        # print("x",len(x))# x is the input to the network and y is the output
        # log_dir = "D:\TF_log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.model_.predict(state)
        else:  # get prediction from local network
            return self.model.predict(state)

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1, self.state_size), target=target).flatten()

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save(self.weight_backup)