import os

import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, LeakyReLU, Concatenate)
from keras.models import Model
from keras.regularizers import l2

from define import DN_FILTERS, DN_INPUT_SHAPE, DN_OUTPUT_SIZE, DN_RESIDUAL_NUM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * keras.backend.square(error)
    linear_loss = clip_delta * (keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def conv(filters):
    return Conv2D(
        filters,
        3,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(0.0005))


def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        # x = Activation('relu')(x)
        x = LeakyReLU(alpha=0.01)(x)
        return x
    return f


def dual_network():
    input = Input(shape=DN_INPUT_SHAPE)

    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = LeakyReLU(alpha=0.01)(x)

    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    p = Conv2D(
        2,
        1,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(0.0005))(x)
    p = BatchNormalization()(p)
    # p = Activation('relu')(p)
    x = LeakyReLU(alpha=0.01)(x)

    p = Flatten()(p)
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(
        0.0005), activation='softmax', name='pi')(p)

    v = Conv2D(
        1,
        1,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(0.0005))(x)
    v = BatchNormalization()(v)
    # v = Activation('relu')(v)
    x = LeakyReLU(alpha=0.01)(x)
    v = Flatten()(v)
    v = Dense(256, kernel_regularizer=l2(0.0005))(v)
    v = Activation('relu')(v)
    v = Dense(1, kernel_regularizer=l2(0.0005))(v)
    v = Activation('linear', name='v')(v)

    model = Model(inputs=input, outputs=[p, v])
    model.summary()

    # K.set_learning_phase(0)

    model.save('C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources/best.h5')

    K.clear_session()
    del model

    # x = GlobalAveragePooling2D()(x)

    # p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(
    #     0.0005), activation='softmax', name='pi')(x)

    # v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    # v = Activation('linear', name='v')(v)

    # model = Model(inputs=input, outputs=[p, v])
    # model.summary()
    # model.save('./model/best.h5')
    # K.clear_session()
    # del model


if __name__ == '__main__':
    dual_network()
