import glob
import os

import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

from utilities import LABELS

OPTS = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}

def load_best_weights_min(model, model_name, wdir=None):
    if wdir is None:
        wdir = str(model_name) + '/'

    print('looking for weights in {}'.format(wdir))
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    elif len(os.listdir(wdir)) > 0:
        print(os.listdir(wdir))
        wf = sorted(glob.glob(os.path.join(wdir, '*.h5')))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)
    return wdir

# shape = (257,98,2)
def baseline(shape, dropout_val=0.5, batch_norm=False, init_filters=16, n_convs=4):
    x_in = Input(shape=shape)
    if batch_norm:
        x = BatchNormalization()(x_in)
    for i in range(n_convs):
        x = Conv2D(init_filters * 2 ** i, (3, 3))(x)
        x = Activation('relu')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(init_filters * 2 ** n_convs, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(256, activation = 'relu')(x)
    if dropout_val > 0.0:
        x = Dropout(dropout_val)(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs = x_in, outputs = x)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def baseline(shape, opt, dropout_val=0.5, batch_norm=False, init_filters=16, n_convs=2):
    x_in = Input(shape=shape)
    if batch_norm:
        x = BatchNormalization()(x_in)
    for i in range(n_convs):
        x = Conv2D(init_filters * 2 ** i, (3, 3))(x)
        x = Activation('relu')(x)
        if dropout_val > 0.0:
            x = Dropout(dropout_val)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs = x_in, outputs = x)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def top_and_dense(top_model_f, shape, opt, dropout_val=0.5):
    top_model = top_model_f(include_top=False, weights=None, input_shape=shape)
    x = top_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = 'relu')(x)
    if dropout_val > 0.0:
        x = Dropout(dropout_val)(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs=top_model.inputs, outputs=x)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def resnet50(shape, opt, dropout_val=0.5):
    resnet_model = ResNet50(include_top=False, weights=None, input_shape=shape)
    x = resnet_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = 'relu')(x)
    if dropout_val > 0.0:
        x = Dropout(dropout_val)(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs = resnet_model.inputs, outputs = x)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def inception_resnet(shape, opt, dropout_val=0.5):
    top_model = InceptionResNetV2(include_top=False, weights=None, input_shape=shape)
    x = top_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = 'relu')(x)
    if dropout_val > 0.0:
        x = Dropout(dropout_val)(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs=top_model.inputs, outputs=x)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def vgg16(shape, opt, dropout_val=0.5):
    vgg_model = VGG16(include_top=False, weights=None, input_shape=shape)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = 'relu')(x)
    if dropout_val > 0.0:
        x = Dropout(dropout_val)(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs = vgg_model.inputs, outputs = x)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def vgg19(shape, opt, dropout_val=0.5):
    vgg_model = VGG19(include_top=False, weights=None, input_shape=shape)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = 'relu')(x)
    if dropout_val > 0.0:
        x = Dropout(dropout_val)(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs = vgg_model.inputs, outputs = x)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def xception(shape, opt, dropout_val=0.5):
    top_model = Xception(include_top=False, weights=None, input_shape=shape)
    x = top_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = 'relu')(x)
    if dropout_val > 0.0:
        x = Dropout(dropout_val)(x)
    x = Dense(len(LABELS), activation = 'softmax')(x)

    model = Model(inputs=top_model.inputs, outputs=x)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def inception(shape, opt, dropout_val=0.5):
    return top_and_dense(InceptionV3, shape, opt, dropout_val)
    
def get_model_f(model_name):
    fmodels = {'baseline': baseline, 'resnet50': resnet50, 'vgg16': vgg16, 'vgg19': vgg19,
        'inception_resnet': inception_resnet, 'xception': xception, 'inception': inception}
    return fmodels[model_name]

def get_model(model_f, shape, opt='adam'):
    if model_f is baseline:
        return model_f(shape, opt, dropout_val=0.5, batch_norm=True, init_filters=16, n_convs=4)
    else:
        return model_f(shape, opt, dropout_val=0.0)