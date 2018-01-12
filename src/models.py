import glob
import os

import tensorflow as tf
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

from keras.applications import *
from utilities import LABELS

OPTS = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}

def load_best_weights_min(model, model_name, wdir=None, fold=None):
    if wdir is None:
        wdir = str(model_name) + '/'
    if fold is not None:
        wdir += '/fold{}/'.format(fold)

    print('looking for weights in {}'.format(wdir))
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    elif len(os.listdir(wdir)) > 0:
        print(os.listdir(wdir))
        wf = sorted(glob.glob(os.path.join(wdir, '*.h5')))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)
    return wdir

def load_k_best_weights_min(model, model_name, k=0, wdir=None, fold=None):
    if wdir is None:
        wdir = str(model_name) + '/'
    if fold is not None:
        wdir += '/fold{}/'.format(fold)

    print('looking for weights in {}'.format(wdir))
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    elif len(os.listdir(wdir)) > 0:
        print(os.listdir(wdir))
        wf = sorted(glob.glob(os.path.join(wdir, '*.h5')))[k]
        model.load_weights(wf)
        print('loaded weights file: ', wf)
    return wdir

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
    return top_and_dense(ResNet50, shape, opt, dropout_val)

def inception_resnet(shape, opt, dropout_val=0.5):
    return top_and_dense(InceptionResNetV2, shape, opt, dropout_val)

def vgg16(shape, opt, dropout_val=0.5):
    return top_and_dense(VGG16, shape, opt, dropout_val)

def vgg19(shape, opt, dropout_val=0.5):
    return top_and_dense(VGG19, shape, opt, dropout_val)

def xception(shape, opt, dropout_val=0.5):
    return top_and_dense(Xception, shape, opt, dropout_val)

def inception(shape, opt, dropout_val=0.5):
    return top_and_dense(InceptionV3, shape, opt, dropout_val)

def nasnet(shape, opt, dropout_val=0.5):
    return top_and_dense(NASNetLarge, shape, opt, dropout_val)

def get_model_f(model_name):
    fmodels = {'resnet50': resnet50, 'vgg16': vgg16, 'vgg19': vgg19,
        'inception_resnet': inception_resnet, 'xception': xception, 'inception': inception, 'nasnet': nasnet}
    return fmodels[model_name]

def get_model(model_f, shape, opt='adam'):
    return model_f(shape, opt, dropout_val=0.0)