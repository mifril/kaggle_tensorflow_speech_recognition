import argparse
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

from generators import *
from utilities import *
from models import *

import gc

from sklearn.metrics import log_loss, accuracy_score

def mean(files, out_file, weights=None, p_rate=0.0):
    preds = np.array([pd.read_csv(os.path.join(PREDS_DIR, f + '.csv'), index_col='fname').values for f in files])
    print(preds[0].shape)
    # print(preds[0, 2:3])
    # print(preds[1, 2:3])
    if weights is None:
        preds = np.mean(preds, axis=0)
    else:
        preds = np.average(preds, weights=weights, axis=0)

    files = glob.glob(os.path.join(TEST_DIR, '*.wav'))
    files = [f.split('\\')[-1] for f in files]
    pred_classes = np.argmax(preds, axis=1)

    if p_rate > 0.0:
        for l in range(len(LABELS) - 1):
            cond = np.logical_and(pred_classes == 11, preds[:,l] > p_rate)
            preds[np.where(cond)[0], 11] = 0.0
        pred_classes = np.argmax(preds, axis=1)
        # for l in range(len(LABELS) - 1):
        #     classes[np.logical_and(classes == 11, preds[:,l] > p_rate)] = l

    dump_preds(preds, 'mean_' + out_file, files)

    with open(os.path.join(OUTPUT_DIR, '{}.csv'.format(out_file)), 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in zip(files, pred_classes):
            fout.write('{},{}\n'.format(fname, ID2NAME[label]))

def search_alpha_itr(preds_1, preds_2, y_val, test_on_acc=False):
    best_alpha = 0.0
    best_loss = 10.0
    best_acc = 0.0
    for alpha in np.arange(0, 1.01, 0.01):
        preds = alpha * preds_1 + (1 - alpha) * preds_2
        loss = log_loss(y_val, preds)
        acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(preds, axis=1))
        if test_on_acc:
            if acc > best_acc:
                best_loss = loss
                best_alpha = alpha
                best_acc = acc
                # print('val log_loss {:.5}; accuracy: {:.5}; alpha: {:.5}'.format(loss, acc, alpha))
        else:
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
                best_acc = acc
                # print('val log_loss {:.5}; accuracy: {:.5}; alpha: {:.5}'.format(loss, acc, alpha))
    # print('val best log_loss {:.5}; accuracy: {:.5}; best_alpha: {:.5}'.format(best_loss, best_acc, best_alpha))
    return best_alpha

def search_alphas(model_names, shapes, wdirs, batches, ttas, settings, test_on_acc=False):
    alphas = np.ones(len(model_names))

    models = []
    for i in range(len(model_names)):
        model_f = get_model_f(model_names[i])
        opt = 'adam'
        models.append(get_model(model_f, shapes[i], opt))
        load_best_weights_min(models[-1], model_names[i], wdir=wdirs[i])

    trainset, valset, hoset = load_train_val_data()
    if hoset is not None:
        valset = hoset
    labels = [v[0] for v in valset]
    y_val = to_categorical(labels, num_classes = len(LABELS))
    files = [v[2] for v in valset]

    audio_transformers = [AudioTransformer(settings[i]) for i in range(len(settings))]
    models_preds = [models[i].predict_generator(test_generator(files, batches[i], audio_transformers[i], ttas[i]), int(np.ceil(len(files) / (batches[i] / ttas[i]))), verbose=1)
                        for i in range(len(models))]

    for i in range(len(models_preds)):
        print('{} : loss = {}, acc = {}'.format(model_names[i], log_loss(y_val, models_preds[i]), accuracy_score(np.argmax(y_val, axis=1), np.argmax(models_preds[i], axis=1))))

    preds_1 = models_preds[0]
    for i in range(1, len(models)):
        preds_2 = models_preds[i]
        alpha = search_alpha_itr(preds_1, preds_2, y_val, test_on_acc)
        for j in range(i):
            alphas[j] *= alpha
        alphas[i] *= (1 - alpha)

        preds_1 = alpha * preds_1 + (1 - alpha) * preds_2

    preds = np.average(models_preds, weights=alphas, axis=0)
    preds_mean = np.mean(models_preds, axis=0)
    print('val log_loss {}; accuracy: {}, alphas: {}'.format(log_loss(y_val, preds), accuracy_score(np.argmax(y_val, axis=1), np.argmax(preds, axis=1)), alphas))
    print('MEAN: val log_loss {}; accuracy: {}'.format(log_loss(y_val, preds_mean), accuracy_score(np.argmax(y_val, axis=1), np.argmax(preds_mean, axis=1))))

    return alphas

if __name__ == '__main__':
    set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str, default='out', help='submission file')
    parser.add_argument('-w', '--weighted', action='store_true', help='if True - weighted mean')
    parser.add_argument('-a', '--acc', action='store_true', help='if True - choose alpha testing on accuracy score')
    parser.add_argument("--p_rate", type=float, default=0.0, help="learning rate reduce rate")
    args = parser.parse_args()

    if args.weighted:
        model_names = ['vgg16', 'vgg16', 'resnet50', 'resnet50', 'inception_resnet']
        shapes = [(241, 49, 1), (129, 124, 1), (199, 199, 1), (224, 224, 1), (140, 140, 1)]
        wdirs = ['vgg16_480_160', 'vgg16', 'resnet50', 'resnet50_224', 'incres_140']
        batches = [32, 32, 16, 16, 16]
        ttas = [1, 1, 1, 1, 1]

        settings_vgg16_480_160 = dict()
        settings_vgg16_480_160['win_size'] = 480
        settings_vgg16_480_160['win_stride'] = 160
        settings_vgg16_480_160['resize'] = False
        settings_vgg16_480_160['resize_w'] = 199
        settings_vgg16_480_160['resize_h'] = 199
        settings_vgg16_480_160['time_shift_p'] = 0.8
        settings_vgg16_480_160['speed_tune_p'] = 0.8
        settings_vgg16_480_160['mix_with_bg_p'] = 0.8

        settings_vgg16 = settings_vgg16_480_160.copy()
        settings_vgg16['win_size'] = 256
        settings_vgg16['win_stride'] = 128

        settings_resnet50 = settings_vgg16.copy()
        settings_resnet50['resize'] = True

        settings_resnet50_224 = settings_resnet50.copy()
        settings_resnet50_224['resize_w'] = 224
        settings_resnet50_224['resize_h'] = 224

        settings_incres = settings_resnet50.copy()
        settings_incres['resize_w'] = 140
        settings_incres['resize_h'] = 140

        settings = [settings_vgg16_480_160, settings_vgg16, settings_resnet50, settings_resnet50_224, settings_incres]

        test_on_acc = args.acc
        weights = search_alphas(model_names, shapes, wdirs, batches, ttas, settings, test_on_acc)
    else:
        weights = None
    
    # pred_files = ['vgg16_480_160', 'vgg16_256_128', 'resnet50', 'resnet50_224', 'incres_140', 'incres_199',
    #           'xception', 'inception', 'resnet50_fixed', 'resnet50_librosa']
    pred_files = ['vgg16_my_my_n', 'inc_139_my_my_n', 'xcep_my_my_n', 'xcep_my_my_n_kbest']
    weights = [2, 2, 1, 1]
    mean(pred_files, args.out_file, weights, args.p_rate)
