import argparse
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import pickle
import gzip

from generators import *
from utilities import *
from models import *

import gc

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

def predict_val(args, batch_size=32, wdir=None, tta=1):
    if args.folds:
        if args.fold_type == 'my':
            print('Loading my kfold ...')
            folds = pickle.load(gzip.open(TRAIN_MODIFIED_DIR + MY_KFOLD_FILENAME, 'rb'))
        else:
            print('Loading mutual kfold ...')
            folds = pickle.load(gzip.open(TRAIN_MODIFIED_DIR + KFOLD_FILENAME, 'rb'))
    else:
        if args.loader == 'new':
            trainset, valset, hoset = load_train_val_data_new()
        else:
            trainset, valset, hoset = load_train_val_data()

    model_name = args.model
    model_f = get_model_f(model_name)
    # default win_size and win_stride
    shape = (129, 124, 1)

    if args.resize:
        shape = (args.resize_w, args.resize_h, 1)

    settings = prepare_settings(args)
    audio_transformer = AudioTransformer(settings)

    if args.folds:
        folds_acc = []
        folds_mpc_acc = []
        folds_class_acc = []
        folds_class_total = []
        for i in range(args.start_fold, len(folds)):
            if args.my_noise:
                trainset, valset = load_fold_my_noise(folds[i])
            else:
                trainset, valset = load_fold(folds[i])

            valfiles = [v[2] for v in valset]
            labels = [v[0] for v in valset]
            y_val = to_categorical(labels, num_classes = len(LABELS))
            
            model = get_model(model_f, shape)
            preds_k = []
            for k in range(args.k_best):
                fold_wdir = load_k_best_weights_min(model, model_name, k=k, wdir=wdir, fold=i)
                preds = model.predict_generator(test_generator(valfiles, batch_size, audio_transformer, tta=1), int(np.ceil(len(valfiles) / batch_size)), verbose=1)
                preds_k.append(preds)
            preds_k = np.mean(preds_k, axis=0)
            p_labels = np.argmax(preds_k, axis=1)

            conf = confusion_matrix(labels, p_labels)
            class_acc = []
            class_total = []
            for l in LABELS:
                total = sum(conf[NAME2ID[l]])
                acc = 1.0 * conf[NAME2ID[l], NAME2ID[l]] / total
                class_total.append(total)
                class_acc.append(acc)
            
            print('Fold ', i)
            print('loss = {}, acc = {}'.format(log_loss(y_val, preds_k), accuracy_score(labels, p_labels)))
            print('mean per class accuracy: {}'.format(np.mean(class_acc)))
            print('----------------------------------------------------')

            folds_class_acc.append(class_acc)
            folds_class_total.append(class_total)
            folds_acc.append(accuracy_score(labels, p_labels))
            folds_mpc_acc.append(np.mean(class_acc))

            fnames = [f.split('\\')[-2] + '_' + f.split('\\')[-1] if len(f.split('\\')) > 2 else f.split('/')[-2] + '_' + f.split('/')[-1] for f in valfiles]
            dump_preds(preds_k, 'train', dump_dir=PREDS_DIR + args.preds_file + '/fold_{}'.format(i), fnames=fnames)

        folds_class_acc = np.mean(folds_class_acc, axis=0)
        folds_class_total = np.sum(folds_class_total, axis=0)
        for l in LABELS:
            total = folds_class_total[NAME2ID[l]]
            acc = folds_class_acc[NAME2ID[l]]
            class_acc.append(acc)
            print('{}, total: {}, accuracy: {}'.format(l, total,  acc))

        print('MEAN acc = {}, mpc acc = {}'.format(np.mean(folds_acc), np.mean(folds_mpc_acc)))

def predict(args, batch_size=32, wdir=None, tta=1):
    model_name = args.model
    # default win_size and win_stride
    shape = (129, 124, 1)

    if args.resize:
        shape = (args.resize_w, args.resize_h, 1)

    model_f = get_model_f(model_name)

    files = glob.glob(os.path.join(TEST_DIR, '*.wav'))

    settings = prepare_settings(args)
    audio_transformer = AudioTransformer(settings)

    if args.folds:
        folds = pickle.load(gzip.open(TRAIN_MODIFIED_DIR + KFOLD_FILENAME, 'rb'))
        preds_folds = []
        for i in range(args.start_fold, len(folds)):

            preds_k = []
            for k in range(args.k_best):
                model = get_model(model_f, shape)
                fold_wdir = load_k_best_weights_min(model, model_name, k=k, wdir=wdir, fold=i)

                preds = model.predict_generator(test_generator(files, batch_size, audio_transformer, tta), int(np.ceil(len(files) / (batch_size / tta))), verbose=1)
                preds_k.append(preds)

            preds_folds.append(np.mean(preds_k, axis=0))

            dump_preds(preds, 'test', dump_dir=PREDS_DIR + args.preds_file + '/fold_{}'.format(i), fnames=[f.split('\\')[-1] for f in files])
        
        mean_preds = np.mean(preds_folds, axis=0)
        mean_labels = np.argmax(mean_preds, axis=1)
        dump_preds(mean_preds, args.preds_file, fnames=[f.split('\\')[-1] for f in files])
        make_submission(files, mean_labels, args.out_file)

def make_submission(files, labels, out_file):
    assert len(files) == labels.shape[0]
    files = [f.split('\\')[-1] for f in files]
    with open(os.path.join(OUTPUT_DIR, '{}.csv'.format(out_file)), 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in zip(files, labels):
            fout.write('{},{}\n'.format(fname, ID2NAME[label]))

if __name__ == '__main__':
    set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument('--tta', type=int, default=1, help='number of test time augmentations. if 1 - only defult image is used')
    parser.add_argument('--out_file', type=str, default='out', help='submission file')
    parser.add_argument('--preds_file', type=str, default='preds', help='raw predictions file')
    parser.add_argument('--model', type=str, default='baseline', help='model name')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--wdir', type=str, default=None, help='weights dir, if None - load by model_name')
    parser.add_argument("-r", "--resize", action="store_true", help="resize image if True")
    parser.add_argument("--resize_w", type=int, default=199, help="resize width")
    parser.add_argument("--resize_h", type=int, default=199, help="resize height")
    parser.add_argument("--loader", type=str, default="old", help="data loader type")
    parser.add_argument("--spect", type=str, default="scipy", help="spectrogram type")

    parser.add_argument("-f", "--folds", action="store_true", help="folds if True")
    parser.add_argument("--fold_type", type=str, default="mutual", help="fold type: mutual / my")
    parser.add_argument("--start_fold", type=int, default=0, help="start fold")

    parser.add_argument("--my_noise", action="store_true", help="use my noise, ignore other")

    parser.add_argument("-v", "--val", action="store_true", help="predict on val")
    parser.add_argument("-t", "--test", action="store_true", help="predict on test")

    parser.add_argument("--k_best", type=int, default=4, help="use k best epochs")

    parser.add_argument('--win_size', type=int, default=256, help='stft window size')
    parser.add_argument('--win_stride', type=int, default=128, help='stft window stride')

    parser.add_argument('--time_shift_p', type=float, default=0, help='time shift augmentation probability')
    parser.add_argument('--speed_tune_p', type=float, default=0, help='speed tune augmentation probability')
    parser.add_argument('--mix_with_bg_p', type=float, default=0, help='mix with background augmentation probability')
    args = parser.parse_args()

    if args.val:
        predict_val(args, batch_size=args.batch, wdir=args.wdir, tta=args.tta)
    if args.test:
        predict(args, batch_size=args.batch, wdir=args.wdir, tta=args.tta)

    gc.collect()
