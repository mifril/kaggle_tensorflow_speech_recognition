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

def dump_preds(preds, preds_file):
    if not os.path.exists(PREDS_DIR):
        os.makedirs(PREDS_DIR)
    pd.DataFrame(preds).to_csv(os.path.join(PREDS_DIR, preds_file + '.csv'), index=False, header=False)

def predict(args, batch_size=32, wdir=None, tta=1):
    model_name = args.model
    # default win_size and win_stride
    shape = (129, 124, 1)
    # 480_160
    # shape = (241, 49, 1)

    # resnet50
    if args.resize:
        shape = (args.resize_w, args.resize_h, 1)

    model_f = get_model_f(model_name)
    model = get_model(model_f, shape)
    load_best_weights_min(model, model_name, wdir=wdir)

    files = glob.glob(os.path.join(TEST_DIR, '*.wav'))

    settings = prepare_settings(args)
    audio_transformer = AudioTransformer(settings)
    preds = model.predict_generator(test_generator(files, batch_size, audio_transformer, tta), int(np.ceil(len(files) / (batch_size / tta))), verbose=1)
    
    if tta > 1:
        preds = preds.reshape((len(files), tta, preds.shape[1]))
        preds = preds.mean(axis=1)
    
    print(preds.shape)
    classes = np.argmax(preds, axis=1)
    dump_preds(preds, args.preds_file)

    assert len(files) == classes.shape[0]

    files = [f.split('\\')[-1] for f in files]

    with open(os.path.join(OUTPUT_DIR, '{}.csv'.format(args.out_file)), 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in zip(files, classes):
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

    parser.add_argument('--win_size', type=int, default=256, help='stft window size')
    parser.add_argument('--win_stride', type=int, default=128, help='stft window stride')

    parser.add_argument('--time_shift_p', type=float, default=0, help='time shift augmentation probability')
    parser.add_argument('--speed_tune_p', type=float, default=0, help='speed tune augmentation probability')
    parser.add_argument('--mix_with_bg_p', type=float, default=0, help='mix with background augmentation probability')
    args = parser.parse_args()

    predict(args, batch_size=args.batch, wdir=args.wdir, tta=args.tta)

    gc.collect()
