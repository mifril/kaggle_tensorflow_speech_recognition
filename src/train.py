import argparse
import numpy as np
from tqdm import tqdm

from generators import *
from utilities import *
from models import *

from tensorflow.python.keras.callbacks import *

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

import gc


def train(model_name, args, n_epochs=10000, batch_size=32, patience=5, reduce_rate=0.5, wdir=None):   
    if args.loader == 'new':
        trainset, valset, hoset = load_train_val_data_new()
    else:
        trainset, valset, hoset = load_train_val_data()
    n_train = len(trainset)
    n_val = len(valset)

    model_f = get_model_f(model_name)
    
    if args.resize:
        shape = (args.resize_w, args.resize_h, 1)
    else:
        # shape = (257, 98, 2)
        # shape = (161, 99, 2)
        
        # default win_size and win_stride 
        shape = (129, 124, 1)
        # win_size and win_stride  480_160
        # shape = (241, 49, 1)
    

    opt = OPTS[args.opt](args.start_lr)
    model = get_model(model_f, shape, opt)

    wdir = load_best_weights_min(model, model_name, wdir=wdir)

    callbacks = [
        EarlyStopping(monitor='val_loss',
            patience=patience * 2,
            verbose=1,
            min_delta=1e-6,
            mode='min'),
        ReduceLROnPlateau(monitor='val_loss',
            factor=reduce_rate,
            patience=patience,
            verbose=1,
            epsilon=1e-6,
            mode='min'),
        ModelCheckpoint(monitor='val_loss',
            filepath=wdir + '/{val_loss:.6f}-{val_acc:.6f}-{epoch:03d}.h5',
            save_best_only=True,
            save_weights_only=True,
            mode='min')
        ]

    print(n_train, n_val)
    settings = prepare_settings(args)
    audio_transformer = AudioTransformer(settings)
    model.fit_generator(generator=data_generator(trainset, batch_size, audio_transformer, mode='train'),
        steps_per_epoch=int(np.ceil(n_train / float(batch_size))),
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=data_generator(valset, batch_size, audio_transformer, mode='val'),
        validation_steps=int(np.ceil(n_val / float(batch_size))))

    if hoset is not None:
        hofiles = [v[2] for v in hoset]
        labels = [v[0] for v in valset]
        y_ho = to_categorical(labels, num_classes = len(LABELS))
        
        ho_preds = model.predict_generator(test_generator(hofiles, batch_size, audio_transformer, tta=1), int(np.ceil(len(hofiles) / batch_size)), verbose=1)
        print('HOLD OUT: loss = {}, acc = {}'.format(log_loss(y_ho, ho_preds), accuracy_score(labels, np.argmax(ho_preds[i], axis=1))))
        print(confusion_matrix(labels, np.argmax(ho_preds[i], axis=1)))

if __name__ == '__main__':
    # set_seeds()

    parser = argparse.ArgumentParser(description='Train TF-speech model')
    parser.add_argument("--start_lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--reduce_rate", type=float, default=0.1, help="learning rate reduce rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=5, help="LRSheduler patience (Early Stopping - 2 * patience)")
    parser.add_argument("--model", type=str, default='baseline', help="model to train")
    parser.add_argument("--opt", type=str, default='adam', help="model optimiser")
    parser.add_argument("--wdir", type=str, default=None, help="weights dir, if None - load by model_name")
    parser.add_argument("-r", "--resize", action="store_true", help="resize image if True")
    parser.add_argument("--resize_w", type=int, default=199, help="resize width")
    parser.add_argument("--resize_h", type=int, default=199, help="resize height")
    parser.add_argument("--loader", type=str, default="old", help="data loader type")
    parser.add_argument("--spect", type=str, default="scipy", help="spectrogram type")

    parser.add_argument("--win_size", type=int, default=256, help="stft window size")
    parser.add_argument("--win_stride", type=int, default=128, help="stft window stride")

    parser.add_argument("--time_shift_p", type=float, default=.8, help="time shift augmentation probability")
    parser.add_argument("--speed_tune_p", type=float, default=.8, help="speed tune augmentation probability")
    parser.add_argument("--mix_with_bg_p", type=float, default=.8, help="mix with background augmentation probability")
    args = parser.parse_args()

    train(args.model, args, n_epochs=10000, batch_size=args.batch, patience=args.patience, reduce_rate=args.reduce_rate, wdir=args.wdir)
    gc.collect()
