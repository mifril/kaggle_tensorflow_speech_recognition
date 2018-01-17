import numpy as np

from utilities import *
from keras.utils import to_categorical

def data_generator(data, batch_size, audio_transformer, mode='train'):
    if mode == 'train':
        np.random.shuffle(data)
    while True:
        cur = 0
        x_batch = []
        y_batch = []

        for (label_id, uid, fname) in data:
            if cur > 0 and cur % batch_size == 0:
                x_batch = np.array(x_batch)
                x_batch = x_batch.reshape(tuple(list(x_batch.shape) + [1]))
                y_batch = to_categorical(y_batch, num_classes = len(LABELS))
                yield x_batch, y_batch
                
                x_batch = []
                y_batch = []
                cur = 0
            
            x_batch.append(audio_transformer.process_wav_file(fname, label_id, mode, normalize=True))
            y_batch.append(label_id)
            cur += 1
        x_batch = np.array(x_batch)
        x_batch = x_batch.reshape(tuple(list(x_batch.shape) + [1]))
        y_batch = to_categorical(y_batch, num_classes = len(LABELS))
        yield x_batch, y_batch

def data_generator_ram(data, batch_size, audio_transformer, mode='train'):
    if mode == 'train':
        np.random.shuffle(data)
    X = []
    y = []
    fnames = []
    for (label_id, uid, fname) in data:
        X.append(audio_transformer.load_wav(fname))
        y.append(label_id)
        fnames.append(fname)

    while True:
        cur = 0
        x_batch = []
        y_batch = []

        for wav, label_id, fname in zip(X, y, fnames):
            if cur > 0 and cur % batch_size == 0:
                x_batch = np.array(x_batch)
                x_batch = x_batch.reshape(tuple(list(x_batch.shape) + [1]))
                y_batch = to_categorical(y_batch, num_classes = len(LABELS))
                yield x_batch, y_batch
                
                x_batch = []
                y_batch = []
                cur = 0
            
            x_batch.append(audio_transformer.process_wav(wav, fname, label_id, mode, normalize=True))
            y_batch.append(label_id)
            cur += 1
        x_batch = np.array(x_batch)
        x_batch = x_batch.reshape(tuple(list(x_batch.shape) + [1]))
        y_batch = to_categorical(y_batch, num_classes = len(LABELS))
        yield x_batch, y_batch

def test_generator(files, batch_size, audio_transformer, tta=1):
    if tta < batch_size:
        for start in range(0, len(files), int(batch_size / tta)):
            x_batch = []
            end = min(start + int(batch_size / tta), len(files))
            batch_files = files[start:end]
            for f in batch_files:
                x_batch.append(audio_transformer.process_wav_file(f, mode='test', normalize=True))
                for i in range(tta - 1):
                    print('here')
                    x_batch.append(audio_transformer.process_wav_file(f, mode='test_tta', normalize=True))

            x_batch = np.array(x_batch)
            x_batch = x_batch.reshape(tuple(list(x_batch.shape) + [1]))
            yield x_batch
    else:
        for f in files:
            x_batch = []
            x_batch.append(audio_transformer.process_wav_file(f, mode='test', normalize=True))
            for i in range(tta - 1):
                x_batch.append(audio_transformer.process_wav_file(f, mode='test_tta', normalize=True))

            x_batch = np.array(x_batch)
            x_batch = x_batch.reshape(tuple(list(x_batch.shape) + [1]))
            yield x_batch[:batch_size]
            x_batch = x_batch[batch_size:]
