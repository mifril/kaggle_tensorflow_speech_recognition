import os
import glob
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import *
import re
import pandas as pd
import gc
import cv2

import tensorflow as tf
from keras import backend as K
import random as rn

INPUT_DIR = '..//input//'
OUTPUT_DIR = '..//output//'
MAIN_DIR = 'C://data//tf_speech//'
TRAIN_DIR = MAIN_DIR + 'train//'
TRAIN_AUDIO_DIR = TRAIN_DIR + 'audio//'
BG_DIR = TRAIN_DIR + 'audio//_background_noise_//'
TEST_DIR = MAIN_DIR + 'test//audio//'
PREDS_DIR = '..//output//preds//'

LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
ID2NAME = {i: name for i, name in enumerate(LABELS)}
NAME2ID = {name: i for i, name in ID2NAME.items()}

LENGTH = 16000

RS = 17
EPS = 1e-12

SILENCE_PERCENT = .09

def set_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(RS)
    rn.seed(RS + 1000)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(RS + 2000)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

def load_train_val_data():
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    pattern = re.compile("C://(.+\\)?(\w+)\\([^_]+)_.+wav")
    all_files = glob.glob(os.path.join(TRAIN_AUDIO_DIR, '*/*wav'))

    speakers = []
    for fname in all_files:
        splits = fname.split('\\')
        if splits[-2] != '_background_noise_':
            speakers.append(splits[-1].split('_')[0])
    speakers = np.unique(speakers)
    train_speakers, val_speakers = train_test_split(speakers, test_size=0.3, random_state=RS)
    val_speakers, test_speakers = train_test_split(val_speakers, test_size=0.33, random_state=RS)
    
    train, val, test = [], [], []
    noise = []
    for fname in all_files:
        splits = fname.split('\\')
        label, uid = splits[-2], splits[-1].split('_')[0]
        if label == '_background_noise_':
            label = 'silence'
        if label not in LABELS:
            label = 'unknown'

        label_id = NAME2ID[label]

        sample = (label_id, uid, fname)
        if uid in val_speakers:
            val.append(sample)
        elif uid in test_speakers:
            test.append(sample)
        elif label == 'silence':
            noise.append(sample)
        else:
            train.append(sample)

    print('There are {} train, {} val, {} test samples'.format(len(train), len(val), len(test)))

    noise_type_percent = (SILENCE_PERCENT / (1 - SILENCE_PERCENT)) / len(noise)
    n_noise_type_train = int(noise_type_percent * len(train))
    n_noise_type_val = int(noise_type_percent * len(val))
    n_noise_type_test = int(noise_type_percent * len(test))

    for sample in noise:
        for i in range(n_noise_type_train):
            train.append(sample)
        for i in range(n_noise_type_val):
            val.append(sample)
        for i in range(n_noise_type_test):
            test.append(sample)

    print('There are {} train, {} val, {} test samples after adding SILENCE'.format(len(train), len(val), len(test)))
    
    train_labels = np.array([s[0] for s in train])
    val_labels = np.array([s[0] for s in val])
    test_labels = np.array([s[0] for s in val])
    train_parts = [len(train_labels[train_labels == l]) / len(train_labels) for l in range(len(LABELS))]
#     val_parts = [len(val_labels[val_labels == l]) / len(val_labels) for l in range(len(LABELS))]
#     test_parts = [len(test_labels[test_labels == l]) / len(test_labels) for l in range(len(LABELS))]
    print(train_parts)
    return train, val, test

# https://www.kaggle.com/alexozerin/end-to-end-baseline-tf-estimator-lb-0-72
def load_train_val_data_no_unknown_in_val():
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("C://(.+\\)?(\w+)\\([^_]+)_.+wav")
    all_files = glob.glob(os.path.join(TRAIN_AUDIO_DIR, '*/*wav'))

    with open(os.path.join(TRAIN_DIR, 'validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for fname in validation_files:
        splits = fname.split('\\')
        valset.add(splits[-1].rstrip())

    train, val = [], []
    noise = [(NAME2ID['silence'], '', '')]
    for fname in all_files:
        splits = fname.split('\\')
        label, uid = splits[-2], splits[-1]
        if label == '_background_noise_':
            label = 'silence'
        if label not in LABELS:
            label = 'unknown'

        label_id = NAME2ID[label]

        sample = (label_id, uid, fname)
        # print (uid)
        if label + '/' + uid in valset:
            val.append(sample)
        elif label == 'silence':
            noise.append(sample)
        else:
            train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))

    noise_type_percent = (SILENCE_PERCENT / (1 - SILENCE_PERCENT)) / (len(noise) + 1)
    n_noise_type_train = int(noise_type_percent * len(train))
    n_noise_type_val = int(noise_type_percent * len(val))

    for sample in noise:
        for i in range(n_noise_type_train):
            train.append(sample)
        for i in range(n_noise_type_val):
            val.append(sample)

    print('There are {} train and {} val samples after adding SILENCE'.format(len(train), len(val)))
    return train, val, None

def prepare_settings(args, resize=None):
    settings = dict()
    settings['win_size'] = args.win_size
    settings['win_stride'] = args.win_stride
    settings['resize'] = args.resize
    settings['resize_w'] = args.resize_w
    settings['resize_h'] = args.resize_h
    settings['time_shift_p'] = args.time_shift_p
    settings['speed_tune_p'] = args.speed_tune_p
    settings['mix_with_bg_p'] = args.mix_with_bg_p
    return settings

class AudioTransformer:
    def __init__(self, settings):
        self.win_size = settings['win_size']
        self.win_stride = settings['win_stride']
        self.resize = settings['resize']
        self.resize_shape = (settings['resize_w'], settings['resize_h'])
        self.time_shift_p = settings['time_shift_p']
        self.speed_tune_p = settings['speed_tune_p']
        self.mix_with_bg_p = settings['mix_with_bg_p']
        self.bg_noises = self.load_bg_noises()

    def load_bg_noises(self):
        bg_files = os.listdir(BG_DIR)
        bg_files.remove('README.md')
        bg_wavs = []
        for bg_file in bg_files:
            _, wav = wavfile.read(os.path.join(BG_DIR, bg_file))
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            bg_wavs.append(wav)
        return bg_wavs

    def time_shift(self, wav):
        shift_range = .1 * LENGTH
        start_ = int(np.random.uniform(-shift_range, shift_range))
        if start_ >= 0:
            wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001, 0.001, start_)]
        else:
            wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), wav[:start_]]
        return wav_time_shift

    def speed_tune(self, wav):
        speed_rate = np.random.uniform(0.7, 1.3)
        wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
        if len(wav_speed_tune) < LENGTH:
            pad_len = LENGTH - len(wav_speed_tune)
            wav_speed_tune = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len/2)),
                                   wav_speed_tune,
                                   np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len/2)))]
        else: 
            cut_len = len(wav_speed_tune) - LENGTH
            wav_speed_tune = wav_speed_tune[int(cut_len / 2) : int(cut_len / 2) + LENGTH]
        return wav_speed_tune

    def mix_with_bg(self, wav):
        bg = self.bg_noises[np.random.randint(len(self.bg_noises))]

        start_ = np.random.randint(bg.shape[0] - LENGTH)
        bg_slice = bg[start_ : start_ + LENGTH]
        wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.1)

        return wav_with_bg

    def apply_augmentation(self, wav, label_id):
        if np.random.random() < self.time_shift_p:
            wav = self.time_shift(wav)
        if np.random.random() < self.speed_tune_p:
            wav = self.speed_tune(wav)
        if np.random.random() < self.mix_with_bg_p and ID2NAME[label_id] != 'silence':
            wav = self.mix_with_bg(wav)
        return wav

    def pad_audio(self, samples):
        if len(samples) >= LENGTH:
            return samples
        else:
            return np.pad(samples, pad_width=(LENGTH - len(samples), 0), mode='constant', constant_values=(0, 0))

    def process_wav_file(self, fname, label_id=None, mode='train', normalize=True):
        if fname == '':
            wav = np.zeros(shape=(LENGTH,))
        else:
            _, wav = wavfile.read(fname)
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max

        if len(wav) > LENGTH:
            i = np.random.randint(0, len(wav) - LENGTH)
            wav = wav[i : i + LENGTH]
        elif len(wav) < LENGTH:
            wav = self.pad_audio(wav)

        if mode == 'train' and fname != '':
            wav = self.apply_augmentation(wav, label_id)

        # specgram = stft(wav, LENGTH, nperseg=self.win_size, noverlap=self.win_stride, nfft=self.win_size, padded=False, boundary=None)
        # phase = np.angle(specgram[2]) / np.pi
        # amp = np.log1p(np.abs(specgram[2]))
        # spect = np.stack([phase, amp], axis=2)

        specgram = spectrogram(wav, LENGTH, nperseg=self.win_size, noverlap=self.win_stride, nfft=self.win_size)
        spect = np.log(specgram[2].astype(np.float32) + EPS)
        if normalize:
            mean = spect.mean()
            std = spect.std()
            spect = (spect - mean) / (std + EPS)

        if self.resize:
            spect = cv2.resize(spect, self.resize_shape)

        return spect
