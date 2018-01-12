import numpy as np
import pandas as pd
import argparse

OUTPUT_DIR = '../output/'
PREDS_DIR = '../output/preds/'
LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
ID2NAME = {i: name for i, name in enumerate(LABELS)}
NAME2ID = {name: i for i, name in ID2NAME.items()}

def diff_labels(f_new, f_best='best', f_old=None):
    f_new = OUTPUT_DIR + f_new + '.csv'
    f_best = OUTPUT_DIR + f_best + '.csv'
    if f_old is not None:
        f_old = OUTPUT_DIR + f_old + '.csv'
    
    p_new = pd.read_csv(f_new).label.values
    p_best = pd.read_csv(f_best).label.values
    if f_old is not None:
        p_old = pd.read_csv(f_old).label.values
    
    print('new - best: {}'.format(sum(p_new != p_best)))
    if f_old is not None:
        print('new - old: {}'.format(sum(p_new != p_old)))
        print('old - best: {}'.format(sum(p_old != p_best)))

def diff_probs_to_labels(f_new, f_best='best', f_old=None):
    f_new = PREDS_DIR + f_new + '.csv'
    f_best = OUTPUT_DIR + f_best + '.csv'
    if f_old is not None:
        f_old = OUTPUT_DIR + f_old + '.csv'
    
    print(f_new)
    p_new = pd.read_csv(f_new, index_col='fname').values
    p_new = np.argmax(p_new, axis=1)
    p_new = [ID2NAME[i] for i in p_new]

    p_best = pd.read_csv(f_best).label.values
    if f_old is not None:
        p_old = pd.read_csv(f_old).label.values
    
    print('new - best: {}'.format(sum(p_new != p_best)))
    if f_old is not None:
        print('new - old: {}'.format(sum(p_new != p_old)))
        print('old - best: {}'.format(sum(p_old != p_best)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diff predictions')
    parser.add_argument("--new", type=str, default=None, help="fname for new preds")
    parser.add_argument("--old", type=str, default=None, help="fname for new preds")
    parser.add_argument("--best", type=str, default='best', help="fname for new preds")
    parser.add_argument("--type", type=str, default='labels', help="Type of new file - labels or probs")
    args = parser.parse_args()

    if args.type == 'labels':
        diff_labels(args.new, args.best, args.old)
    elif args.type == 'probs':
        diff_probs_to_labels(args.new, args.best, args.old)
    else:
        print('Bad type: ', args.type)
