import numpy as np
import pandas as pd
import argparse

OUTPUT_DIR = '../output/'

def diff_preds(f_new, f_best='best', f_old=None):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diff predictions')
    parser.add_argument("--new", type=str, default=None, help="fname for new preds")
    parser.add_argument("--old", type=str, default=None, help="fname for new preds")
    parser.add_argument("--best", type=str, default='best', help="fname for new preds")
    args = parser.parse_args()

    diff_preds(args.new, args.best, args.old)
