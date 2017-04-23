import pandas as pd
import argparse
from dateutil.parser import parse as dparse
import os

def normalize_window_data(w):
    res = []
    for i in range(1, len(w)):
        res.append((w[i]-w[i-1]) / w[i-1])
    return res

def window_to_str(w, debug_info=False):
    res = ''
    for i in range(len(w)):
        if debug_info and i >= len(w)-2:
            res += '\t' + str(w[i])
            continue
        if i > 0:
            res += '\t'
        res += '%.5g' % w[i]
    return res

def prepare_one_stock(fname, wsize=6, col_name='Adj Close', date_name='Date', first_date=None, last_date=None, debug_info=False):
    df = pd.read_csv(fname)
    df_r = df[::-1]
    res = []
    for i in range(0, df_r.shape[0]-wsize): # range may need to change after more target options become available
        tdate = dparse(df_r.iloc[i+wsize][date_name])
        if (first_date is not None and tdate < first_date) or (last_date is not None and tdate > last_date):
            continue
        window = list(df_r[i:(i+wsize)][col_name].values) # these are input variables
        window.append(df_r.iloc[i+wsize][col_name]) # this is target, more options to come
        window = normalize_window_data(window)
        if debug_info:
            window.append(fname[:-4])
            window.append(df_r.iloc[i]['Date'])
        res.append(window)
    return res

def prepare_all_in_dir(dname, outname, wsize=6, col_name='Adj Close', date_name='Date', first_date=None, last_date=None, debug_info=False):
    fnames = os.listdir(dname)
    with open(outname, 'w') as fout:
        for f in fnames:
            res = prepare_one_stock(dname+'/'+f, wsize=wsize, col_name=col_name, date_name=date_name, 
                                    first_date=first_date, last_date=last_date, debug_info=debug_info)
            for row in res:
                fout.write(window_to_str(row, debug_info=debug_info)) 
                fout.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument('idir', help='path to input directory', type=str)
parser.add_argument('ofname', help='path to output file', type=str)
parser.add_argument('--first', help='first target date in yyyy-mm-dd', type=str)
parser.add_argument('--last', help='last target date in yyyy-mm-dd, inclusive', type=str)
parser.add_argument('--shuffle', help='whether to shuffle the records in output file; default is True', type=bool)

if __name__ == '__main__':
    args = parser.parse_args()
    first_date = pd.to_datetime(args.first) if args.first else None
    last_date = pd.to_datetime(args.last) if args.last else None
    prepare_all_in_dir(args.idir, args.ofname, first_date=first_date, last_date=last_date, debug_info=False)
    doShuffle = args.shuffle if args.shuffle else True
    if doShuffle:
        print('shuffling records in output file...')
        os.system('python shuffle_file.py %s %s' % (args.ofname, args.ofname))


