import pandas as pd
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

prepare_all_in_dir('daily_prices', 'training.tsv', first_date=dparse('2010-01-01'), last_date=dparse('2015-12-31'), debug_info=False)

prepare_all_in_dir('daily_prices', 'test.tsv', first_date=dparse('2016-01-01'), debug_info=False)


# prepare_all_in_dir('../data/test', 'debug.tsv', first_date=dparse('2010-01-01'), last_date=dparse('2015-12-31'), debug_info=False)
