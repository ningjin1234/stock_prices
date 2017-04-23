import pandas as pd
import os
import argparse

DELIMITER = '\t'

def get_header(cols, wsize):
    header = 'Date'
    tmp_cols = list(cols)
    if 'Date' in tmp_cols:
        tmp_cols.remove('Date')
    for i in range(wsize-1):
        for c in tmp_cols:
            header += DELIMITER + c + str(i)
    return header

def prepare_joined_data(ifname, ofname, wsize=6, first_date=None, last_date=None):
    df = pd.read_csv(ifname)
    df = df[::-1]
    col_names = list(df.columns.values)
    col_names.remove('Date')
    df_r = df[col_names]
    header = get_header(col_names, wsize)
    with open(ofname, 'w') as fout:
        fout.write(header + '\n')
        for i in range(0, df_r.shape[0]-wsize):
            tdate = pd.to_datetime(df.iloc[i+wsize]['Date'])
            if (first_date is not None and tdate < first_date) or (last_date is not None and tdate > last_date):
                continue
            row_str = df.iloc[i+wsize]['Date']
            prev_row = None
            for j in range(wsize):
                cur_row = df_r.iloc[i+j].values
                if prev_row is not None:
                    vals = (cur_row-prev_row)/prev_row
                    row_str += DELIMITER
                    row_str += DELIMITER.join(map(str, vals))
                prev_row = cur_row
            fout.write(row_str + '\n')



parser = argparse.ArgumentParser()
parser.add_argument('ifname', help='path to input file', type=str)
parser.add_argument('ofname', help='path to output file', type=str)
parser.add_argument('--first', help='first target date in yyyy-mm-dd', type=str)
parser.add_argument('--last', help='last target date in yyyy-mm-dd, inclusive', type=str)
parser.add_argument('--shuffle', help='whether to shuffle the records, default is True', type=bool)

if __name__ == '__main__':
    args = parser.parse_args()
    first_date = pd.to_datetime(args.first) if args.first else None
    last_date = pd.to_datetime(args.last) if args.last else None
    prepare_joined_data(args.ifname, args.ofname, last_date=last_date, first_date=first_date)
    doShuffle = args.shuffle if args.shuffle else True
    if doShuffle:
        print('shuffling records in output file...')
        os.system('python shuffle_file.py %s %s' % (args.ofname, args.ofname))
