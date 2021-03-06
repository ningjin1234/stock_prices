import pandas as pd
import argparse
import os

def join_all_in_dir(dname, outname, cols=['Adj Close'], date_name='Date'):
    fnames = os.listdir(dname)
    joined = None
    for f in fnames:
        if not f.endswith('.txt') and not f.endswith('.csv'):
            continue
        selected = cols + [date_name]
        df = pd.read_csv(dname+'/'+f)[selected]
        symbol = f[:-4]
        if joined is not None and joined.shape[0] != df.shape[0]:
            print('insufficient records for %s' % symbol)
            continue
        name_dict = dict()
        for col in cols:
            name_dict[col] = col + ' ' + symbol
        df = df.rename(columns=name_dict)
        df.round(5)
        if joined is None:
            joined = df
            continue
        joined = pd.merge(joined, df, how='inner', on=[date_name]) 
    print('shape of joined dataframe: %s' % str(joined.shape))
    joined.to_csv(outname, index=False)

parser = argparse.ArgumentParser()
parser.add_argument('idir', help='path to input directory', type=str)
parser.add_argument('ofname', help='path to output file', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    join_all_in_dir(args.idir, args.ofname)
