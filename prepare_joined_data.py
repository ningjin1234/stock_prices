import pandas as pd

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

prepare_joined_data('joined_prices.txt', 'training.txt', last_date=pd.to_datetime('2016-01-01'))
prepare_joined_data('joined_prices.txt', 'test.txt', first_date=pd.to_datetime('2016-01-01'))
