import urllib.request as url
import pandas as pd
import argparse
from datetime import datetime 
import ssl

'''
potential source for more data:
https://www.quandl.com
'''

# symbol for S&P500 is %5EGSPC (^GSPC)
# to get dividend and date: http://ichart.finance.yahoo.com/table.csv?s=MSFT&g=v (it only gives dividend and date)
# g=m gives monthly data; g=w gives weekly data
# a is start month (0 for January), b is start day (starts at 1), c is start year, 
# d is end month (0 for January), e is end day, f is end year; for example:
# &a=0&b=1&c=2010&d=3&e=17&f=2017
URL_TEMPLATE = 'http://ichart.finance.yahoo.com/table.csv?s=%s&g=%s'
SDATE_TEMPLATE = '&a=%d&b=%d&c=%d'  # month-1, day, year

context = ssl._create_unverified_context()

# start_date is a list of integers: [year, month, day]; month begins at 1 and gets adjusted inside the function
def get_data_for_symbol(symbol, fname=None, path='.', mute=False, start_date=None, data_type='d'):
    try:
        url_symbol = '%5EGSPC' if symbol=='SP500' else symbol
        myurl = URL_TEMPLATE % (url_symbol, data_type)
        if start_date is not None:
            myurl += SDATE_TEMPLATE % (start_date.month-1, start_date.day, start_date.year)
        response = url.urlopen(myurl, context=context)
        lines = response.readlines()
        if fname is None:
            fname = '%s.txt' % symbol
        fname = path + '/' + fname
        with open(fname, 'wb') as fout:
            for line in lines:
                fout.write(line)
        if not mute:
            print('data collected for symbol %s [%s]' % (symbol, str(datetime.now())))
    except Exception as err:
        print(myurl)
        print('failed to get data for symbol %s (%s)' % (symbol, str(err)))

def get_data_for_symbols(symbols, path='.', mute=False, start_date=None, data_type='d'):
    print(symbols)
    for s in symbols:
        if s.strip() != '':
            get_data_for_symbol(s, path=path, mute=mute, start_date=start_date, data_type=data_type)


parser = argparse.ArgumentParser()
parser.add_argument("sfile", help="path to file containing all symbols delimited by space", type=str)
parser.add_argument('opath', help="path to directory containing fetched files for each symbol", type=str)
parser.add_argument('--sdate', help="start date in yyyy-mm-dd", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    symbols = set() 
    with open(args.sfile, 'r') as fin:
        for line in fin.readlines():
            symbols |= set(line.strip().split(' '))
    symbols -= set(['', '\n'])
    start_date = pd.to_datetime(args.sdate) if args.sdate else None
    get_data_for_symbols(symbols, path=args.opath, start_date=start_date)
