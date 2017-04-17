import urllib.request as url
import datetime as dt

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
URL_TEMPLATE = 'http://ichart.finance.yahoo.com/table.csv?s=%s'

def get_data_for_symbol(symbol, fname=None, path='.', mute=False):
    try:
        response = url.urlopen(URL_TEMPLATE % symbol)
        lines = response.readlines()
        if fname is None:
            fname = '%s.txt' % symbol
        fname = path + '/' + fname
        with open(fname, 'wb') as fout:
            for line in lines:
                fout.write(line)
        if not mute:
            print('data collected for symbol %s [%s]' % (symbol, str(dt.datetime.now())))
    except Exception as err:
        print('failed to get data for symbol %s (%s)' % (symbol, str(err)))

def get_data_for_symbols(symbols, path='.', mute=False):
    for s in symbols:
        get_data_for_symbol(s, path=path, mute=mute)

symbols = []
with open('symbols.txt', 'r') as fin:
    for line in fin.readlines():
        symbols += line.split(' ')
symbols.remove('')
get_data_for_symbols(symbols)
