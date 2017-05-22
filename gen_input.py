import pandas as pd
import argparse
import os

def read_symbols(fname):
    res = set()
    with open(fname, 'r') as fin:
        for line in fin:
            splitted = line.strip().split(' ')
            res.update(splitted)
    return res

# get all specified columns in the i-th row as a list 
def get_selected_list(df, i, cols):
    return list(df.iloc[i][cols].values)

# get min and max value of specified column between [istart, iend) rows
def get_minmax_in_range(df, istart, iend, col):
    vals = df.iloc[istart:iend][col]
    return vals.min(), vals.max()
