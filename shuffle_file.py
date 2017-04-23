import pandas as pd
import os
import argparse

def shuffle_file(ifname, ofname, hasHeader=True):
    if hasHeader:
        df = pd.read_csv(ifname)
    else:
        df = pd.read_csv(ifname, header=None)
    df_shuffled = df.sample(frac=1.0)
    df_shuffled.to_csv(ofname, index=False)



parser = argparse.ArgumentParser() 
parser.add_argument("ifname", help='path to input file', type=str)
parser.add_argument('ofname', help='path to output file', type=str)
parser.add_argument('--hasHeader', help='whether input file has header', type=bool)

if __name__ == '__main__':
    args = parser.parse_args()
    hasHeader = args.hasHeader if args.hasHeader else True
    shuffle_file(args.ifname, args.ofname, hasHeader=hasHeader)
