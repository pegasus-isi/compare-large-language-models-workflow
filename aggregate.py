#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='agg.csv')
    parser.add_argument('--plot', default='agg.pdf')
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    series = []
    for fn in args.input:
        s = pd.read_json(fn, typ='series')
        s['variant'] = os.path.basename(fn).removesuffix('.json')
        series.append(s)

    df = pd.DataFrame(series)
    df = df.set_index('variant')
    df.to_csv(args.output)

    fig = plt.figure()
    df.plot.bar()
    plt.savefig(args.plot,  bbox_inches="tight")

if __name__ == '__main__':
    main()

