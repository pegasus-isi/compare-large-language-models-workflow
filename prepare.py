#!/usr/bin/env python3
import argparse
from datasets import load_dataset
from datasets.arrow_dataset import pa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yelp_review_full')
    parser.add_argument('--dataset-head')
    parser.add_argument('--test-split', default=0.1, type=float)
    parser.add_argument('--validation-split', default=0.1, type=float)
    parser.add_argument('--train-out', default='train.csv')
    parser.add_argument('--test-out', default='test.csv')
    parser.add_argument('--validation-out', default='validation.csv')
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split='test')
    if args.dataset_head:
        ds = ds.select(range(int(args.dataset_head)))
    train_split = ds.train_test_split(test_size=args.test_split+args.validation_split, shuffle=True)
    train_split["train"].to_csv(args.train_out)

    test_split = train_split["test"].train_test_split(test_size = args.validation_split / (args.test_split + args.validation_split), shuffle=True)
    test_split['train'].to_csv(args.test_out)
    test_split['test'].to_csv(args.validation_out)

if __name__ == '__main__':
    main()
