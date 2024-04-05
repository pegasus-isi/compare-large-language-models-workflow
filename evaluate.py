#!/usr/bin/env python3
# Based on https://huggingface.co/docs/transformers/training#finetuning-in-pytorch-with-the-trainer-api
import argparse
import json
import torch
from datasets import load_dataset
from itertools import chain
from numpy import indices
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, get_scheduler
from transformers.data import metrics
from transformers.pipelines import AutoModelForSequenceClassification
import evaluate


def compute_metrics(logits, labels , metrics):
    preditctions = torch.argmax(logits, dim=-1)
    return metrics.co

def fine_tune(model, train_data, test_data, batch_size):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_dataloader))
    model.train()
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', default='train.csv')
    parser.add_argument('--test-data', default='test.csv')
    parser.add_argument('--eval-data', default='validation.csv')
    parser.add_argument('--model', default='albert-base-v2')
    parser.add_argument('--metrics', nargs='+', default=['accuracy'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', default='out.json')
    parser.add_argument('--n-labels', default=5)
    parser.add_argument('--fine-tune', action='store_true', default=False)
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.fine_tune:
        data = load_dataset('csv', data_files={
            'train': args.train_data,
            'test': args.test_data,
            'validation': args.eval_data,
        })
    else:
        data = load_dataset('csv', data_files={
            'validation': args.eval_data,
        })
    n_labels = int(args.n_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_data = data.map(lambda item: tokenizer(item['text'], padding="max_length", truncation=True), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_data = tokenized_data.remove_columns(['text'])
    tokenized_data = tokenized_data.rename_column('label', 'labels')
    tokenized_data.set_format('torch')


    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=n_labels).to(device)
    if args.fine_tune:
        fine_tune(model, tokenized_data['train'], tokenized_data['test'], args.batch_size)

    model.eval()
    metrics = evaluate.combine(args.metrics)
    eval_dataloader = DataLoader(tokenized_data['validation'], batch_size=args.batch_size)
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        metrics.add_batch(predictions=predictions, references=batch['labels'])
    res = metrics.compute()
    with open(args.output, 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    main()
