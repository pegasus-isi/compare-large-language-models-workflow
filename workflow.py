#!/usr/bin/env python
import argparse
import os
from os.path import dirname
from pathlib import Path
from Pegasus.api import *

TRAIN_DATA = File('train.csv')
TEST_DATA = File('test.csv')
VALIDATION_DATA = File('validation.csv')

def main(top_dir: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yelp_review_full')
    parser.add_argument('--dataset-head')
    parser.add_argument('--test-split', default=0.1, type=float)
    parser.add_argument('--validation-split', default=0.1, type=float)
    parser.add_argument('--models', default=['albert-base-v2'], nargs='+')
    parser.add_argument('--metrics', nargs='+', default=['accuracy'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-labels', default=5)
    parser.add_argument('--image')

    args = parser.parse_args()

    props = Properties()
    props['pegasus.mode'] = 'development'
    props.write()

    tc = TransformationCatalog()
    base_container = Container('base',
                               Container.SINGULARITY,
                               image=args.image if args.image else 'docker://pzuk/compare-llms',
                               image_site='local' if args.image else 'docker_hub')
    tc.add_containers(base_container)

    prepare = Transformation('prepare', site='local', pfn=top_dir / 'prepare.py',
                             is_stageable=True,
                             container=base_container)\
                .add_pegasus_profile(memory=2048)

    evaluate = Transformation('evaluate', site='local', pfn=top_dir / 'evaluate.py',
                             is_stageable=True,
                             container=base_container)\
                .add_pegasus_profile(cores=8, runtime=14400, memory=12*1024, gpus=1)

    aggregate = Transformation('aggregate', site='local', pfn=top_dir / 'aggregate.py',
                             is_stageable=True,
                             container=base_container)

    tc.add_transformations(prepare, evaluate, aggregate)
    tc.write()

    rc = ReplicaCatalog()
    rc.write()

    wf = Workflow('llm_accuracy', infer_dependencies=True)
    prepare_job = Job(prepare)\
            .add_args('--dataset', args.dataset,
                      '--test-split', args.test_split,
                      '--validation-split', args.validation_split)\
            .add_outputs(TRAIN_DATA, TEST_DATA, VALIDATION_DATA)

    if args.dataset_head:
        prepare_job.add_args('--dataset-head', args.dataset_head)

    wf.add_jobs(prepare_job)

    outputs = []
    for m in args.models:
        common_args = [
            '--model', m, 
            '--metrics', *args.metrics,
            '--batch-size', args.batch_size,
            '--n-labels', args.n_labels,
        ]

        pretrained_out_name = File(f'{m}-pretrained.json')
        eval_pretrained = Job(evaluate)\
            .add_args(*common_args,
                      '--output', pretrained_out_name,
            ).add_inputs(VALIDATION_DATA)\
            .add_outputs(pretrained_out_name)

        sft_out_name = File(f'{m}-sft.json')
        eval_sft = Job(evaluate)\
            .add_args(*common_args,
                      '--fine-tune',
                      '--output', sft_out_name,
            ).add_inputs(TRAIN_DATA, TEST_DATA, VALIDATION_DATA)\
            .add_outputs(sft_out_name)

        outputs += [pretrained_out_name, sft_out_name]
        wf.add_jobs(eval_pretrained, eval_sft)

    agg_job = Job(aggregate)\
            .add_args(*outputs)\
            .add_inputs(*outputs)\
            .add_outputs(File('agg.csv'), File('agg.pdf'))
    wf.add_jobs(agg_job)

    wf.plan(submit=True, dir='runs', sites=['condorpool'], output_sites=['local'])

import logging

logging.basicConfig(level=logging.DEBUG)
BASE_DIR = Path(".").resolve()
if __name__ == '__main__':
    main(Path(os.path.dirname(__file__)))
