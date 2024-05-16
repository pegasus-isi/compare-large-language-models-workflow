# Compare LLMs Workflow

This repository demonstrates how to use the Pegasus Workflow Management System (WMS) to compare the accuracy of Supervised Fine-Tuning (SFT) and pretrained models available in [HuggingFace model repository](https://huggingface.co/models).

## Overview

Pegasus WMS is a powerful tool for managing and executing complex workflows on distributed computing resources. In this example, we use its abilities to run computations simultaneously, making it easier to compare various models efficiently.

### File description:

- `prepare.py` - fetches and prepares the dataset
- `evaluate.py` - evaluates performance of single model
- `aggregate.py` - aggregates results of evaluation steps
- `workflow.py` - builds and submits workflow

## Dataset

The example included in this repository utilizes the [Yelp review dataset](https://huggingface.co/datasets/yelp_review_full). This dataset contains reviews along with their associated ratings, making it suitable for training and evaluating various natural language processing models.

## Usage

To run the example:

1. Clone this repository to your Linux machine.
2. Create a virtual environment using Python:

```bash
python3 -m venv env
source env/bin/activate
```
3. Install requirements
```bash
pip install -r requirements.txt
```
4. Run workflow
```bash
./workflow.py --models bert-base-cased albert-base-v2 --batch-size 8
```

## Container
The workflow uses Singularity containers to execute each step. In the default setup, a container is created using the prebuilt Docker image from DockerHub.

To build the Singularity image locally, execute the following commands:
```bash
docker build -t compare-llms-workflow .
singularity build base.sif docker-daemon://compare-llms-workflow:latest
```
Next, specify the path to the built image using the `--image` option.
```bash
./workflow.py --image $PWD/base.sif ...
```

## Results
Once the computations are finished, the results will be aggregated into agg.csv and rendered as plots for easy interpretation (`agg.pdf`)
