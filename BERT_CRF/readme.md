# Bert-based Method

This directory contains the BERT+CRF method for the task.

## Requirements

- tensorflow==2.6.0
- sklearn
- transformers
- tensorflow_addons
- pandas
- numpy
## Results
output_{1-5}.txt represent the prediction result of fold_{1-5}.csv, which is the arguments of eval_tsd.py to exam the F1 score.

reporting F1-score: 0.6783

## training time

This model should be trained within 30mins with the help of GPU, within 8hours with CPU only.

## trained model
[Dropbox](https://www.dropbox.com/sh/p5gm8ulatmyvb06/AADRVqH2LXBC4msqnD7zQNbOa?dl=0)
