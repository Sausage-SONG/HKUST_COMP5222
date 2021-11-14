# HKUST 2021 Fall COMP 5222 Course Project

## Task
SemEval 2021 Task 5: Toxic Spans Detection \[[Link](https://competitions.codalab.org/competitions/25623)\] \[[PDF](https://aclanthology.org/2021.semeval-1.6.pdf)\]

## Group Member
- [SONG Sizhe](https://github.com/Sausage-SONG)
- [CHAN Tsz Ho](https://github.com/Giochen)
- [LAU Yik Lun](https://github.com/Cynwell)

## Results

| Model | F1-Score (%) |
|:-:|:-:|
| CNN | 61.75 |
| BERT-CRF | 67.83 |
| Lexicon | 58.0 |

## How to generate F1 score
(First navigate into this folder containing the file `eval_tsd.py`)

Command:
```
python eval_tsd.py --prediction_file data/output_1.txt --test_file data/fold_1.csv
python eval_tsd.py --prediction_file data/output_2.txt --test_file data/fold_2.csv
python eval_tsd.py --prediction_file data/output_3.txt --test_file data/fold_3.csv
python eval_tsd.py --prediction_file data/output_4.txt --test_file data/fold_4.csv
python eval_tsd.py --prediction_file data/output_5.txt --test_file data/fold_5.csv
```
Result to be shown on the screen:
```
F1 score:  0.6783732474605003
F1 score:  0.6765120478551607
F1 score:  0.673281514892958
F1 score:  0.6861183825905394
F1 score:  0.6619573275698896
```

## Trained Models

Please see the README file of each method.
