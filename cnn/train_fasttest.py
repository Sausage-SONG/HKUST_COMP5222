import csv
import fasttext
from pathlib import Path

def train_fasttext(*csv_files, model_path='model.bin'):

    all_sentences = []
    for file in csv_files:
        with file.open(newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            iterator = enumerate(reader)
            iterator.__next__()
            for idx, line in iterator:
                all_sentences.append(line[1])
    
    corpus_path = Path(__file__).with_name('corpus.tmp')
    with corpus_path.open('w') as f:
        for sentence in all_sentences:
            f.write(str(sentence)+'\n')
    
    print(f'Training model at {model_path}')
    model = fasttext.train_unsupervised(str(corpus_path))
    model.save_model(str(model_path))
    corpus_path.unlink()

if __name__ == '__main__':

    csv_files = [Path(__file__).parent.parent / f'data/fold_{i}.csv' for i in range(1,6)]
    for i in range(len(csv_files)):
        model_path = Path(__file__).with_name('results') / f'test_{i}.bin'
        train_fasttext(*csv_files[:i], *csv_files[i+1:], model_path=model_path)
    