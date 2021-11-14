import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np


def get_toxic_span(X, y):
    if len(y) == 0:
        return []
    word_list = []
    i = 0
    for k in range(1, len(y)):
        if y[k] - y[k-1] > 1:
            word_list.append(X[y[i]:y[k-1]+1])
            i = k
    word_list.append(X[y[i]:y[k]+1])
    return word_list


class HITSZ_HLT_ToxicScore:
    def __init__(self, X, y, threshold=0.5):
        self.corpus = ' '.join([x.lower() for x in X])
        self.label = y.copy()
        self.toxic_corpus = [get_toxic_span(X[i], y[i]) for i in range(len(y))]
        self.toxic_corpus = ' '.join([word for span in self.toxic_corpus for word in span]).lower()
        self.threshold = threshold
        self.toxic_corpus_word_count_dict = {}
        self.corpus_word_count_dict = {}

    def _appearance_in_toxic_span(self, w):
        if self.toxic_corpus_word_count_dict.get(w) is None:
            self.toxic_corpus_word_count_dict[w] = self.toxic_corpus.count(w)
        return self.toxic_corpus_word_count_dict[w]

    def _appearance_in_whole_corpus(self, w):
        if self.corpus_word_count_dict.get(w) is None:
            self.corpus_word_count_dict[w] = self.corpus.count(w)
        return self.corpus_word_count_dict[w]

    def calculate(self, w):
        '''
        Returns a detailed score.
        '''
        try:
            score = self._appearance_in_toxic_span(w) / self._appearance_in_whole_corpus(w)
        except ZeroDivisionError:
            score = 0
        finally:
            return score

    def __call__(self, w):
        '''
        Returns a True / False answer only.
        '''
        return self.calculate(w) > self.threshold


def cleanse_dataset(sentence_list):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    return vectorizer.build_tokenizer()(vectorizer.build_preprocessor()(sentence_list))


def ensemble(w, model_list, vote_count):
    return sum([int(model(w)) for model in model_list]) >= vote_count


def generate_span(raw_data, identified_keywords):
    return [np.concatenate([np.array([], dtype=np.int32), *[np.arange(m.start(), m.start()+len(w)) for w in identified_keywords[i] for m in re.finditer(w, raw_data[i].lower())]]).tolist() for i in range(len(raw_data))]


def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)


def main():
    # Read train and validation dataset, and split data into X_train, y_train, X_val, y_val
    train_df = pd.concat([pd.read_csv(f'../data/fold_{i}.csv') for i in range(1, 5)])
    val_df = pd.read_csv('../data/fold_5.csv')

    X_train = train_df['text'].tolist()
    y_train = train_df['spans'].apply(eval).tolist()
    X_val = val_df['text'].tolist()
    y_val = val_df['spans'].apply(eval).tolist()

    # Initialize the model
    toxic_score = HITSZ_HLT_ToxicScore(X_train, y_train, threshold=0.5)
    
    # (Train) Further process input data
    processed_data = [cleanse_dataset(x) for x in X_train]
    processed_data = [[w for w in word_list if ensemble(w, model_list=[toxic_score], vote_count=1)] for word_list in processed_data]
    predicted_span_list = generate_span(X_train, processed_data)
    result = [f1(p, g) for p, g in zip(predicted_span_list, y_train)]
    print('Training set F1 score:', sum(result) / len(result))

    # (Validation) Further process input data
    val_processed_data = [cleanse_dataset(x) for x in X_val]
    val_processed_data = [[w for w in word_list if ensemble(w, model_list=[toxic_score], vote_count=1)] for word_list in val_processed_data]
    val_predicted_span_list = generate_span(X_val, val_processed_data)
    val_result = [f1(p, g) for p, g in zip(val_predicted_span_list, y_val)]
    print('Validation set F1 score:', sum(val_result) / len(val_result))


if __name__ == '__main__':
    main()