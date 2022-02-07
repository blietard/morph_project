import numpy as np
import pandas as pd
from collections import defaultdict


class BytePairEncoder:
    """
    BPE algorithm
    """

    def __init__(self):
        self.ngrams = {'UNK'}
        self.word_series = None
        self.ntrain = None
        self.most_freq_pair = None

    def get_most_freq_pair(self):
        pairs = defaultdict(int)
        for form in self.word_series:
            symbols = form.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += 1
        return max(pairs, key=pairs.get)

    def merge(self):
        ab = ''.join(self.most_freq_pair)
        # add 'ab' to the set of ngrams
        self.ngrams.add(ab)
        new_series = np.empty(len(self.word_series), dtype='object')
        for i, form in enumerate(self.word_series):
            # turn 'a b' to 'ab'
            new_series[i] = form.replace(' '.join(self.most_freq_pair), ab)
        self.word_series = new_series

    def fit(self, corpus, niter):
        self.word_series = [' '.join(list(w)) for w in corpus]
        self.ntrain = len(self.word_series)
        for word in self.word_series:
            s = set(word)
            self.ngrams = self.ngrams | s
        for _ in range(niter):
            self.most_freq_pair = self.get_most_freq_pair()
            self.merge()

    def get_frequencies(self):
        ngrams_freqs = defaultdict(int)
        for word in self.word_series:
            ngrams_list = word.split()
            for ngram in ngrams_list:
                ngrams_freqs[ngram] += 1
        return ngrams_freqs


class NgramsExtractor:
    """

    """

    def __init__(self):
        self.train_data = None
        self.train_stems = None
        self.ntrains = 0
        self.mapping = dict()

    def fit(self, train_set, niter=10):
        self.train_data = pd.DataFrame(train_set, columns=['Lemma', 'Forms', 'Attrs'], dtype='object')
        self.ntrains = self.train_data.shape[0]
        grouped = self.train_data.groupby('Attrs')
        groups = grouped.groups.keys()
        for group in groups:
            df = grouped.get_group(group)
            bpe = BytePairEncoder()
            if df.shape[0] < 2:
                bpe.fit(list(df['Forms']), 0)
            else:
                bpe.fit(list(df['Forms']), niter)
            self.mapping[group] = (bpe, bpe.ntrain)

    def get_ngrams_from_attrs(self, attrs, threshold=0.5):
        try:
            bpe, n = self.mapping[attrs]
        except KeyError:
            raise KeyError('Morphological attributes never encounters in training')
        frequencies = bpe.get_frequencies()
        ngrams_freqs = np.array(list(frequencies.items()), dtype='object')
        mask = ngrams_freqs[:, 1].astype('int32') > threshold * n
        above_threshold = ngrams_freqs[mask][:, 0]
        return above_threshold
