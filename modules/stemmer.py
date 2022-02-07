import numpy as np
import enchant
from scipy.stats import mode


# Function to find the stem (longest common substring) from the string array
def find_stem(arr):
    accept = False
    res = ""
    s = arr[0] # Take first word from array as reference

    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):

            # generating all possible substrings of our reference string
            stem = s[i:j]
            k = 1
            for k in range(1, len(arr)):
                accept = True
                if stem not in arr[k]:
                    accept = False
                    break

            # If current substring is present in all strings and its length is greater than current result
            if accept and len(res) < len(stem):
                res = stem

    return res


def stem_dict(dic):
    stem_dic = dict()
    for lemma in list(dic.keys()):
        lemma_plus_form = [lemma] + [a[0] for a in dic[lemma]]
        stem = find_stem(lemma_plus_form)
        start = lemma.index(stem)
        end = len(lemma) - (start+len(stem))
        pos = (start, end)

        try :
            stem_dic[pos].append(lemma)
        except KeyError:
            stem_dic[pos] = [lemma,]
    return stem_dic


def stem_dict_to_xy(dic):
    X = []
    Y = []
    for key in dic.keys():
        X += [[a] for a in dic[key]]
        Y += [key]*len(dic[key])
    return np.array(X), np.array(Y)


def custom_metric(x, y):
    return enchant.utils.levenshtein(x, y) / (len(x) + len(y))


class Knn:
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, x, y):
        self.X = np.array(x)
        self.y = np.array(y)

    def predict(self, test):
        dists = np.zeros(len(self.X))
        for i in range(len(self.X)):
            dists[i] = custom_metric(self.X[i], test)
        nn_ids = dists.argsort()[:self.k]
        return tuple(mode(self.y[nn_ids])[0][0])