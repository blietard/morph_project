import numpy as np
import pandas as pd
import enchant
from scipy.stats import mode


# Method to measure similarity between lemmas (using a 'normalized' Levenshtein distance)
def custom_metric(x, y):
    return enchant.utils.levenshtein(x, y) / (len(x)+len(y))


class Knn:
    
    """
    K-Nearest Neighbors model to cluster the similar lemmas for stem extraction
    
    """
    
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


class Stemmer:
    
    """
    Model that is trained to extract the stem from lemmas
    
    """
    
    def __init__(self, n_neighbors):
        self.knn = Knn(n_neighbors)
        self.grouped_stems = {}

    # Method to find the stem (longest common substring) from the string array
    def find_stem(self, arr):
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

    
    # Method to build a dict that links the cutting distances from the start and end of a lemma to the list of lemmas
    def stem_dict(self, dic):
        stem_dic = dict()
        for lemma in dic:
            lemma_plus_form = [lemma] + [a[0] for a in dic[lemma]]
            stem = self.find_stem(lemma_plus_form)
            start = lemma.index(stem)
            end = len(lemma) - (start+len(stem))
            pos = (start, end)
            
              

            try :
                stem_dic[pos].append(lemma)
                self.grouped_stems[pos] += list((zip(len(dic[lemma])*[stem],dic[lemma])))
            except KeyError:
                stem_dic[pos] = [lemma,]
                self.grouped_stems[pos] = list(zip(len(dic[lemma])*[stem],dic[lemma]))
        return stem_dic


    # Method to build the input data for the K-NN model
    def stem_dict_to_xy(self, dic):
        X = []
        Y = []
        for key in dic:
            X += dic[key]
            Y += [key]*len(dic[key])
        return np.array(X), np.array(Y)
    
    def fit(self, data):
        
        # Building dictionary that links lemmas to the list of forms that are in the dataset
        lemma_dict = dict()
        count = dict()
        for sent in data:
            try :
                count[sent[0]]+=1
                lemma_dict[sent[0]].append( (sent[1],sent[2]) )
            except KeyError:
                try : 
                    count[sent[0]] = 1
                    lemma_dict[sent[0]] =  [ (sent[1],sent[2]) ,]
                except IndexError:
                    raise IndexError(str(sent))
        
        # Building the input for the K-NN model
        stem_dict = self.stem_dict(lemma_dict)
        X, y = self.stem_dict_to_xy(stem_dict)
        
        # Fitting the K-NN to the training data
        self.knn.fit(X, y)
        
    def predict(self, lemma):
        
        pos = self.knn.predict(lemma)
        start, end = pos
        
        return pos, lemma[start:len(lemma)-end]
        
        
        
        
        
            
        
        
        
        
        
        