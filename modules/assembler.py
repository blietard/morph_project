from typing import Dict
import numpy as np


class Assembler:

    def __init__(self):
        self.grouped_stems = None
        self.ngrams_per_attr_group = None
        self.assemblers = {}

    def fit(self, grouped_stems: Dict, ngrams_per_attr_group: Dict):
        
        self.grouped_stems = grouped_stems
        self.ngrams_per_attr_group = ngrams_per_attr_group

        results = {}

        # Parsing each group of lemmas
        for lemma_group in self.grouped_stems:
            # Parsing each entry
            
            entries = self.grouped_stems[lemma_group]
            
            for stem, (final, morph_attr_group) in entries:

                # Computing matching array
                # TODO: only handling exact matching at this point
                matching_array = -2 * np.ones(len(final))

                ngrams = self.ngrams_per_attr_group[morph_attr_group]

                # Finding stem and updating matching array
                i_stem = final.find(stem)
                if i_stem != -1:
                    matching_array[i_stem:i_stem + len(stem)] = -1

                # Trying to match every ngram of the list into the matching array
                for i, ngram in enumerate(ngrams):
                    res = 0
                    while True:
                        res = final.find(ngram, res)
                        if res == -1:
                            break                            
                        if matching_array[res] == -2:
                            matching_array[res:res + len(ngram)] = i
                            break
                        else:
                            res += len(ngram)
                            if res >= len(final):
                                break

                # TODO: if there are non-matched characters, we ignore them for the moment
                # We could maybe "augment" the ngrams, adding them at the end of the list ?
                matching_array = np.delete(matching_array, np.where(matching_array == -2))

                # Reducing the array into an ordering
                _, idx = np.unique(matching_array, return_index=True)
                ordering = list(matching_array[np.sort(idx)].astype(int))

                # Adding the ordering to the list of results
                # for its lemma group and morphological attributes group
                if (lemma_group, morph_attr_group) in results:
                    results[(lemma_group, morph_attr_group)].append(ordering)
                else:
                    results[(lemma_group, morph_attr_group)] = [ordering]

        # For each combination, determining the most common ordering
        for key in results:

            orderings_list = results[key]

            # TODO: in case of equality, the first ordering in the list wins
            max_freq = 0
            most_common = orderings_list[0]
            for ordering in orderings_list:
                freq = orderings_list.count(list(ordering))
                if freq > max_freq:
                    max_freq = freq
                    most_common = ordering

            self.assemblers[key] = most_common

    def predict(self, stem: str, lemma_group: tuple, morph_attr_group: str):

        assembler = self.assemblers[(lemma_group, morph_attr_group)]
        possible_ngrams = self.ngrams_per_attr_group[morph_attr_group]

        prediction = ""

        for i in assembler:
            if i == -1:
                prediction += stem
            else:
                prediction += possible_ngrams[i]

        return prediction