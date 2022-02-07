from typing import List
import numpy as np


class Assembler:

    def __init__(self, grouped_stems: List, ngrams_per_attr_group: List):
        self.grouped_stems = grouped_stems
        self.ngrams_per_attr_group = ngrams_per_attr_group
        self.assemblers = {}

    def fit(self):

        results = {}

        # Parsing each group of lemmas
        for i_lemma_group, lemma_group in enumerate(self.grouped_stems):
            # Parsing each entry
            for stem, final, i_morph_attr_group in lemma_group:

                # Computing matching array
                # TODO: only handling exact matching at this point
                matching_array = -2 * np.ones(len(final))

                ngrams = self.ngrams_per_attr_group[i_morph_attr_group]

                # Finding stem and updating matching array
                i_stem = final.find(stem)
                if i_stem != -1:
                    matching_array[i_stem:i_stem + len(stem)] = -1

                # Trying to match every ngram of the list into the matching array
                for i, ngram in enumerate(ngrams):
                    # print(ngram)
                    res = 0
                    while True:
                        res = final.find(ngram, res)
                        # print(res)
                        if res == -1:
                            break
                        if matching_array[res] == -2:
                            matching_array[res:res + len(ngram)] = i
                            break
                        else:
                            res += len(ngram)
                            if res >= len(final):
                                res = -1

                # TODO: if there are non-matched characters, we ignore them for the moment
                # We could maybe "augment" the ngrams, adding them at the end of the list ?
                matching_array = np.delete(matching_array, np.where(matching_array == -2))

                # Reducing the array into an ordering
                _, idx = np.unique(matching_array, return_index=True)
                ordering = list(matching_array[np.sort(idx)].astype(int))

                # Adding the ordering to the list of results
                # for its lemma group and morphological attributes group
                if (i_lemma_group, i_morph_attr_group) in results:
                    results[(i_lemma_group, i_morph_attr_group)].append(ordering)
                else:
                    results[(i_lemma_group, i_morph_attr_group)] = [ordering]

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

    def predict(self, stem: str, i_lemma_group: int, i_morph_attr_group: int):

        assembler = self.assemblers[(i_lemma_group, i_morph_attr_group)]
        possible_ngrams = self.ngrams_per_attr_group[i_morph_attr_group]

        prediction = ""

        for i in assembler:
            if i == -1:
                prediction += stem
            else:
                prediction += possible_ngrams[i]

        return prediction