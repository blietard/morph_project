# Main file for the project

# Importing useful packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import nltk
from collections import Counter

# Loading the data
file_path = './task0-data/DEVELOPMENT-LANGUAGES/uralic/fin.dev'

file = open(file_path)

raw_words = []
final_words = []
inflections_words = []

for line in file:
    raw, final, inflections = line.strip().split("	")
    inflections_list = inflections.split(';')
    # print("Raw: '{}', Final: '{}', Inflections: {}".format(raw, final, inflections_set))
    raw_words.append(raw)
    final_words.append(final)
    inflections_words.append(inflections_list)

# Extracting the characters
print("Studying the characters of the language")
characters_raw_set = set("".join(raw_words))
characters_final_set = set("".join(final_words))
characters_set = characters_raw_set.union(characters_final_set)
full_characters = "".join(sorted(characters_set))
print("Total number of characters : {}\nCharacters: {}".format(len(full_characters), full_characters))
lowercase_characters = "".join(sorted(set(full_characters.lower())))
print("Total number of lower characters : {}\nLower characters: {}\n".format(len(lowercase_characters),
                                                                             lowercase_characters))

# Extracting tags and relating them
print("Studying the tags of the language")
all_tags = sorted(set.union(*[set(l) for l in inflections_words]))
n_tags = len(all_tags)
print("Number of tags (including combinations): {} \nTags: {}\n".format(n_tags, all_tags))

# First try to observe compatible tags
tag = 'V'
compatible_tags = set()
for inflections in inflections_words:
    if tag in inflections:
        compatible_tags = compatible_tags.union(set(inflections))
print(tag, len(compatible_tags), sorted(compatible_tags))

co_occurrence_matrix = np.zeros((n_tags, n_tags))

tag_to_index = {tag: i for i, tag in enumerate(all_tags)}
#print(tag_to_index)

for i, tag in enumerate(all_tags):
    for inflections in inflections_words:
        if tag in inflections:
            for occuring_tag in inflections:
                co_occurrence_matrix[i, tag_to_index[occuring_tag]] = 1

n_cc, labels = sparse.csgraph.connected_components(co_occurrence_matrix)
print(n_cc)

# plt.matshow(co_occurrence_matrix)
# plt.show()

# Now trying to link n-grams to tags

n = 3

tag = 'TRANS'
type_tag = 'N'
n_grams_for_tag = {}
for final, inflections in zip(final_words, inflections_words):
    if type_tag in inflections:
        if tag in inflections:
            n_grams = nltk.ngrams(final, n)
            for gram in n_grams:
                gram = "".join(gram)
                if gram in n_grams_for_tag:
                    n_grams_for_tag[gram] += 1
                else:
                    n_grams_for_tag[gram] = 1

c = Counter(n_grams_for_tag)
top_k = 10
most_common_ngrams = c.most_common(5)
print(tag, most_common_ngrams)