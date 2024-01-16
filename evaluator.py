# Based on the implementation of "Compositionality and Generalization in Emergent Languages"
# by Rahma Chaabouni, Eugene Kharitonov, Diane Bouchacourt, Emmanuel Dupoux, Marco Baroni

import numpy as np
import editdistance
import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed


class EvaluationMetrics:
    def __init__(self, meaning, representation, vocab_size):
        super(EvaluationMetrics, self).__init__()
        self.meaning = meaning
        self.representation = representation
        self.vocab_size = vocab_size

    def calculate_topographic_similarity(self):
        distance_representation = edit_dist(self.representation)
        distance_meaning = cosine_dist(self.meaning.astype(float))
        corr = spearmanr(distance_representation, distance_meaning).correlation
        return corr

    def calculate_position_disentanglement(self):
        meanings = torch.from_numpy(self.meaning)
        representations = torch.from_numpy(self.representation)
        return information_gap_representation(meanings, representations)

    def calculate_bag_of_symbols_disentanglement(self):
        meanings = torch.from_numpy(self.meaning)
        representations = torch.from_numpy(self.representation)
        data_size = representations.size(0)
        histogram = torch.zeros(data_size, self.vocab_size)
        for i, row in enumerate(representations):
            for word_index in row:
                histogram[i, word_index] += 1
        return information_gap_representation(meanings, histogram)

    def calculate_mutual_information_gap(self):

        def compute_mi(attribute_index):
            mi_scores = mutual_info_regression(self.representation, self.meaning[:, attribute_index])
            sorted_mi_scores = sorted(mi_scores, reverse=True)
            if len(sorted_mi_scores) > 1:
                return (sorted_mi_scores[0] - sorted_mi_scores[1]) / sorted_mi_scores[0]
            else:
                return 0

        mig_scores = Parallel(n_jobs=-1)(delayed(compute_mi)(i) for i in range(self.meaning.shape[1]))
        overall_mig = np.mean(mig_scores)
        return overall_mig


def edit_dist(_list):
    n = len(_list)

    def calculate_edit_distance(el1, el2):
        str_el1 = np.array2string(el1)
        str_el2 = np.array2string(el2)
        return editdistance.eval(str_el1, str_el2) / max(len(str_el1), len(str_el2))

    result = Parallel(n_jobs=-1)(delayed(calculate_edit_distance)(_list[i], _list[j]) for i in range(n - 1) for j in range(i + 1, n))
    return result


def cosine_dist(_list):
    n = len(_list)

    def calculate_cosine_distance(el1, el2):
        return cosine(el1, el2)

    result = Parallel(n_jobs=-1)(delayed(calculate_cosine_distance)(_list[i], _list[j]) for i in range(n - 1) for j in range(i + 1, n))
    return result


def information_gap_representation(meanings, representations):
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0

    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(meanings.size(1)):
            x, y = meanings[:, i], representations[:, j]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())
    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)


def entropy(messages):
    from collections import defaultdict
    freq_table = defaultdict(float)
    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t
    try:
        t = t.item()
    except ValueError:
        t = tuple(t.view(-1).tolist())
    return t


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)
    xys = []
    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)
    e_xy = entropy(xys)
    return e_x + e_y - e_xy
