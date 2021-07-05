# get metrics from data
import numpy as np


def metric(scores: np.ndarray, performances: np.ndarray):
    num_po = scores.shape[0]
    num_po_pair = int(num_po * (num_po - 1) / 2)
    real_rank = np.argsort(performances)
    scores = scores[real_rank]
    performances = performances[real_rank]
    real_rank = np.arange(num_po)
    rank = np.argsort(scores)
    coef = np.corrcoef(rank, real_rank)[0][1]
    score1 = np.zeros(num_po_pair)
    score2 = np.zeros(num_po_pair)
    label = np.zeros(num_po_pair)
    tip = []
    count = 0
    for i in range(num_po):
        for j in range(i+1, num_po):
            score1[count] = scores[i]
            score2[count] = scores[j]
            tip.append((i, j))
            label[count] = 1 if performances[i] > performances[j] else 0
            count += 1
    diff_score = (score1 - score2)>=0
    error_rate = sum(diff_score!=label) / num_po_pair
    ind = np.where(diff_score!=label)[0]
    error_pair = []
    for i in ind.tolist():
        s1 = scores[tip[i][0]]
        s2 = scores[tip[i][1]]
        error_pair.append(((tip[i][0], s1, performances[tip[i][0]]), (tip[i][1], s2, performances[tip[i][1]])))
    return coef, error_rate, error_pair, abs(performances[rank[-1]] - performances[real_rank[-1]])


