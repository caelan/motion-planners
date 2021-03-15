from itertools import combinations, product, permutations

import numpy as np

from motion_planners.tkinter.viewer import get_distance
from motion_planners.utils import INF

from scipy.spatial.kdtree import KDTree

def compute_median_distance(path1, path2):
    differences = [get_distance(q1, q2) for q1, q2 in product(path1, path2)]
    return np.median(differences)


def compute_minimax_distance(path1, path2):
    overall_distance = 0.
    for path, other in permutations([path1, path2]):
        tree = KDTree(other)
        for q1 in path:
            #closest_distance = min(get_distance(q1, q2) for q2 in other)
            closest_distance = get_distance(q1, tree.query(q1, k=1, eps=0.))
            overall_distance = max(overall_distance, closest_distance)
    return overall_distance


def score_portfolio(portfolio):
    # TODO: score based on collision voxel overlap at different resolutions
    score_fn = compute_minimax_distance # compute_median_distance | compute_minimax_distance
    score = INF
    for path1, path2 in combinations(portfolio, r=2):
        score = min(score, score_fn(path1, path2))
    return score


def exhaustively_select_portfolio(candidates, k=4):
    if len(candidates) <= k:
        return candidates
    best_portfolios, best_score = [], 0
    for portfolio in combinations(candidates, r=k):
        score = score_portfolio(portfolio)
        if score > best_score:
            best_portfolios, best_score = portfolio, score
    return best_portfolios

def greedily_select_portfolio(candidates, k=10):
    # Higher score is better
    if len(candidates) <= k:
        return candidates
    raise NotImplementedError()
    #return best_portfolios
