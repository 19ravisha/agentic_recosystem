import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recsys.evaluate import ndcg_at_k, recall_at_k


def _toy_held_out() -> csr_matrix:
    # user0 relevant items: {2, 9}; user1 relevant items: {1, 4}
    rows = [0, 0, 1, 1]
    cols = [2, 9, 1, 4]
    data = [1] * len(rows)
    return csr_matrix((data, (rows, cols)), shape=(2, 10))


def test_recall_at_k_hand_computed():
    top_k = np.array([[5, 2, 7], [1, 2, 3]])
    held_out = _toy_held_out()

    # user0: hit {2} out of 2 relevant -> 0.5; user1: hit {1} out of 2 relevant -> 0.5
    assert recall_at_k(top_k, held_out, k=3) == pytest.approx(0.5)


def test_ndcg_at_k_hand_computed():
    top_k = np.array([[5, 2, 7], [1, 2, 3]])
    held_out = _toy_held_out()

    assert ndcg_at_k(top_k, held_out, k=3) == pytest.approx(0.5, abs=1e-6)


def test_recall_and_ndcg_skip_users_with_no_held_out():
    top_k = np.array([[5, 2, 7]])
    held_out = csr_matrix((1, 10))  # no relevant items for this user

    assert np.isnan(recall_at_k(top_k, held_out, k=3))
    assert np.isnan(ndcg_at_k(top_k, held_out, k=3))
