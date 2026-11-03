import numpy as np
from scipy.sparse import csr_matrix

from recsys.models import PopularityRecommender, SVDRecommender


def _toy_train_matrix() -> csr_matrix:
    # user0: items {0,1}; user1: items {0,2,3}; user2: items {4}
    rows = [0, 0, 1, 1, 1, 2]
    cols = [0, 1, 0, 2, 3, 4]
    data = [1] * len(rows)
    return csr_matrix((data, (rows, cols)), shape=(3, 5))


def test_popularity_recommender_orders_by_frequency():
    train = _toy_train_matrix()  # item0 appears twice, every other item once
    model = PopularityRecommender()
    model.fit(train)

    idx, _ = model.recommend(n_users=3, k=1, exclude_matrix=None)
    assert (idx[:, 0] == 0).all()


def test_popularity_recommender_excludes_seen_items():
    train = _toy_train_matrix()
    model = PopularityRecommender()
    model.fit(train)

    idx, _ = model.recommend(n_users=3, k=2, exclude_matrix=train)
    # user1 already saw items {0,2,3}; recommendations must come from {1,4}.
    assert set(idx[1]) == {1, 4}


def test_svd_recommender_shapes_and_excludes_seen():
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 3]
    cols = [0, 1, 1, 2, 2, 3, 3, 4, 5]
    data = [1] * len(rows)
    train = csr_matrix((data, (rows, cols)), shape=(4, 6))

    model = SVDRecommender(n_components=2, random_state=0)
    model.fit(train)

    k = 3
    idx, scores = model.recommend(n_users=4, k=k, exclude_matrix=train)
    assert idx.shape == (4, k)
    assert scores.shape == (4, k)

    train_csr = train.tocsr()
    for u in range(4):
        seen = set(train_csr.indices[train_csr.indptr[u] : train_csr.indptr[u + 1]])
        assert not (seen & set(idx[u]))

    # scores within each row should be sorted descending
    for u in range(4):
        assert np.all(np.diff(scores[u]) <= 0)
