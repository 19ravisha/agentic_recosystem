import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def _batched_top_k(
    score_batch_fn,
    n_users: int,
    k: int,
    exclude_matrix: sparse.csr_matrix | None,
    batch_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k item indices/scores per user, batched to bound memory use."""
    all_indices = np.empty((n_users, k), dtype="int64")
    all_scores = np.empty((n_users, k), dtype="float32")

    for start in range(0, n_users, batch_size):
        stop = min(start + batch_size, n_users)
        user_idx = np.arange(start, stop)
        scores = score_batch_fn(user_idx)

        if exclude_matrix is not None:
            sub = exclude_matrix[user_idx].tocoo()
            scores[sub.row, sub.col] = -np.inf

        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        all_indices[start:stop] = np.take_along_axis(part, order, axis=1)
        all_scores[start:stop] = np.take_along_axis(part_scores, order, axis=1)

    return all_indices, all_scores


class PopularityRecommender:
    """Recommends the globally most-popular train items to every user."""

    def fit(self, train_matrix: sparse.csr_matrix) -> None:
        self.item_scores = np.asarray(train_matrix.sum(axis=0)).ravel().astype("float32")
        self.n_items = train_matrix.shape[1]

    def recommend(
        self, n_users: int, k: int, exclude_matrix: sparse.csr_matrix | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        def score_batch(user_idx: np.ndarray) -> np.ndarray:
            return np.tile(self.item_scores, (len(user_idx), 1))

        return _batched_top_k(score_batch, n_users, k, exclude_matrix)


class SVDRecommender:
    """Matrix-factorization recommender via TruncatedSVD on the implicit interaction matrix."""

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, train_matrix: sparse.csr_matrix) -> None:
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.user_factors = svd.fit_transform(train_matrix).astype("float32")
        self.item_factors = svd.components_.T.astype("float32")
        self.n_items = train_matrix.shape[1]

    def recommend(
        self, n_users: int, k: int, exclude_matrix: sparse.csr_matrix | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        def score_batch(user_idx: np.ndarray) -> np.ndarray:
            return self.user_factors[user_idx] @ self.item_factors.T

        return _batched_top_k(score_batch, n_users, k, exclude_matrix)
