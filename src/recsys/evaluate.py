import numpy as np
from scipy import sparse


def recall_at_k(top_k_indices: np.ndarray, held_out_matrix: sparse.csr_matrix, k: int) -> float:
    """Mean Recall@k over users that have at least one held-out interaction."""
    held_out = held_out_matrix.tocsr()
    recalls = []
    for u in range(top_k_indices.shape[0]):
        relevant = set(held_out.indices[held_out.indptr[u] : held_out.indptr[u + 1]])
        if not relevant:
            continue
        hits = sum(1 for item in top_k_indices[u, :k] if item in relevant)
        recalls.append(hits / len(relevant))
    return float(np.mean(recalls)) if recalls else float("nan")


def ndcg_at_k(top_k_indices: np.ndarray, held_out_matrix: sparse.csr_matrix, k: int) -> float:
    """Mean NDCG@k over users that have at least one held-out interaction."""
    held_out = held_out_matrix.tocsr()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcgs = []
    for u in range(top_k_indices.shape[0]):
        relevant = set(held_out.indices[held_out.indptr[u] : held_out.indptr[u + 1]])
        if not relevant:
            continue
        gains = np.array([1.0 if item in relevant else 0.0 for item in top_k_indices[u, :k]])
        dcg = float(np.sum(gains * discounts[: len(gains)]))
        idcg = float(np.sum(discounts[: min(k, len(relevant))]))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def evaluate_model(
    model,
    n_users: int,
    exclude_matrix: sparse.csr_matrix,
    held_out_matrix: sparse.csr_matrix,
    k_list: list[int],
) -> dict[int, dict[str, float]]:
    """Evaluate a fitted recommender on a held-out matrix at each k in k_list."""
    k_max = max(k_list)
    top_k_idx, _ = model.recommend(n_users, k_max, exclude_matrix)

    results = {}
    for k in k_list:
        results[k] = {
            "recall": recall_at_k(top_k_idx, held_out_matrix, k),
            "ndcg": ndcg_at_k(top_k_idx, held_out_matrix, k),
        }
    return results
