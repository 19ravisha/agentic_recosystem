# Decisions Log

## D1: Dataset and problem framing
- **Dataset:** Kaggle "Amazon - Ratings (Beauty Products)" (`ratings_Beauty.csv`):
  ~2.02M ratings, ~1.21M users, ~249K products, columns `UserId, ProductId,
  Rating, Timestamp`. Chosen over the full Amazon Reviews 2023 corpus because
  it's large enough to show real cold-start/popularity-bias effects but small
  enough to train and evaluate fully on a local machine.
- **Feedback type:** Implicit. Ratings >= 4 are treated as a positive
  interaction (`data.positive_rating_threshold` in `configs/config.yaml`).
  Caveat: ratings are a proxy for purchases and biased toward users who review.
- **Task:** Top-N recommendation evaluated with full ranking, not sampled metrics.
- **Split:** Global temporal split (train/val/test by timestamp) to avoid leakage.
- **Filtering:** k-core threshold finalized at min 5 interactions per user and
  per item (`data.min_user_interactions` / `min_item_interactions`), applied to
  the train split only. Cold items (absent from the filtered train vocab) are
  kept in val/test rather than silently dropped, and their volume is reported
  as a visibility metric. A pure CF model with no content features can never
  recommend them, so Recall/NDCG against that slice would always be an empty
  (structurally-zero) metric, not an informative one — see D2.
- **Open risks:** this dataset is extremely sparse (avg ~1.67 ratings/user), so
  k-core filtering removes most users/items; popularity bias; review-vs-purchase
  gap; no item metadata (title/brand/price) in this dataset, so content-based
  features and cold-start mitigation beyond popularity aren't available yet.

## D2: Model choice for first implementation
- **Candidate considered:** EASE (named first in the README's planned
  architecture) — a closed-form item-item linear model.
- **Why rejected for v1:** EASE requires inverting a dense (n_items x n_items)
  Gram matrix. At this dataset's ~249K raw products, that's infeasible in
  memory even after k-core filtering likely leaves a smaller vocabulary; the
  risk of a local-machine OOM wasn't worth it for a first pass.
- **Decision:** Use `sklearn.decomposition.TruncatedSVD` matrix factorization
  on the implicit interaction matrix instead, scored in user batches to bound
  memory (see `src/recsys/models.py`). No new dependencies (scikit-learn is
  already required). A `PopularityRecommender` baseline is trained alongside
  it for comparison and as a cold-start fallback.
- **Future work:** EASE/iALS/SASRec/two-tower remain on the roadmap (see
  `CLAUDE.md` backlog) once we can evaluate them against a known, bounded
  item vocabulary size.
