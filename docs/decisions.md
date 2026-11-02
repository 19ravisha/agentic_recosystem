# Decisions Log

## D1: Dataset and problem framing
- **Dataset:** Amazon Reviews 2023, single category: <fill in category>.
  Chosen for rich item metadata (title, description, brand, price, category)
  that supports content features, cold-start handling, and agent constraints.
- **Feedback type:** Implicit. Any review is treated as a positive interaction
  (optionally rating >= 4). Caveat: reviews are a proxy for purchases and are
  biased toward users who review.
- **Task:** Top-N recommendation evaluated with full ranking, not sampled metrics.
- **Split:** Global temporal split (train/val/test by timestamp) to avoid leakage.
- **Filtering:** <k-core threshold, to be set on Day 2>. Cold items are kept in
  the test period for a dedicated cold-item slice.
- **Planned second dataset:** H&M transactions, to show generalization
  (stretch goal, Week 4 or 6).
- **Open risks:** popularity bias, review-vs-purchase gap, price/brand fields
  missing for some items.
