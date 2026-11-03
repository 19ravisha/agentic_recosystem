# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**agentic-recsys** is an end-to-end product recommendation system built on the Amazon Reviews 2023 dataset. It combines classical and neural recommender models, a retrieval-to-ranking pipeline, and an LLM-powered agent layer for tool-calling, served via FastAPI.

**Current status:** Early-stage development (Week 1: evaluation foundations)

**Planned architecture:**
- **Candidate generation**: EASE, iALS, SASRec, two-tower
- **Retrieval**: FAISS-based dense retrieval
- **Ranking**: LightGBM ranker
- **Post-processing**: Diversity re-ranking
- **Agent layer**: Claude tool-calling for constraint handling and personalization

## Quick Start

```bash
# Install development dependencies
make install

# Run linting
make lint

# Format code
make format

# Run tests
make test

# Run a single test
pytest tests/test_utils.py::test_set_seed_is_reproducible -v
```

## Project Structure

```
├── src/recsys/          # Main package source code
│   ├── __init__.py      # Package exports
│   └── utils.py         # Shared utilities (seeding, reproducibility)
├── tests/               # Test suite
│   ├── test_utils.py    # Unit tests for utilities
│   └── test_smoke.py    # Smoke tests for package imports
├── configs/             # Configuration files
│   └── config.yaml      # Data paths, thresholds, hyperparameters
├── data/                # Data directory (raw/ and processed/ subdirs)
├── docs/                # Documentation
│   └── decisions.md     # ADR-style design decisions log
├── notebooks/           # Exploratory analysis and prototyping
├── Makefile             # Common development tasks
├── pyproject.toml       # Package configuration & dependencies
└── .pre-commit-config.yaml  # Pre-commit hooks (formatting, linting)
```

## Development Workflow

### Code Quality

- **Linter**: [Ruff](https://github.com/astral-sh/ruff) checks E (pycodestyle), F (Pyflakes), I (isort), B (flake8-bugbear), UP (pyupgrade)
- **Formatter**: Ruff format (enforces line length of 100)
- **Pre-commit hooks**: Run automatically to fix trailing whitespace, YAML format, and apply ruff fixes
- **Target Python version**: 3.11+

Install pre-commit hooks:
```bash
pre-commit install
```

### Testing

- **Framework**: pytest
- **Run all tests**: `make test` or `pytest`
- **Run specific test file**: `pytest tests/test_utils.py -v`
- **Run with coverage** (if coverage plugin installed): `pytest --cov=src/recsys`

Tests should live in `tests/` and follow the naming convention `test_*.py`.

## Key Design Decisions

See [docs/decisions.md](docs/decisions.md) for the decisions log (ADR format). Current entries:

- **D1 (Dataset & Problem Framing)**: Amazon Reviews 2023 single category, implicit feedback (reviews ≥4 stars as positives), global temporal train/val/test split to avoid leakage, k-core filtering (threshold TBD), dedicated cold-item evaluation slice.

## Configuration

Configuration is centralized in `configs/config.yaml`:
- `seed`: Reproducibility seed (default 42)
- `paths.raw`: Raw data directory
- `paths.processed`: Processed data directory
- `data.category`: Amazon product category (to be filled in)
- `data.positive_rating_threshold`: Rating threshold for positive feedback
- `data.min_user_interactions` / `data.min_item_interactions`: k-core filtering thresholds

Loading and overriding config is a future concern; for now, use YAML for documentation.

## Dependencies

**Core ML/Data stack:**
- numpy, pandas, scipy (numerical computing, dataframes, sparse matrices)
- scikit-learn (preprocessing, metrics, classical models like EASE)
- pyarrow (efficient columnar data handling)

**Development stack:**
- pytest (unit testing)
- ruff (linting & formatting)
- pre-commit (git hooks)

See `pyproject.toml` for exact versions and optional dependencies.

## Common Issues & Tips

1. **Reproducibility:** Always call `set_seed()` at the start of experiments. The `utils.set_seed()` function seeds Python's `random`, NumPy, and sets `PYTHONHASHSEED`.

2. **New model implementations:** Place in separate modules under `src/recsys/` (e.g., `models.py`, `candidates.py`, `rankers.py`, `agents.py`).

3. **Data loading & preprocessing:** Will eventually belong in a dedicated module (e.g., `src/recsys/data.py`). For now, prototype in notebooks.

4. **Agent layer:** Will integrate Claude API for tool-calling. Placeholder for future implementation.

## Next Steps (Backlog)

- Finalize k-core filtering thresholds and data splits (Week 1, Day 2)
- Implement data loading and preprocessing pipeline
- Implement candidate generation models (EASE, iALS, SASRec, two-tower)
- Integrate FAISS retrieval
- Implement LightGBM ranker
- Implement diversity re-ranking
- Agent layer with Claude tool use
- FastAPI service endpoint
- Optional: H&M dataset integration for generalization validation
