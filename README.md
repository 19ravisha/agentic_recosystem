# agentic_recosystem

# agentic-recsys

An end-to-end product recommendation system on Amazon Reviews 2023:
classical and neural recommenders, a retrieval-to-ranking pipeline,
and an LLM tool-calling agent, served via FastAPI.

## Status
Work in progress (Week 1: evaluation foundations).

## Planned architecture
Candidate generation (EASE, iALS, SASRec, two-tower) -> FAISS retrieval
-> LightGBM ranker -> diversity re-ranking -> agent layer (Claude tool use).

## Setup
    python -m venv .venv
    source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
    pip install -e ".[dev]"
    pytest

## Docs
See `docs/decisions.md` for design decisions.
