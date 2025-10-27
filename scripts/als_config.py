"""
Typed configuration objects for the ALS matrix-factorization model.

This module defines small dataclasses that hold validated configuration for
ALS matrix-factorization recommender.

## Public API:

- `CoreConfig`     : latent factors/iterations and core L2 regularization
- `BiasesConfig`   : user/item bias regularization
- `GraphConfig`    : Laplacian regularization (with `GraphSimConfig`)
- `ALSConfig`      : a simple container grouping the above

## Typical usage:

```python
from scripts.als_config import (
    ALSConfig, CoreConfig, BiasesConfig, GraphConfig, GraphSimConfig
    )
from scripts.als import ALS

cfg = ALSConfig(
    core=CoreConfig(
        n_factors=50,
        n_iters=20,
        lambda_u=10.0,
        lambda_v=10.0,
        pop_reg_mode="inverse_sqrt",
        random_state=42,
        update_w_every=5,
    ),
    biases=BiasesConfig(
        lambda_bu=5.0,
        lambda_bi=5.0
    ),
    graph=GraphConfig(
        alpha=0.5,
        sim=GraphSimConfig(
            source="feature",
            feature_name="genres",
            metric="cosine",
            topk=100,
            eps=1e-5,
        ),
    ),
)

model = ALS(
    lambda_w={"genres": 5.0, "years": 10.0},
    config=cfg
)
"""

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class CoreConfig:
    """Configuration for core ALS matrix factorization."""
    n_factors: int
    n_iters: int
    lambda_u: float
    lambda_v: float
    pop_reg_mode: Literal["inverse_sqrt"] | None = None
    random_state: int = 42
    update_w_every: int = 5

@dataclass
class BiasesConfig:
    """Configuration for user/item bias regularization."""
    lambda_bu: float | None = None
    lambda_bi: float | None = None

@dataclass
class GraphSimConfig:
    """Configuration for item-item similarity graph."""
    source: Literal["feature", "precomputed"] = "feature"
    feature_name: str = "genres"          # which feature to use if source="feature"
    metric: Literal["cosine"] = "cosine"  # keeping it simple for now
    topk: int | None = 50
    eps: float = 1e-8

@dataclass
class GraphConfig:
    """Configuration for graph Laplacian regularization."""
    alpha: float = 0.0                    # laplacian strength
    sim: GraphSimConfig | None = None     # None ⇒ no similarity graph

@dataclass
class ALSConfig:
    """Top-level configuration for ALS matrix factorization."""
    core: CoreConfig
    # nested dataclasses must use default_factory (NOT plain defaults)
    biases: BiasesConfig = field(default_factory=BiasesConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
