from dataclasses import dataclass, field
from typing import Literal

@dataclass
class CoreConfig:
    n_factors: int
    n_iters: int
    lambda_u: float
    lambda_v: float
    pop_reg_mode: Literal["inverse_sqrt"] | None = None
    random_state: int = 42
    update_w_every: int = 5

@dataclass
class BiasesConfig:
    lambda_bu: float | None = None
    lambda_bi: float | None = None

@dataclass
class GraphSimConfig:
    # how to build the similarity graph
    source: Literal["feature", "precomputed"] = "feature"
    feature_name: str = "genres"          # which feature to use if source="feature"
    metric: Literal["cosine"] = "cosine"  # keeping it simple for now
    topk: int | None = 50
    eps: float = 1e-8

@dataclass
class GraphConfig:
    alpha: float = 0.0                    # laplacian strength
    sim: GraphSimConfig | None = None     # None ⇒ no similarity graph

@dataclass
class ALSConfig:
    core: CoreConfig
    # ⚠️ nested dataclasses must use default_factory (NOT plain defaults)
    biases: BiasesConfig = field(default_factory=BiasesConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

def config_from_legacy_params(params: dict) -> ALSConfig:
    core = CoreConfig(
        n_factors=int(params.get("n_factors")),
        n_iters=int(params.get("n_iters")),
        lambda_u=float(params.get("lambda_u")),
        lambda_v=float(params.get("lambda_v")),
        pop_reg_mode=params.get("pop_reg_mode"),
        random_state=int(params.get("random_state", 42)),
        update_w_every=int(params.get("update_w_every", 5)),
    )
    biases = BiasesConfig(
        lambda_bu=float(params["lambda_bu"]) if params.get("lambda_bu") is not None else None,
        lambda_bi=float(params["lambda_bi"]) if params.get("lambda_bi") is not None else None,
    )

    alpha = float(params.get("alpha", 0.0) or 0.0)
    sim: GraphSimConfig | None = None
    # only build a sim config if we actually have graph params
    if alpha > 0.0 and params.get("S_topk") is not None:
        sim = GraphSimConfig(
            source="feature",
            feature_name="genres",
            metric="cosine",
            topk=int(params.get("S_topk")),
            eps=float(params.get("S_eps", 1e-8)),
        )

    graph = GraphConfig(alpha=alpha, sim=sim)
    return ALSConfig(core=core, biases=biases, graph=graph)
