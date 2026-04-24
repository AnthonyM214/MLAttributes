"""Small trainable reranker for evidence and source authority.

The model is intentionally lightweight:
* pure Python
* no external dependencies
* usable as an optional reranker over search results

It is not meant to replace the heuristic path. It is meant to learn a better
authority preference from labeled examples when such data is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Iterable, Mapping
from urllib.parse import urlparse


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _normalize_text(value: str | None) -> str:
    return (value or "").lower().strip()


def _classify_source(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    domain = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()

    if any(domain.endswith(suffix) for suffix in (".gov", ".ca.gov", ".nyc.gov")):
        return "government"
    if domain in {"google.com", "maps.google.com"} and ("/maps" in path or "/place" in path or "/search" in path):
        return "google_places"
    if domain in {"openstreetmap.org", "osm.org"} or domain.endswith(".openstreetmap.org"):
        return "osm"
    if any(domain == agg or domain.endswith(f".{agg}") for agg in {"yelp.com", "tripadvisor.com", "foursquare.com", "doordash.com", "ubereats.com", "grubhub.com"}):
        return "aggregator"
    if domain in {"facebook.com", "instagram.com"} or domain.endswith(".facebook.com") or domain.endswith(".instagram.com"):
        return "social"
    if domain:
        return "official_site"
    return "unknown"


def _bucket_recency(recency_days: float | None) -> dict[str, float]:
    if recency_days is None:
        return {}
    days = max(0.0, float(recency_days))
    features: dict[str, float] = {}
    for name, threshold in (("recent_30", 30), ("recent_90", 90), ("recent_180", 180), ("recent_365", 365)):
        features[f"recency:{name}"] = 1.0 if days <= threshold else 0.0
    return features


def build_feature_vector(result: object, query: str = "", page_text: str = "") -> dict[str, float]:
    """Turn a search result-like object into a feature vector."""

    url = getattr(result, "url", "")
    title = getattr(result, "title", "")
    snippet = getattr(result, "snippet", "")
    layer = getattr(result, "layer", "fallback")
    recency_days = getattr(result, "recency_days", None)
    zombie_score = float(getattr(result, "zombie_score", 0.0) or 0.0)
    identity_change_score = float(getattr(result, "identity_change_score", 0.0) or 0.0)

    text = " ".join(part for part in [title, snippet, page_text] if part).lower()
    source_type = _classify_source(url)

    features: dict[str, float] = {
        f"source:{source_type}": 1.0,
        f"layer:{layer}": 1.0,
        "text:contact": 1.0 if "contact" in text else 0.0,
        "text:about": 1.0 if "about" in text else 0.0,
        "text:hours": 1.0 if "hours" in text else 0.0,
        "text:menu": 1.0 if "menu" in text else 0.0,
        "text:services": 1.0 if "services" in text else 0.0,
        "text:schema": 1.0 if "schema.org" in text or "ld+json" in text else 0.0,
        "text:current": 1.0 if "current" in text else 0.0,
        "text:updated": 1.0 if "updated" in text or "last updated" in text else 0.0,
        "text:review": 1.0 if "review" in text or "reviews" in text else 0.0,
        "text:directory": 1.0 if "directory" in text or "listing" in text else 0.0,
        "query:match": 1.0 if query and _normalize_text(query) in text else 0.0,
        "zombie_score": zombie_score,
        "identity_change_score": identity_change_score,
    }
    features.update(_bucket_recency(recency_days))
    if recency_days is not None:
        features["recency:days_norm"] = 1.0 / (1.0 + max(0.0, float(recency_days)))
    return features


@dataclass
class TrainingExample:
    features: dict[str, float]
    label: int


@dataclass
class TinyLinearModel:
    """A tiny logistic model trained with SGD."""

    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def score(self, features: Mapping[str, float]) -> float:
        logit = self.bias
        for name, value in features.items():
            logit += self.weights.get(name, 0.0) * float(value)
        return _sigmoid(logit)

    def predict(self, features: Mapping[str, float], threshold: float = 0.5) -> bool:
        return self.score(features) >= threshold

    def update(self, features: Mapping[str, float], label: int, learning_rate: float = 0.1, l2: float = 0.0) -> None:
        prediction = self.score(features)
        error = float(label) - prediction
        self.bias += learning_rate * error
        for name, value in features.items():
            weight = self.weights.get(name, 0.0)
            weight += learning_rate * (error * float(value) - l2 * weight)
            self.weights[name] = weight

    def fit(
        self,
        examples: Iterable[TrainingExample],
        epochs: int = 25,
        learning_rate: float = 0.1,
        l2: float = 0.0,
    ) -> "TinyLinearModel":
        materialized = list(examples)
        for _ in range(max(1, epochs)):
            for example in materialized:
                self.update(example.features, int(example.label), learning_rate=learning_rate, l2=l2)
        return self

    def score_result(self, result: object, query: str = "", page_text: str = "") -> float:
        return self.score(build_feature_vector(result, query=query, page_text=page_text))


def train_tiny_model(
    examples: Iterable[TrainingExample],
    epochs: int = 25,
    learning_rate: float = 0.1,
    l2: float = 0.0,
) -> TinyLinearModel:
    return TinyLinearModel().fit(examples, epochs=epochs, learning_rate=learning_rate, l2=l2)

