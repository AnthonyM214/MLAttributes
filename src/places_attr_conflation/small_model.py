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

from .dorking import classify_source


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _normalize_text(value: str | None) -> str:
    return (value or "").lower().strip()


def _bucket_recency(recency_days: float | None) -> dict[str, float]:
    if recency_days is None:
        return {}
    days = max(0.0, float(recency_days))
    features: dict[str, float] = {}
    for name, threshold in (("recent_30", 30), ("recent_90", 90), ("recent_180", 180), ("recent_365", 365)):
        features[f"recency:{name}"] = 1.0 if days <= threshold else 0.0
    return features


def _path_features(path: str) -> dict[str, float]:
    lowered = (path or "").lower()
    features: dict[str, float] = {}
    for token in ("contact", "about", "locations", "location", "store-locator", "store locator", "directions", "hours", "menu", "services", "home"):
        features[f"path:{token.replace(' ', '_')}"] = 1.0 if token in lowered else 0.0
    features["path:root"] = 1.0 if lowered in {"", "/", "/index.html", "/index.htm"} else 0.0
    return features


def build_conflict_row_feature_vector(row: Mapping[str, object]) -> dict[str, float]:
    """Turn a conflict dork workplan row into a tiny-model feature vector."""

    attribute = _normalize_text(str(row.get("attribute", "")))
    priority = _normalize_text(str(row.get("priority", "")))
    layer = _normalize_text(str(row.get("layer", "")))
    query = _normalize_text(str(row.get("query", "")))
    preferred_sources = _normalize_text(str(row.get("preferred_sources", "")))
    current_value = _normalize_text(str(row.get("current_value", "")))
    base_value = _normalize_text(str(row.get("base_value", "")))
    truth_value = _normalize_text(str(row.get("truth", "")))
    truth_source = _normalize_text(str(row.get("truth_source", "")))
    prediction = _normalize_text(str(row.get("prediction", "")))

    def _domain_from_value(value: str) -> str:
        if not value:
            return ""
        parsed = urlparse(value if "://" in value else f"https://{value}")
        domain = parsed.netloc.lower().removeprefix("www.")
        return domain if domain and "." in domain else ""

    def _looks_like_url(value: str) -> float:
        return 1.0 if value.startswith(("http://", "https://")) else 0.0

    def _looks_like_domain(value: str) -> float:
        return 1.0 if value and "." in value and " " not in value else 0.0

    current_domain = _domain_from_value(current_value)
    base_domain = _domain_from_value(base_value)
    truth_domain = _domain_from_value(truth_value)

    features: dict[str, float] = {
        f"attr:{attribute}": 1.0 if attribute else 0.0,
        f"priority:{priority}": 1.0 if priority else 0.0,
        f"layer:{layer}": 1.0 if layer else 0.0,
        "query:official": 1.0 if "official" in query else 0.0,
        "query:site_restricted": 1.0 if "site:" in query else 0.0,
        "query:quoted_anchor": 1.0 if query.count('"') >= 2 else 0.0,
        "query:authority": 1.0 if any(token in query for token in ("contact", "about", "locations", "store", "schema.org", "registry", "permit", "license")) else 0.0,
        "query:freshness": 1.0 if any(token in query for token in ("current", "updated", "hours", "open now", "now open", "latest")) else 0.0,
        "query:fallback": 1.0 if layer == "fallback" else 0.0,
        "source_pref:official_site": 1.0 if "official_site" in preferred_sources else 0.0,
        "source_pref:government": 1.0 if "government" in preferred_sources else 0.0,
        "source_pref:registry": 1.0 if "business_registry" in preferred_sources else 0.0,
        "source_pref:corroboration": 1.0 if "google_places" in preferred_sources or "osm" in preferred_sources else 0.0,
        "value:current_url_like": _looks_like_url(current_value),
        "value:base_url_like": _looks_like_url(base_value),
        "value:current_domain_like": _looks_like_domain(current_value),
        "value:base_domain_like": _looks_like_domain(base_value),
        "value:current_equals_base": 1.0 if current_value and current_value == base_value else 0.0,
        "value:truth_source_present": 1.0 if truth_source else 0.0,
        "value:prediction_present": 1.0 if prediction else 0.0,
        "value:current_len_norm": 1.0 / (1.0 + float(len(current_value))),
        "value:base_len_norm": 1.0 / (1.0 + float(len(base_value))),
        "domain:current_present": 1.0 if current_domain else 0.0,
        "domain:base_present": 1.0 if base_domain else 0.0,
        "domain:truth_present": 1.0 if truth_domain else 0.0,
        "domain:query_current": 1.0 if current_domain and current_domain in query else 0.0,
        "domain:query_base": 1.0 if base_domain and base_domain in query else 0.0,
        "domain:query_truth": 1.0 if truth_domain and truth_domain in query else 0.0,
        "domain:query_any_candidate": 1.0 if any(domain and domain in query for domain in (current_domain, base_domain, truth_domain)) else 0.0,
    }
    return {name: float(value) for name, value in features.items()}


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
    source_type = classify_source(url)
    parsed = urlparse(url if "://" in url else f"https://{url}")
    domain = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    query_text = _normalize_text(query)

    features: dict[str, float] = {
        f"source:{source_type}": 1.0,
        f"layer:{layer}": 1.0,
        "url:official_site": 1.0 if source_type == "official_site" else 0.0,
        "url:government": 1.0 if source_type == "government" else 0.0,
        "url:registry": 1.0 if source_type == "business_registry" else 0.0,
        "url:aggregator": 1.0 if source_type == "aggregator" else 0.0,
        "url:social": 1.0 if source_type == "social" else 0.0,
        "url:site_restricted": 1.0 if domain and f"site:{domain}" in query_text else 0.0,
        "text:contact": 1.0 if "contact" in text else 0.0,
        "text:about": 1.0 if "about" in text else 0.0,
        "text:hours": 1.0 if "hours" in text else 0.0,
        "text:menu": 1.0 if "menu" in text else 0.0,
        "text:services": 1.0 if "services" in text else 0.0,
        "text:schema": 1.0 if "schema.org" in text or "ld+json" in text else 0.0,
        "text:official": 1.0 if "official" in text or "official" in query_text else 0.0,
        "text:current": 1.0 if "current" in text else 0.0,
        "text:updated": 1.0 if "updated" in text or "last updated" in text else 0.0,
        "text:stale": 1.0 if any(token in text for token in ("former", "formerly", "moved", "closed", "directory", "listing")) else 0.0,
        "text:review": 1.0 if "review" in text or "reviews" in text else 0.0,
        "text:directory": 1.0 if "directory" in text or "listing" in text else 0.0,
        "query:match": 1.0 if query and query_text in text else 0.0,
        "zombie_score": zombie_score,
        "identity_change_score": identity_change_score,
    }
    features.update(_path_features(path))
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
