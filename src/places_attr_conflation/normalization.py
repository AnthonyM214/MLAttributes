"""Normalization helpers used before attribute conflict resolution."""

from __future__ import annotations

import re
from urllib.parse import urlparse


_SOCIAL_OR_AGGREGATOR_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "yelp.com",
    "tripadvisor.com",
    "doordash.com",
    "ubereats.com",
    "grubhub.com",
    "opentable.com",
    "foursquare.com",
}


def normalize_phone(value: str | None) -> str:
    """Return comparable US-style phone digits, preserving country code when present."""
    if not value:
        return ""
    digits = re.sub(r"\D+", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        return digits[1:]
    return digits


def normalize_website(value: str | None) -> str:
    """Return comparable host/path without scheme, www, query, fragment, or trailing slash."""
    if not value:
        return ""
    raw = value.strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    host = parsed.netloc.removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    return f"{host}{path}"


def website_domain(value: str | None) -> str:
    normalized = normalize_website(value)
    return normalized.split("/", 1)[0]


def is_social_or_aggregator(value: str | None) -> bool:
    domain = website_domain(value)
    return any(domain == blocked or domain.endswith(f".{blocked}") for blocked in _SOCIAL_OR_AGGREGATOR_DOMAINS)


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"&", " and ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_address(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower()
    replacements = {
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bboulevard\b": "blvd",
        r"\broad\b": "rd",
        r"\bsuite\b": "ste",
        r"\bcalifornia\b": "ca",
    }
    for pattern, replacement in replacements.items():
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_category(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip()

