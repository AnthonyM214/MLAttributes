"""Deterministic claim extraction from evidence rows and replay pages."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable

from .freshness import freshness_bonus, staleness_penalty
from .manifest import EvidenceItem, SOURCE_RANK
from .normalization import (
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
)
from .identity import identity_alignment_score
from .replay import FetchedPage, ReplayEpisode, SearchAttempt


_URL_PATTERN = re.compile(r"https?://[^\s<>'\")]+|www\.[^\s<>'\")]+", re.IGNORECASE)
_PHONE_PATTERN = re.compile(
    r"(?:\+?1[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]*\d{4}",
    re.IGNORECASE,
)
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way|Ct|Court|Pl|Place|Pkwy|Parkway|Hwy|Highway)\b",
    re.IGNORECASE,
)
_CATEGORY_TOKENS = (
    "restaurant",
    "cafe",
    "coffee",
    "hotel",
    "motel",
    "store",
    "shop",
    "clinic",
    "medical",
    "museum",
    "school",
    "bar",
    "bakery",
    "gym",
    "pharmacy",
    "bank",
    "gas station",
    "library",
    "salon",
)
_PAGE_RELEVANCE_KEYWORDS = {
    "place_page": ("contact", "about", "location", "locations", "directions", "hours", "menu", "visit us"),
    "contact_page": ("contact", "phone", "address", "hours"),
    "locator_page": ("locator", "locations", "find a store", "store locator", "branch"),
    "official_homepage": ("home", "welcome"),
    "registry_page": ("registry", "business", "license"),
    "aggregator_listing": ("yelp", "tripadvisor", "foursquare", "doordash", "ubereats"),
    "social_page": ("facebook", "instagram", "linkedin", "x.com", "twitter"),
}

class _StructuredHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.visible_parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._in_script = False
        self._script_type = ""
        self.json_ld_blocks: list[str] = []
        self.meta_fields: dict[str, list[str]] = {
            "url": [],
            "telephone": [],
            "name": [],
            "address": [],
            "type": [],
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        if tag.lower() == "title":
            self._in_title = True
        elif tag.lower() == "script":
            self._in_script = True
            self._script_type = attr_map.get("type", "").lower()
        elif tag.lower() == "meta":
            raw_field = (
                attr_map.get("property")
                or attr_map.get("name")
                or attr_map.get("itemprop")
                or ""
            ).strip().lower()
            content = attr_map.get("content", "").strip()
            if not raw_field or not content:
                return
            key = None
            if raw_field in {"url", "og:url", "twitter:url", "canonical"}:
                key = "url"
            elif raw_field in {"telephone", "phone", "og:phone_number", "contact:phone_number"}:
                key = "telephone"
            elif raw_field in {"name", "og:title", "title"}:
                key = "name"
            elif raw_field in {"address", "og:street-address", "street-address"}:
                key = "address"
            elif raw_field in {"@type", "type", "og:type", "business:type", "category"}:
                key = "type"
            if key is not None and content not in self.meta_fields[key]:
                self.meta_fields[key].append(content)
        elif tag.lower() == "link":
            rel = attr_map.get("rel", "").lower()
            href = attr_map.get("href", "").strip()
            if "canonical" in rel and href and href not in self.meta_fields["url"]:
                self.meta_fields["url"].append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() == "script":
            self._in_script = False
            self._script_type = ""

    def handle_data(self, data: str) -> None:
        text = html.unescape(data).strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        if self._in_script:
            if "ld+json" in self._script_type:
                self.json_ld_blocks.append(text)
            return
        self.visible_parts.append(text)


def _looks_like_html(text: str) -> bool:
    lowered = text.lower()
    return "<html" in lowered or "<body" in lowered or "<script" in lowered or "<meta" in lowered or "<title" in lowered


def _flatten_jsonld_value(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        if value.strip():
            values.append(value.strip())
    elif isinstance(value, dict):
        for key in ("url", "telephone", "name", "streetAddress", "addressLocality", "addressRegion", "postalCode", "addressCountry", "@type"):
            item = value.get(key)
            values.extend(_flatten_jsonld_value(item))
        for nested in value.values():
            values.extend(_flatten_jsonld_value(nested))
    elif isinstance(value, list):
        for item in value:
            values.extend(_flatten_jsonld_value(item))
    return values


def _extract_jsonld_fields(blocks: list[str]) -> dict[str, list[str]]:
    fields = {
        "url": [],
        "telephone": [],
        "name": [],
        "address": [],
        "type": [],
    }
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        payloads = payload if isinstance(payload, list) else [payload]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            for key in ("url", "telephone", "name", "@type"):
                raw = item.get(key)
                for value in _flatten_jsonld_value(raw):
                    if value not in fields["type" if key == "@type" else key]:
                        fields["type" if key == "@type" else key].append(value)
            address = item.get("address")
            if isinstance(address, dict):
                parts = []
                for key in ("streetAddress", "addressLocality", "addressRegion", "postalCode", "addressCountry"):
                    raw = address.get(key)
                    if isinstance(raw, str) and raw.strip():
                        parts.append(raw.strip())
                if parts:
                    value = ", ".join(parts)
                    if value not in fields["address"]:
                        fields["address"].append(value)
            elif isinstance(address, str) and address.strip():
                if address.strip() not in fields["address"]:
                    fields["address"].append(address.strip())
    return fields


def _structured_html_signals(page_text: str, page_title: str = "") -> tuple[str, str, list[str], dict[str, list[str]]]:
    parser = _StructuredHTMLParser()
    parser.feed(page_text)
    title = page_title.strip() or " ".join(parser.title_parts).strip()
    visible_text = "\n".join(parser.visible_parts).strip()
    jsonld_fields = _extract_jsonld_fields(parser.json_ld_blocks)
    for key, values in parser.meta_fields.items():
        for value in values:
            if value not in jsonld_fields[key]:
                jsonld_fields[key].append(value)
    return title, visible_text, parser.json_ld_blocks, jsonld_fields


@dataclass(frozen=True)
class AttributeClaim:
    claim_id: str
    place_id: str
    attribute: str
    value: str
    normalized_value: str
    source_url: str
    source_type: str
    extraction_method: str
    evidence_text: str
    page_title: str = ""
    query: str = ""
    page_relevance: str = "unknown"
    extraction_confidence: float = 0.0
    source_authority_score: float = 0.0
    freshness_score: float = 0.0
    stale_signal_score: float = 0.0
    identity_signal_score: float = 0.0
    notes: str = ""


def _claim_id(*parts: str) -> str:
    digest = hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _page_relevance(url: str, page_title: str, page_text: str, source_type: str) -> str:
    text = f"{page_title} {page_text}".lower()
    if source_type in {"aggregator", "social"}:
        return "aggregator_listing" if source_type == "aggregator" else "social_page"
    parsed_url = normalize_website(url)
    if any(token in text for token in ("moved", "relocated", "formerly known", "new location")):
        return "place_page"
    if any(token in parsed_url for token in ("contact", "about", "location", "locator", "branch", "store", "directions")):
        return "place_page"
    for label, tokens in _PAGE_RELEVANCE_KEYWORDS.items():
        if any(token in text for token in tokens):
            return label
    return "unknown"


def _extract_urls(text: str) -> list[str]:
    values: list[str] = []
    for match in _URL_PATTERN.findall(text or ""):
        candidate = match.strip(".,;:()[]{}<>")
        if candidate:
            values.append(normalize_website(candidate))
    return values


def _extract_url_contexts(text: str) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []
    for match in _URL_PATTERN.finditer(text or ""):
        candidate = match.group(0).strip(".,;:()[]{}<>")
        if not candidate:
            continue
        prefix = (text or "")[max(0, match.start() - 40): match.start()].lower()
        if "new location" in prefix or "new site" in prefix or "visit us at" in prefix:
            context = "new_location"
        elif "contact" in prefix or "website" in prefix:
            context = "contact"
        else:
            context = "source"
        contexts.append((normalize_website(candidate), context))
    return contexts


def _extract_phone_candidates(text: str) -> list[str]:
    values: list[str] = []
    for match in _PHONE_PATTERN.findall(text or ""):
        normalized = normalize_phone(match)
        if normalized:
            values.append(normalized)
    return values


def _extract_address_candidates(text: str) -> list[str]:
    values: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip(" \t-•")
        if not stripped:
            continue
        if _ADDRESS_PATTERN.search(stripped):
            values.append(normalize_address(stripped))
    return values


def _extract_name_candidate(page_title: str, text: str) -> str:
    if page_title.strip():
        return normalize_name(page_title)
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return normalize_name(stripped)
    return ""


def _extract_category_candidates(text: str) -> list[str]:
    values: list[str] = []
    lowered = (text or "").lower()
    if "schema.org/" in lowered or "localbusiness" in lowered:
        values.append("local business")
    for token in _CATEGORY_TOKENS:
        if token in lowered:
            values.append(normalize_category(token))
    return values


def _claim(
    *,
    place_id: str,
    attribute: str,
    value: str,
    source_url: str,
    source_type: str,
    extraction_method: str,
    evidence_text: str,
    place_context: dict[str, str] | None = None,
    page_title: str = "",
    query: str = "",
    page_relevance: str = "unknown",
    extraction_confidence: float = 0.0,
    source_authority_score: float | None = None,
    freshness_score: float = 0.0,
    stale_signal_score: float = 0.0,
    identity_signal_score: float = 0.0,
    notes: str = "",
) -> AttributeClaim:
    inferred_identity, inferred_stale = identity_alignment_score(
        place_context=place_context,
        attribute=attribute,
        value=value,
        source_url=source_url,
        evidence_text=evidence_text + "\n" + notes,
        page_title=page_title,
        source_type=source_type,
    )
    lowered_context = f"{evidence_text}\n{notes}\n{page_title}".lower()
    if "new_location" in extraction_method:
        inferred_stale = 0.0
        inferred_identity = max(inferred_identity, 0.9)
    elif source_type in {"official_site", "government"} and any(
        token in lowered_context for token in ("moved", "relocated", "formerly", "new location")
    ):
        inferred_stale = 0.0
        inferred_identity = max(inferred_identity, 0.9)
    normalized = {
        "website": normalize_website,
        "phone": normalize_phone,
        "address": normalize_address,
        "name": normalize_name,
        "category": normalize_category,
    }.get(attribute, lambda raw: (raw or "").strip().lower())(value)
    final_stale_signal = max(stale_signal_score, inferred_stale)
    if "new_location" in extraction_method or (
        source_type in {"official_site", "government"} and any(token in lowered_context for token in ("moved", "relocated", "formerly", "new location"))
    ):
        final_stale_signal = min(final_stale_signal, 0.15)

    return AttributeClaim(
        claim_id=_claim_id(place_id, attribute, normalized, source_url, extraction_method, evidence_text[:200]),
        place_id=place_id,
        attribute=attribute,
        value=value,
        normalized_value=normalized,
        source_url=source_url,
        source_type=source_type,
        extraction_method=extraction_method,
        evidence_text=evidence_text,
        page_title=page_title,
        query=query,
        page_relevance=page_relevance,
        extraction_confidence=max(0.0, min(1.0, extraction_confidence)),
        source_authority_score=max(
            0.0,
            min(1.0, source_authority_score if source_authority_score is not None else SOURCE_RANK.get(source_type, SOURCE_RANK["unknown"])),
        ),
        freshness_score=max(0.0, min(1.0, freshness_score)),
        stale_signal_score=max(0.0, min(1.0, final_stale_signal)),
        identity_signal_score=max(0.0, min(1.0, max(identity_signal_score, inferred_identity))),
        notes=notes,
    )

def extract_claims_from_text(
    *,
    place_id: str,
    attribute: str,
    page_text: str,
    source_url: str,
    source_type: str,
    page_title: str = "",
    query: str = "",
    place_context: dict[str, str] | None = None,
) -> list[AttributeClaim]:
    text = (page_text or "").strip()
    if not text and not page_title.strip() and not source_url.strip():
        return []

    structured_title = page_title
    structured_text = text
    jsonld_blocks: list[str] = []
    jsonld_fields: dict[str, list[str]] = {}
    if text and _looks_like_html(text):
        structured_title, structured_text, jsonld_blocks, jsonld_fields = _structured_html_signals(text, page_title=page_title)
        if structured_title and not page_title.strip():
            page_title = structured_title
        if structured_text:
            text = structured_text

    relevance = _page_relevance(source_url, page_title, text, source_type)
    source_score = SOURCE_RANK.get(source_type, SOURCE_RANK["unknown"])
    freshness = 0.0
    stale = 0.0
    identity, stale = identity_alignment_score(
        place_context=place_context,
        attribute=attribute,
        value="",
        source_url=source_url,
        evidence_text=text,
        page_title=page_title,
        source_type=source_type,
    )

    claims: list[AttributeClaim] = []
    if attribute == "website":
        url_contexts = _extract_url_contexts(text)
        if page_text:
            for value in re.findall(r"(?:https?://|www\.)[^\s<>'\")]+", page_text, flags=re.IGNORECASE):
                url_contexts.append((normalize_website(value), "text_url"))
        for value in jsonld_fields.get("url", []):
            if "://" in value or value.startswith("www."):
                url_contexts.append((normalize_website(value), "jsonld_url"))
        normalized_urls: list[str] = []
        for url, context in url_contexts:
            if url and url not in normalized_urls:
                normalized_urls.append(url)
        for idx, value in enumerate(normalized_urls):
            context = next((ctx for url, ctx in url_contexts if url == value), "text_url")
            extraction_method = "text_url"
            if context == "source_url":
                extraction_method = "source_url"
            elif context == "new_location":
                extraction_method = "text_url_new_location"
            elif context == "jsonld_url":
                extraction_method = "jsonld_url"
            claims.append(
                _claim(
                    place_id=place_id,
                    attribute=attribute,
                    value=value,
                    source_url=source_url,
                    source_type=source_type,
                    extraction_method=extraction_method if idx else extraction_method,
                    evidence_text=text,
                    place_context=place_context,
                    page_title=page_title,
                    query=query,
                    page_relevance=relevance,
                    extraction_confidence=0.95 if idx else 0.9,
                    source_authority_score=source_score,
                    freshness_score=freshness,
                    stale_signal_score=stale,
                    identity_signal_score=identity,
                )
            )
    elif attribute == "phone":
        seen: set[str] = set()
        for value in _extract_phone_candidates(text + "\n" + "\n".join(jsonld_fields.get("telephone", []))):
            if value in seen:
                continue
            seen.add(value)
            claims.append(
                _claim(
                    place_id=place_id,
                    attribute=attribute,
                    value=value,
                    source_url=source_url,
                    source_type=source_type,
                    extraction_method="phone_regex",
                    evidence_text=text,
                    place_context=place_context,
                    page_title=page_title,
                    query=query,
                    page_relevance=relevance,
                    extraction_confidence=0.88,
                    source_authority_score=source_score,
                    freshness_score=freshness,
                    stale_signal_score=stale,
                    identity_signal_score=identity,
                )
            )
    elif attribute == "address":
        seen = set()
        address_text = text + "\n" + "\n".join(jsonld_fields.get("address", []))
        for value in _extract_address_candidates(address_text):
            if value in seen:
                continue
            seen.add(value)
            claims.append(
                _claim(
                    place_id=place_id,
                    attribute=attribute,
                    value=value,
                    source_url=source_url,
                    source_type=source_type,
                    extraction_method="address_regex",
                    evidence_text=text,
                    place_context=place_context,
                    page_title=page_title,
                    query=query,
                    page_relevance=relevance,
                    extraction_confidence=0.84,
                    source_authority_score=source_score,
                    freshness_score=freshness,
                    stale_signal_score=stale,
                    identity_signal_score=identity,
                )
            )
    elif attribute == "name":
        candidates = []
        value = _extract_name_candidate(page_title, text)
        if value:
            candidates.append((value, 0.72 if page_title.strip() else 0.58, "title_or_first_line"))
        for jsonld_value in jsonld_fields.get("name", []):
            normalized_jsonld = normalize_name(jsonld_value)
            if normalized_jsonld and normalized_jsonld not in {candidate[0] for candidate in candidates}:
                candidates.append((normalized_jsonld, 0.82, "jsonld_name"))
        for value, confidence, method in candidates:
            claims.append(
                _claim(
                    place_id=place_id,
                    attribute=attribute,
                    value=value,
                    source_url=source_url,
                    source_type=source_type,
                    extraction_method=method,
                    evidence_text=text,
                    place_context=place_context,
                    page_title=page_title,
                    query=query,
                    page_relevance=relevance,
                    extraction_confidence=confidence,
                    source_authority_score=source_score,
                    freshness_score=freshness,
                    stale_signal_score=stale,
                    identity_signal_score=identity,
                )
            )
    elif attribute == "category":
        seen = set()
        for value in _extract_category_candidates(text + "\n" + "\n".join(jsonld_fields.get("type", []))):
            if value in seen:
                continue
            seen.add(value)
            claims.append(
                _claim(
                    place_id=place_id,
                    attribute=attribute,
                    value=value,
                    source_url=source_url,
                    source_type=source_type,
                    extraction_method="category_token",
                    evidence_text=text,
                    page_title=page_title,
                    query=query,
                    page_relevance=relevance,
                    extraction_confidence=0.78,
                    source_authority_score=source_score,
                    freshness_score=freshness,
                    stale_signal_score=stale,
                    identity_signal_score=identity,
                )
            )

    if jsonld_blocks and not claims:
        # JSON-LD blocks often contain structured info even when the visible page text is sparse.
        candidate_values = []
        if attribute == "website":
            candidate_values.extend(jsonld_fields.get("url", []))
        elif attribute == "phone":
            candidate_values.extend(jsonld_fields.get("telephone", []))
        elif attribute == "address":
            candidate_values.extend(jsonld_fields.get("address", []))
        elif attribute == "name":
            candidate_values.extend(jsonld_fields.get("name", []))
        elif attribute == "category":
            candidate_values.extend(jsonld_fields.get("type", []))
        for value in candidate_values:
            if not value:
                continue
            if attribute == "website" and ("." in value or value.startswith("http")):
                claims.append(
                    _claim(
                        place_id=place_id,
                        attribute=attribute,
                        value=value,
                        source_url=source_url,
                        source_type=source_type,
                        extraction_method="jsonld_url",
                        evidence_text="\n".join(jsonld_blocks),
                        place_context=place_context,
                        page_title=page_title,
                        query=query,
                        page_relevance=relevance,
                        extraction_confidence=0.88,
                        source_authority_score=source_score,
                        freshness_score=freshness,
                        stale_signal_score=stale,
                        identity_signal_score=identity,
                    )
                )
            elif attribute == "phone" and normalize_phone(value):
                claims.append(
                    _claim(
                        place_id=place_id,
                        attribute=attribute,
                        value=value,
                        source_url=source_url,
                        source_type=source_type,
                        extraction_method="jsonld_phone",
                        evidence_text="\n".join(jsonld_blocks),
                        place_context=place_context,
                        page_title=page_title,
                        query=query,
                        page_relevance=relevance,
                        extraction_confidence=0.9,
                        source_authority_score=source_score,
                        freshness_score=freshness,
                        stale_signal_score=stale,
                        identity_signal_score=identity,
                    )
                )
            elif attribute == "address" and any(token in normalize_address(value) for token in ("st", "ave", "rd", "blvd", "lane", "dr", "way", "court")):
                claims.append(
                    _claim(
                        place_id=place_id,
                        attribute=attribute,
                        value=value,
                        source_url=source_url,
                        source_type=source_type,
                        extraction_method="jsonld_address",
                        evidence_text="\n".join(jsonld_blocks),
                        place_context=place_context,
                        page_title=page_title,
                        query=query,
                        page_relevance=relevance,
                        extraction_confidence=0.9,
                        source_authority_score=source_score,
                        freshness_score=freshness,
                        stale_signal_score=stale,
                        identity_signal_score=identity,
                    )
                )
            elif attribute == "name" and normalize_name(value):
                claims.append(
                    _claim(
                        place_id=place_id,
                        attribute=attribute,
                        value=value,
                        source_url=source_url,
                        source_type=source_type,
                        extraction_method="jsonld_name",
                        evidence_text="\n".join(jsonld_blocks),
                        place_context=place_context,
                        page_title=page_title,
                        query=query,
                        page_relevance=relevance,
                        extraction_confidence=0.85,
                        source_authority_score=source_score,
                        freshness_score=freshness,
                        stale_signal_score=stale,
                        identity_signal_score=identity,
                    )
                )
            elif attribute == "category" and normalize_category(value):
                claims.append(
                    _claim(
                        place_id=place_id,
                        attribute=attribute,
                        value=value,
                        source_url=source_url,
                        source_type=source_type,
                        extraction_method="jsonld_type",
                        evidence_text="\n".join(jsonld_blocks),
                        place_context=place_context,
                        page_title=page_title,
                        query=query,
                        page_relevance=relevance,
                        extraction_confidence=0.82,
                        source_authority_score=source_score,
                        freshness_score=freshness,
                        stale_signal_score=stale,
                        identity_signal_score=identity,
                    )
                )

    return claims


def extract_claims_from_evidence_item(
    *,
    place_id: str,
    item: EvidenceItem,
    place_context: dict[str, str] | None = None,
) -> list[AttributeClaim]:
    value = str(item.extracted_value or "")
    if not value and item.attribute == "website" and item.url:
        value = item.url
    if not value:
        return []
    text = item.notes or value
    source_score = item.source_rank if item.source_rank is not None else SOURCE_RANK.get(item.source_type, SOURCE_RANK["unknown"])
    extraction_confidence = 0.9 if item.extracted_value else 0.7
    freshness = freshness_bonus(item.recency_days)
    stale = staleness_penalty(item.recency_days, item.zombie_score, item.identity_change_score)
    identity = max(0.0, min(1.0, 1.0 - min(1.0, float(item.identity_change_score or 0.0))))
    relevance = _page_relevance(item.url, "", item.notes or value, item.source_type)
    return [
        _claim(
            place_id=place_id,
            attribute=item.attribute,
            value=value,
            source_url=item.url,
            source_type=item.source_type,
            extraction_method="evidence_item",
            evidence_text=text,
            place_context=place_context,
            query=item.query,
            page_relevance=relevance,
            extraction_confidence=extraction_confidence,
            source_authority_score=source_score,
            freshness_score=freshness,
            stale_signal_score=stale,
            identity_signal_score=identity,
            notes=item.notes,
        )
    ]


def extract_claims_from_replay_episode(episode: ReplayEpisode) -> list[AttributeClaim]:
    claims: list[AttributeClaim] = []
    for attempt in episode.search_attempts:
        for page in attempt.fetched_pages:
            claims.extend(
                extract_claims_from_text(
                    place_id=episode.case_id,
                    attribute=episode.attribute,
                    page_text="\n".join(part for part in [page.page_text, page.notes] if part),
                    source_url=page.url,
                    source_type=page.source_type,
                    page_title=page.title,
                    query=attempt.query,
                    place_context=episode.place,
                )
            )
            if page.extracted_values.get(episode.attribute):
                claims.append(
                    _claim(
                        place_id=episode.case_id,
                        attribute=episode.attribute,
                        value=page.extracted_values[episode.attribute],
                        source_url=page.url,
                        source_type=page.source_type,
                        extraction_method="page_extracted_value",
                        evidence_text=page.page_text or page.title,
                        place_context=episode.place,
                        page_title=page.title,
                        query=attempt.query,
                        page_relevance=_page_relevance(page.url, page.title, page.page_text, page.source_type),
                        extraction_confidence=0.95,
                        source_authority_score=SOURCE_RANK.get(page.source_type, SOURCE_RANK["unknown"]),
                        freshness_score=freshness_bonus(page.recency_days),
                        stale_signal_score=staleness_penalty(page.recency_days, page.zombie_score, page.identity_change_score),
                        identity_signal_score=max(0.0, min(1.0, 1.0 - min(1.0, float(page.identity_change_score or 0.0)))),
                        notes=page.notes,
                    )
                )
    return claims


def claims_to_values(claims: Iterable[AttributeClaim]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if not claim.normalized_value or claim.normalized_value in seen:
            continue
        seen.add(claim.normalized_value)
        values.append(claim.value)
    return values
