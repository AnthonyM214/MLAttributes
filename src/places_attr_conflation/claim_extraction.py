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
_ADDRESS_TOKEN = r"(?=[A-Za-z0-9.'-]*[A-Za-z])[A-Za-z0-9.'-]+"
_ADDRESS_PATTERN = re.compile(
    rf"\b\d{{1,5}}\s+{_ADDRESS_TOKEN}(?:\s+{_ADDRESS_TOKEN}){{0,4}}\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way|Ct|Court|Pl|Place|Pkwy|Parkway|Hwy|Highway)\b",
    re.IGNORECASE,
)
_CAMPUS_ADDRESS_PATTERN = re.compile(
    rf"\b\d{{1,5}}\s+{_ADDRESS_TOKEN}(?:\s+{_ADDRESS_TOKEN}){{0,4}}\s+"
    r"(?:Building|Hall|Center|Centre|Library|Services)\b",
    re.IGNORECASE,
)
_ADDRESS_CUE_PATTERN = re.compile(
    r"\b(?:mailing address|office address|physical address|street address|address|location|located at|visit us at)\b\s*:?",
    re.IGNORECASE,
)
_POSTAL_CODE_PATTERN = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_PLACE_CONTEXT_STOPWORDS = {
    "branch",
    "center",
    "city",
    "department",
    "library",
    "libraries",
    "office",
    "public",
    "santa",
    "services",
    "support",
    "uc",
    "ucsc",
    "university",
    "california",
    "cruz",
}
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
_CATEGORY_PHRASES = {
    "academic support": "academic support",
    "natural history museum": "museum",
    "tutoring center": "tutoring service",
    "tutoring services": "tutoring service",
    "tutoring sessions": "tutoring service",
}
_GENERIC_TITLE_SEGMENTS = {
    "about",
    "about us",
    "contact",
    "contact information",
    "contact us",
    "directions",
    "home",
    "hours",
    "location",
    "locations",
    "visit",
    "visit us",
}
_PAGE_RELEVANCE_KEYWORDS = {
    "place_page": ("contact", "about", "location", "locations", "directions", "hours", "menu", "visit us"),
    "contact_page": ("contact", "phone", "address", "hours"),
    "locator_page": ("locator", "locations", "find a store", "store locator", "branch"),
    "official_homepage": ("home", "welcome"),
    "registry_page": ("registry", "business", "license"),
    "aggregator_listing": ("yelp", "tripadvisor", "foursquare", "doordash", "ubereats"),
    "social_page": ("facebook", "instagram", "linkedin", "x.com", "twitter"),
}
_PRIMARY_PHONE_LABELS = {
    "phone",
    "main",
    "main phone",
    "main line",
    "contact phone",
    "general inquiries",
    "general business line",
    "police general business line",
}
_SECONDARY_PHONE_LABELS = {
    "anonymous tip line",
    "billing",
    "direct",
    "emergency",
    "fax",
    "hotline",
    "non emergency",
    "non-emergency",
    "property",
    "property section",
    "records",
    "records section",
    "tip line",
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


def _place_context_terms(place_context: dict[str, str] | None) -> set[str]:
    if not place_context:
        return set()
    raw_name = place_context.get("name", "")
    city = set(normalize_name(place_context.get("city", "")).split())
    region = set(normalize_name(place_context.get("region", "")).split())
    terms = set()
    for token in normalize_name(raw_name).split():
        if len(token) < 4 or token in _PLACE_CONTEXT_STOPWORDS or token in city or token in region:
            continue
        terms.add(token)
    return terms


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for line in (text or "").splitlines(keepends=True):
        start = offset
        end = offset + len(line)
        spans.append((start, end, line.strip()))
        offset = end
    return spans


def _line_context_for_span(text: str, start: int, end: int, *, before: int = 4, after: int = 1) -> str:
    spans = _line_spans(text)
    if not spans:
        return text[max(0, start - 120): min(len(text), end + 120)].strip()
    match_index = 0
    for idx, (line_start, line_end, _) in enumerate(spans):
        if line_start <= start < line_end or line_start < end <= line_end:
            match_index = idx
            break
    selected = spans[max(0, match_index - before): min(len(spans), match_index + after + 1)]
    return "\n".join(line for _, _, line in selected if line).strip()


def _phone_label_for_span(text: str, start: int, end: int) -> tuple[str, str]:
    spans = _line_spans(text)
    if not spans:
        return "", ""
    match_index = 0
    for idx, (line_start, line_end, _) in enumerate(spans):
        if line_start <= start < line_end or line_start < end <= line_end:
            match_index = idx
            break
    lines = [line for _, _, line in spans]
    current = lines[match_index].strip()
    previous = ""
    for prior in reversed(lines[:match_index]):
        if prior.strip():
            previous = prior.strip()
            break
    label_text = normalize_name(f"{previous} {current}")
    if any(label in label_text for label in _SECONDARY_PHONE_LABELS):
        return "secondary", previous or current
    if current.lower().lstrip().startswith(("p ", "p:", "p(")):
        return "secondary", current
    if normalize_name(previous) in _PRIMARY_PHONE_LABELS:
        return "primary", previous
    # Staff/person direct numbers often appear under a named person and title,
    # not under a generic "Phone" label. Keep them visible but weaker.
    nearby = "\n".join(line for line in lines[max(0, match_index - 2): match_index + 1] if line.strip())
    if "," in nearby and normalize_name(previous) not in _PRIMARY_PHONE_LABELS:
        return "secondary", previous or current
    return "", previous or current


def _context_addresses(place_context: dict[str, str] | None) -> list[str]:
    if not place_context:
        return []
    values: list[str] = []
    for key in ("address", "current_address", "base_address", "current_value", "base_value"):
        normalized = normalize_address(place_context.get(key, ""))
        if normalized and any(token in normalized for token in (" st ", " ave ", " rd ", " blvd ", " dr ", " way ")):
            values.append(normalized)
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _context_address_values(place_context: dict[str, str] | None) -> list[tuple[str, str, str]]:
    if not place_context:
        return []
    values: list[tuple[str, str, str]] = []
    for key in ("address", "current_address", "address_current", "current_value", "base_address", "address_base", "base_value"):
        raw = place_context.get(key, "")
        normalized = normalize_address(raw)
        if not normalized:
            continue
        if len(normalized.split()) < 3:
            continue
        values.append((key, raw, normalized))
    output: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for item in values:
        if item[2] not in seen:
            seen.add(item[2])
            output.append(item)
    return output


def _extract_context_address_matches(
    text: str,
    *,
    place_context: dict[str, str] | None = None,
) -> list[tuple[str, str, float, float, str]]:
    normalized_text = normalize_address(text)
    matches: list[tuple[str, str, float, float, str]] = []
    for role, raw, normalized in _context_address_values(place_context):
        if normalized not in normalized_text:
            continue
        confidence = 0.93 if role in {"address", "current_address", "address_current", "current_value"} else 0.82
        identity = 0.95 if role in {"address", "current_address", "address_current", "current_value"} else 0.82
        matches.append((raw, text, confidence, identity, f"context_address_role={role}"))
    return matches


def _row_matches_context_address(row: str, place_context: dict[str, str] | None) -> bool:
    row_address = normalize_address(row)
    if not row_address:
        return False
    for context_address in _context_addresses(place_context):
        if context_address in row_address or row_address in context_address:
            return True
    return False


def _is_branch_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.search(r"\d", stripped):
        return False
    if _PHONE_PATTERN.search(stripped) or _ADDRESS_PATTERN.search(stripped):
        return False
    normalized = normalize_name(stripped)
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) > 4:
        return False
    return bool(set(tokens) - _PLACE_CONTEXT_STOPWORDS)


def _branch_rows(text: str) -> list[tuple[str, str]]:
    lines = [line.strip(" \t-•#") for line in (text or "").splitlines()]
    headers = [idx for idx, line in enumerate(lines) if _is_branch_header(line)]
    rows: list[tuple[str, str]] = []
    for pos, start in enumerate(headers):
        end = headers[pos + 1] if pos + 1 < len(headers) else len(lines)
        header = lines[start]
        body = "\n".join(line for line in lines[start:end] if line)
        if body:
            rows.append((header, body))
    return rows


def _extract_branch_directory_phone_contexts(
    text: str,
    *,
    place_context: dict[str, str] | None = None,
) -> list[tuple[str, str, float, float, str, str]]:
    terms = _place_context_terms(place_context)
    if not terms:
        return []
    matches: list[tuple[str, str, str]] = []
    for header, row in _branch_rows(text):
        header_terms = set(normalize_name(header).split())
        if not (terms & header_terms):
            continue
        phones = _extract_phone_candidates(row)
        if len(phones) == 1:
            matches.append((phones[0], row, header))
    if len(matches) != 1:
        return []
    value, row, header = matches[0]
    if not _row_matches_context_address(row, place_context):
        return []
    return [
        (
            value,
            row,
            0.95,
            0.95,
            f"branch_directory_header={normalize_name(header)}; corroborator=address",
            "branch_directory_phone",
        )
    ]


def _extract_phone_contexts(
    text: str,
    *,
    place_context: dict[str, str] | None = None,
) -> list[tuple[str, str, float, float, str, str]]:
    matches = list(_PHONE_PATTERN.finditer(text or ""))
    terms = _place_context_terms(place_context)
    contexts: list[tuple[str, str, float, float, str, str]] = _extract_branch_directory_phone_contexts(text, place_context=place_context)
    for match in matches:
        normalized = normalize_phone(match.group(0))
        if not normalized:
            continue
        snippet = _line_context_for_span(text, match.start(), match.end())
        label_kind, label = _phone_label_for_span(text, match.start(), match.end())
        snippet_terms = set(normalize_name(snippet).split())
        hits = terms & snippet_terms
        if label_kind == "primary":
            contexts.append((normalized, snippet, 0.97, 0.92, f"phone_label={normalize_name(label)}", "phone_regex_primary"))
        elif label_kind == "secondary":
            contexts.append((normalized, snippet, 0.56, 0.48, f"secondary_phone_label={normalize_name(label)}", "phone_regex_secondary"))
        elif hits:
            contexts.append((normalized, snippet, 0.9, 0.85, f"place_context_terms={','.join(sorted(hits))}", "phone_regex"))
        else:
            contexts.append((normalized, snippet, 0.78, 0.6, "", "phone_regex"))
    return contexts


def _trim_address_candidate(value: str) -> str:
    value = value.strip(" \t-:;,.")
    postal = _POSTAL_CODE_PATTERN.search(value)
    if postal:
        value = value[: postal.end()]
    return value.strip(" \t-:;,.")


def _address_segments(line: str) -> list[str]:
    cues = list(_ADDRESS_CUE_PATTERN.finditer(line))
    if cues:
        return [line[cue.end():].strip() for cue in cues if line[cue.end():].strip()]
    return [line]


def _extract_address_candidates(text: str) -> list[str]:
    values: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip(" \t-•")
        if not stripped:
            continue
        for segment in _address_segments(stripped):
            for pattern in (_ADDRESS_PATTERN, _CAMPUS_ADDRESS_PATTERN):
                for match in pattern.finditer(segment):
                    candidate = _trim_address_candidate(segment[match.start():])
                    normalized = normalize_address(candidate)
                    if normalized:
                        values.append(normalized)
    return values


def _address_claim_overlaps_explicit(claim: AttributeClaim, explicit_claim: AttributeClaim) -> bool:
    if claim.attribute != "address" or claim.extraction_method != "address_regex":
        return False
    left = claim.normalized_value
    right = explicit_claim.normalized_value
    if not left or not right or left == right:
        return False
    return left in right or right in left


def _clean_title_name(page_title: str) -> str:
    segments = [
        normalize_name(segment)
        for segment in re.split(r"\s+(?:[-|–—:])\s+|[|–—]", page_title)
        if normalize_name(segment)
    ]
    specific = [segment for segment in segments if segment not in _GENERIC_TITLE_SEGMENTS]
    if specific:
        return max(specific, key=lambda segment: (len(segment.split()), len(segment)))
    return normalize_name(page_title)


def _extract_name_candidate(page_title: str, text: str) -> str:
    if page_title.strip():
        return _clean_title_name(page_title)
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
    for phrase, category in _CATEGORY_PHRASES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            values.append(category)
    for token in _CATEGORY_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
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
        phone_text = text + "\n" + "\n".join(jsonld_fields.get("telephone", []))
        for value, evidence_text, confidence, claim_identity, notes, extraction_method in _extract_phone_contexts(phone_text, place_context=place_context):
            key = f"{extraction_method}:{value}"
            if key in seen:
                continue
            seen.add(key)
            claim_source_score = source_score
            if extraction_method == "phone_regex_secondary":
                claim_source_score = source_score * 0.45
            claims.append(
                _claim(
                    place_id=place_id,
                    attribute=attribute,
                    value=value,
                    source_url=source_url,
                    source_type=source_type,
                    extraction_method=extraction_method,
                    evidence_text=evidence_text or text,
                    place_context=place_context,
                    page_title=page_title,
                    query=query,
                    page_relevance=relevance,
                    extraction_confidence=confidence,
                    source_authority_score=claim_source_score,
                    freshness_score=freshness,
                    stale_signal_score=stale,
                    identity_signal_score=max(identity, claim_identity),
                    notes=notes,
                )
            )
    elif attribute == "address":
        seen = set()
        address_text = text + "\n" + "\n".join(jsonld_fields.get("address", []))
        explicit_context_claims: list[AttributeClaim] = []
        for value, evidence_text, confidence, claim_identity, notes in _extract_context_address_matches(address_text, place_context=place_context):
            normalized = normalize_address(value)
            if not normalized or normalized in seen:
                continue
            claim = _claim(
                place_id=place_id,
                attribute=attribute,
                value=value,
                source_url=source_url,
                source_type=source_type,
                extraction_method="context_address_in_text",
                evidence_text=evidence_text or text,
                place_context=place_context,
                page_title=page_title,
                query=query,
                page_relevance=relevance,
                extraction_confidence=confidence,
                source_authority_score=source_score,
                freshness_score=freshness,
                stale_signal_score=stale,
                identity_signal_score=max(identity, claim_identity),
                notes=notes,
            )
            explicit_context_claims.append(claim)
            claims.append(claim)
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
            page_claims = extract_claims_from_text(
                place_id=episode.case_id,
                attribute=episode.attribute,
                page_text="\n".join(part for part in [page.page_text, page.notes] if part),
                source_url=page.url,
                source_type=page.source_type,
                page_title=page.title,
                query=attempt.query,
                place_context=episode.place,
            )
            explicit_claim = None
            if page.extracted_values.get(episode.attribute):
                explicit_claim = _claim(
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
            if explicit_claim is not None and episode.attribute == "address":
                page_claims = [
                    claim for claim in page_claims
                    if not _address_claim_overlaps_explicit(claim, explicit_claim)
                ]
            claims.extend(page_claims)
            if explicit_claim is not None:
                claims.append(explicit_claim)
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
