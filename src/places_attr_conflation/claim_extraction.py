from __future__ import annotations

import re

PHONE_RE = re.compile(r'(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')
URL_RE = re.compile(r'https?://[^\s\"\'<>]+')


SCHEMA_MARKERS = (
    'schema.org',
    'ld+json',
    'localbusiness',
)


def extract_phone(text: str) -> str:
    match = PHONE_RE.search(text or '')
    return match.group(0) if match else ''


def extract_website(text: str) -> str:
    match = URL_RE.search(text or '')
    return match.group(0) if match else ''


def has_schema_org(text: str) -> bool:
    lowered = (text or '').lower()
    return any(marker in lowered for marker in SCHEMA_MARKERS)


def extract_claims(text: str) -> dict[str, object]:
    return {
        'phone': extract_phone(text),
        'website': extract_website(text),
        'has_schema_org': has_schema_org(text),
    }
