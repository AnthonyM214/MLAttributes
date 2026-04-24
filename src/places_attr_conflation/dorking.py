"""Targeted query generation and source classification for evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


AGGREGATOR_DOMAINS = {
    "yelp.com",
    "tripadvisor.com",
    "foursquare.com",
    "facebook.com",
    "instagram.com",
    "doordash.com",
    "ubereats.com",
    "grubhub.com",
}

GOVERNMENT_SUFFIXES = (".gov", ".ca.gov", ".nyc.gov")
GOOGLE_PLACES_DOMAINS = {"google.com", "maps.google.com"}
OSM_DOMAINS = {"openstreetmap.org", "osm.org"}


@dataclass(frozen=True)
class DorkQueryPlan:
    attribute: str
    loose: str
    targeted: list[str]
    preferred_sources: list[str]


@dataclass(frozen=True)
class DorkLayer:
    name: str
    queries: list[str]
    preferred_sources: list[str]


@dataclass(frozen=True)
class MultiLayerDorkPlan:
    attribute: str
    layers: list[DorkLayer]


def quoted(value: str | None) -> str:
    value = (value or "").strip()
    return f'"{value}"' if value else ""


def loose_query(place: dict[str, str]) -> str:
    return " ".join(filter(None, [place.get("name", ""), place.get("city", ""), place.get("region", "")])).strip()


def targeted_queries(place: dict[str, str], attribute: str) -> list[str]:
    name = quoted(place.get("name"))
    city = quoted(place.get("city"))
    region = quoted(place.get("region"))
    address = quoted(place.get("address"))
    phone = quoted(place.get("phone"))
    website = place.get("website", "")
    domain = urlparse(website if "://" in website else f"https://{website}").netloc if website else ""
    exclusions = "-site:yelp.com -site:tripadvisor.com -site:facebook.com -site:instagram.com"

    if attribute == "website":
        queries = [
            f"{name} {city} official website {exclusions}",
            f"{name} {address} {city} {exclusions}",
            f"inurl:locations {name} {city}",
        ]
    elif attribute == "phone":
        queries = [
            f"{phone} {name}",
            f"{name} {city} phone {exclusions}",
        ]
        if domain:
            queries.append(f"site:{domain} {phone}")
    elif attribute == "address":
        queries = [
            f"{name} {address}",
            f"{name} {city} address {exclusions}",
        ]
        if domain:
            queries.append(f"site:{domain} {address}")
    elif attribute == "category":
        queries = [
            f"{name} {city} services menu about {exclusions}",
            f"{name} {city} {region} category",
        ]
        if domain:
            queries.append(f"site:{domain} about OR services OR menu")
    elif attribute == "name":
        queries = [
            f"{address} {city} business name",
            f"{phone} {address}",
        ]
    else:
        queries = [loose_query(place)]
    return [query.strip() for query in queries if query.strip()]


def build_query_plan(place: dict[str, str], attribute: str) -> DorkQueryPlan:
    loose = loose_query(place)
    targeted = targeted_queries(place, attribute)
    preferred_sources = ["official_site", "government", "business_registry", "google_places", "osm", "social", "aggregator"]
    return DorkQueryPlan(attribute=attribute, loose=loose, targeted=targeted, preferred_sources=preferred_sources)


def build_multi_layer_plan(place: dict[str, str], attribute: str) -> MultiLayerDorkPlan:
    name = quoted(place.get("name"))
    city = quoted(place.get("city"))
    region = quoted(place.get("region"))
    address = quoted(place.get("address"))
    phone = quoted(place.get("phone"))
    website = place.get("website", "")
    domain = urlparse(website if "://" in website else f"https://{website}").netloc if website else ""

    official_layers = DorkLayer(
        name="official",
        queries=targeted_queries(place, attribute),
        preferred_sources=["official_site", "government", "business_registry"],
    )

    corroboration_queries: list[str] = []
    if attribute in {"website", "phone", "address"}:
        corroboration_queries.extend(
            [
                f"{name} {city} {region} official OR contact OR about",
                f"{name} {address} {city}",
                f"{phone} {name}" if phone else "",
                f"site:google.com/maps {name} {city}",
            ]
        )
    elif attribute == "category":
        corroboration_queries.extend(
            [
                f"{name} {city} services OR menu OR about",
                f"site:{domain} schema.org LocalBusiness" if domain else "",
                f"site:openstreetmap.org {name} {city}",
            ]
        )
    else:
        corroboration_queries.extend([f"{name} {city}", f"{address} {city}", f"site:google.com/maps {name}"])

    corroboration_layer = DorkLayer(
        name="corroboration",
        queries=[q.strip() for q in corroboration_queries if q.strip()],
        preferred_sources=["official_site", "google_places", "osm", "social"],
    )

    freshness_queries: list[str] = []
    if attribute in {"website", "phone", "address"}:
        freshness_queries.extend(
            [
                f"{name} {city} open now hours contact",
                f"{name} {city} updated contact hours",
                f"{name} {city} current address phone",
            ]
        )
    elif attribute == "category":
        freshness_queries.extend(
            [
                f"{name} {city} current menu services hours",
                f"{name} {city} updated about services",
                f"site:{domain} hours menu updated" if domain else "",
            ]
        )
    else:
        freshness_queries.extend([f"{name} {city} updated", f"{name} {city} current"])

    freshness_layer = DorkLayer(
        name="freshness",
        queries=[q.strip() for q in freshness_queries if q.strip()],
        preferred_sources=["official_site", "google_places", "osm"],
    )

    fallback_layer = DorkLayer(
        name="fallback",
        queries=[loose_query(place)],
        preferred_sources=["google_places", "osm", "social", "aggregator"],
    )

    return MultiLayerDorkPlan(
        attribute=attribute,
        layers=[official_layers, corroboration_layer, freshness_layer, fallback_layer],
    )


def classify_source(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    domain = parsed.netloc.lower().removeprefix("www.")
    if any(domain.endswith(suffix) for suffix in GOVERNMENT_SUFFIXES):
        return "government"
    if domain in GOOGLE_PLACES_DOMAINS or domain.endswith(".google.com"):
        path = parsed.path.lower()
        if "/maps" in path or "/place" in path or "/search" in path:
            return "google_places"
    if domain in OSM_DOMAINS or domain.endswith(".openstreetmap.org"):
        return "osm"
    if any(domain == agg or domain.endswith(f".{agg}") for agg in AGGREGATOR_DOMAINS):
        return "aggregator" if domain not in {"facebook.com", "instagram.com"} else "social"
    if domain:
        return "official_site"
    return "unknown"


def rank_source(url: str, page_text: str = "", query: str = "") -> float:
    """Return a coarse source authority score for a fetched page.

    The score is intentionally simple: it lets the resolver prefer official or
    government evidence, then use page-level attribute evidence to break ties.
    """
    source_type = classify_source(url)
    base = {
        "official_site": 1.0,
        "government": 0.96,
        "business_registry": 0.9,
        "google_places": 0.82,
        "osm": 0.7,
        "social": 0.45,
        "aggregator": 0.35,
        "unknown": 0.2,
    }.get(source_type, 0.2)

    text = (page_text or "").lower()
    bonus = 0.0
    if any(token in text for token in ("contact", "about", "locations", "store locator", "directions")):
        bonus += 0.03
    if any(token in text for token in ("phone", "tel", "address", "hours", "menu", "services")):
        bonus += 0.03
    if "schema.org" in text or "ld+json" in text:
        bonus += 0.04
    if any(token in text for token in ("current", "updated", "open now", "last updated", "copyright")):
        bonus += 0.02
    if any(token in text for token in ("reviews", "review", "directory", "listing", "aggregate")):
        bonus -= 0.03
    if query and query.lower() in text:
        bonus += 0.02

    return min(1.0, base + bonus)
