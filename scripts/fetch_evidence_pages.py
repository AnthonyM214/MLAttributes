#!/usr/bin/env python3
"""Populate evidence CSV title/page_text fields from public URLs.

This is intentionally small and low-volume: it only reads explicit evidence
CSV rows and writes the same schema back with fetch status in notes.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


def _clean_text(value: str) -> str:
    value = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", value or "")
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _fetch(url: str, timeout: float) -> dict[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MLAttributes public evidence replay/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(32768)
            text = raw.decode(response.headers.get_content_charset() or "utf-8", errors="replace")
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
            return {
                "status": "ok",
                "http_status": str(getattr(response, "status", 200)),
                "title": _clean_text(title_match.group(1))[:180] if title_match else urlparse(url).netloc,
                "page_text": _clean_text(text)[:1200],
                "error": "",
            }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": "error",
            "http_status": "",
            "title": urlparse(url).netloc,
            "page_text": "",
            "error": str(exc)[:220],
        }


def _update_notes(notes: str, result: dict[str, str]) -> str:
    notes = re.sub(r";? ?fetch_status=[^;]+", "", notes or "").strip("; ")
    notes = re.sub(r";? ?http_status=[^;]+", "", notes).strip("; ")
    notes = re.sub(r";? ?fetch_error=[^;]+", "", notes).strip("; ")
    notes = re.sub(r";? ?curl_error=[^;]+", "", notes).strip("; ")
    parts = [part for part in [notes, f"fetch_status={result['status']}"] if part]
    if result.get("http_status"):
        parts.append(f"http_status={result['http_status']}")
    if result.get("error"):
        parts.append(f"fetch_error={result['error']}")
    return "; ".join(parts)


def populate_csv(path: Path, timeout: float, workers: int) -> dict[str, int | str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    urls = sorted({row.get("url", "") for row in rows if row.get("url", "")})
    results: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, url, timeout): url for url in urls}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    ok = 0
    for row in rows:
        url = row.get("url", "")
        if not url:
            continue
        result = results[url]
        if result["status"] == "ok":
            ok += 1
            row["title"] = result["title"] or row.get("title", "")
            row["page_text"] = result["page_text"] or row.get("page_text", "")
        elif not row.get("page_text", ""):
            row["page_text"] = f"Public website URL evidence: {url}"
        row["notes"] = _update_notes(row.get("notes", ""), result)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path), "urls": len(urls), "fetch_ok": ok, "fetch_error": len(urls) - ok}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="Evidence CSV file(s) to update in place.")
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for csv_path in args.csv:
        print(populate_csv(Path(csv_path), timeout=args.timeout, workers=args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
