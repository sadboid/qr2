"""Persistent OA paper library — the harvester behind the engine's learning layer.

Builds a local library of real Q1/Scopus papers by walking journals through
Crossref, then resolving each work to a LEGALLY downloadable full text
(Semantic Scholar openAccessPdf -> Unpaywall -> arXiv). Paywalled works are
never scraped: they are logged to needs_library_access.csv so a human can
fetch them through their institution's subscription.

Design notes
------------
* Discovery runs on Crossref (journal -> ISSN -> works, cursor-paged). OpenAlex
  would be the more natural index, but it rate-limits this environment's
  shared egress IP; Crossref and Semantic Scholar do not.
* Journals are resolved by NAME at runtime rather than from a hardcoded ISSN
  table — a wrong ISSN silently harvests the wrong journal with nothing in the
  output to reveal it.
* A publisher link is only followed when the work carries an open licence.
  Everything else goes to the human queue.
* Every request carries a mailto contact (Crossref/Unpaywall polite pools) and
  is rate limited; the crawl is resumable and never re-downloads a DOI. The
  manifest records licence, the PDF's origin and its SHA-256, so anything
  mined downstream traces back to a specific file.
"""

import csv
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

CROSSREF_JOURNALS = "https://api.crossref.org/journals"
UNPAYWALL = "https://api.unpaywall.org/v2/"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper/"
ARXIV_PDF = "https://arxiv.org/pdf/"

_MIN_PDF_BYTES = 20_000          # smaller than this is a landing page, not a paper
_DELAY = 1.1                     # polite pause between requests (S2 wants ~1/s)


@dataclass
class LibraryEntry:
    doi: str
    title: str
    authors: List[str]
    year: Optional[int]
    journal: str
    citations: int
    license: str = ""
    pdf_path: Optional[str] = None
    pdf_sha256: Optional[str] = None
    pdf_source: Optional[str] = None      # which resolver produced the file
    subjects: List[str] = field(default_factory=list)
    harvested_at: str = ""


def _slug(doi: str) -> str:
    return re.sub(r"[^\w.-]", "_", doi.replace("https://doi.org/", ""))[:120]


def _clean_doi(doi: Optional[str]) -> str:
    return (doi or "").replace("https://doi.org/", "").strip().lower()


def _is_open_licence(work: dict) -> bool:
    for lic in work.get("license") or []:
        url = (lic.get("URL") or "").lower()
        if "creativecommons.org" in url or "/open-access" in url:
            return True
    return False


def _from_crossref(work: dict) -> Optional[LibraryEntry]:
    doi = _clean_doi(work.get("DOI"))
    if not doi:
        return None
    date = (work.get("published") or work.get("issued") or {}).get("date-parts") or [[None]]
    year = date[0][0] if date and date[0] else None
    authors = []
    for a in (work.get("author") or [])[:12]:
        name = " ".join(x for x in [a.get("given"), a.get("family")] if x)
        if name:
            authors.append(name)
    licences = [(l.get("URL") or "") for l in (work.get("license") or [])]
    return LibraryEntry(
        doi=doi,
        title=(work.get("title") or [""])[0],
        authors=authors,
        year=year,
        journal=(work.get("container-title") or [""])[0],
        citations=work.get("is-referenced-by-count", 0),
        license=licences[0] if licences else "",
        subjects=(work.get("subject") or [])[:6],
        harvested_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


class OALibrary:
    """A resumable, provenance-tracked library of open-access papers."""

    def __init__(self, root: str, email: str):
        self.root = Path(root)
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.paywalled_path = self.root / "needs_library_access.csv"
        self.email = email
        self.entries: Dict[str, LibraryEntry] = {}
        self.paywalled: Dict[str, dict] = {}
        self.client = httpx.Client(
            follow_redirects=True, timeout=60,
            headers={"User-Agent": f"qr2-research-machine (mailto:{email})"},
        )
        self._load()

    # ---------------------------------------------------------------- state
    def _load(self):
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text())
            for d in raw.get("entries", []):
                self.entries[d["doi"]] = LibraryEntry(**d)
        if self.paywalled_path.exists():
            with open(self.paywalled_path) as f:
                for row in csv.DictReader(f):
                    self.paywalled[row["doi"]] = row
        if self.entries or self.paywalled:
            logger.info(
                f"[Library] Resumed: {len(self.entries)} works indexed "
                f"({self.n_with_pdf} with PDF), {len(self.paywalled)} paywalled"
            )

    def save(self):
        self.manifest_path.write_text(json.dumps({
            "count": len(self.entries),
            "with_pdf": self.n_with_pdf,
            "entries": [asdict(e) for e in self.entries.values()],
        }, indent=1))
        with open(self.paywalled_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["doi", "title", "journal", "year", "url"])
            w.writeheader()
            for row in self.paywalled.values():
                w.writerow(row)

    @property
    def n_with_pdf(self) -> int:
        return sum(1 for e in self.entries.values() if e.pdf_path)

    # ------------------------------------------------------------ transport
    def _get(self, url: str, params: Optional[dict] = None,
             retries: int = 4) -> Optional[httpx.Response]:
        """GET with polite backoff. The egress IP is shared, so 429s are
        routine rather than exceptional — honour Retry-After and keep going
        instead of dropping the record."""
        delay = 2.0
        for attempt in range(retries):
            try:
                r = self.client.get(url, params=params)
            except Exception as e:
                logger.debug(f"[Library] GET {url[:60]} failed ({e!r})")
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < retries - 1:
                    wait = min(float(r.headers.get("Retry-After") or delay), 60)
                    logger.debug(f"[Library] {r.status_code} — waiting {wait:.0f}s")
                    time.sleep(wait)
                    delay *= 2
                    continue
                return None
            time.sleep(_DELAY)
            return r
        return None

    # ------------------------------------------------------------ discovery
    def resolve_journal(self, name: str) -> Optional[dict]:
        """Resolve a journal NAME to a Crossref journal record (title + ISSNs)."""
        r = self._get(CROSSREF_JOURNALS, {
            "query": name, "rows": 5, "mailto": self.email,
        })
        if r is None or r.status_code != 200:
            return None
        try:
            items = r.json()["message"]["items"]
        except Exception:
            return None
        if not items:
            return None
        target = name.lower().strip()
        for j in items:                      # prefer an exact title match
            if (j.get("title") or "").lower().strip() == target:
                return j
        for j in items:
            if (j.get("title") or "").lower().startswith(target[:24]):
                return j
        return items[0]

    def iter_journal_works(
        self, issn: str, from_year: int, topic: Optional[str] = None,
        rows: int = 100, max_pages: int = 4,
    ) -> Iterable[dict]:
        """Page through a journal's articles, most-cited first."""
        params = {
            "filter": f"from-pub-date:{from_year}-01-01,type:journal-article",
            "sort": "is-referenced-by-count", "order": "desc",
            "rows": rows, "cursor": "*", "mailto": self.email,
        }
        if topic:
            params["query.bibliographic"] = topic
        for _ in range(max_pages):
            r = self._get(f"{CROSSREF_JOURNALS}/{issn}/works", params)
            if r is None or r.status_code != 200:
                logger.warning(f"[Library] works page unavailable for ISSN {issn}")
                return
            try:
                msg = r.json()["message"]
            except Exception:
                return
            items = msg.get("items", [])
            if not items:
                return
            yield from items
            cursor = msg.get("next-cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # -------------------------------------------------------------- fetching
    def _s2_oa_pdf(self, doi: str) -> Optional[str]:
        r = self._get(f"{S2_PAPER}DOI:{doi}", {"fields": "openAccessPdf"}, retries=2)
        if r is None or r.status_code != 200:
            return None
        try:
            oa = r.json().get("openAccessPdf") or {}
            return oa.get("url")
        except Exception:
            return None

    def _unpaywall_pdf(self, doi: str) -> Optional[str]:
        r = self._get(f"{UNPAYWALL}{doi}", {"email": self.email}, retries=2)
        if r is None or r.status_code != 200:
            return None
        try:
            loc = r.json().get("best_oa_location") or {}
            return loc.get("url_for_pdf") or loc.get("url")
        except Exception:
            return None

    def _pdf_candidates(self, work: dict, doi: str) -> List[Tuple[str, str]]:
        """Ordered (url, source_label) candidates for a LEGAL full text."""
        out: List[Tuple[str, str]] = []
        url = self._s2_oa_pdf(doi)
        if url:
            out.append((url, "semantic_scholar_oa"))
        url = self._unpaywall_pdf(doi)
        if url:
            out.append((url, "unpaywall"))
        # Publisher links only when the work itself carries an open licence —
        # Crossref also lists TDM links for paywalled content, which are not
        # ours to fetch.
        if _is_open_licence(work):
            for link in work.get("link") or []:
                if "pdf" in (link.get("content-type") or "").lower() and link.get("URL"):
                    out.append((link["URL"], "publisher_open_licence"))
        seen, uniq = set(), []
        for u, label in out:
            if u not in seen:
                seen.add(u)
                uniq.append((u, label))
        return uniq

    def _download(self, url: str, dest: Path) -> Optional[bytes]:
        r = self._get(url, retries=2)
        if r is None or r.status_code != 200:
            return None
        data = r.content
        if not data[:5].startswith(b"%PDF") or len(data) < _MIN_PDF_BYTES:
            return None
        dest.write_bytes(data)
        return data

    def add_work(self, work: dict) -> Optional[LibraryEntry]:
        """Record a Crossref work and try to fetch its legal full text."""
        entry = _from_crossref(work)
        if entry is None:
            return None
        prior = self.entries.get(entry.doi)
        if prior and prior.pdf_path:
            return prior                      # resume path — already have it

        dest = self.pdf_dir / f"{_slug(entry.doi)}.pdf"
        if dest.exists() and dest.stat().st_size >= _MIN_PDF_BYTES:
            data = dest.read_bytes()
            entry.pdf_path = str(dest.relative_to(self.root))
            entry.pdf_sha256 = hashlib.sha256(data).hexdigest()
            entry.pdf_source = "cached"
        else:
            for url, label in self._pdf_candidates(work, entry.doi):
                data = self._download(url, dest)
                if data:
                    entry.pdf_path = str(dest.relative_to(self.root))
                    entry.pdf_sha256 = hashlib.sha256(data).hexdigest()
                    entry.pdf_source = label
                    break

        self.entries[entry.doi] = entry
        if not entry.pdf_path:
            # No legal copy reachable — hand it to the human, do not scrape.
            self.paywalled[entry.doi] = {
                "doi": entry.doi, "title": entry.title, "journal": entry.journal,
                "year": entry.year or "", "url": f"https://doi.org/{entry.doi}",
            }
        return entry

    def harvest_dois(self, dois: List[str]) -> dict:
        """Harvest specific DOIs (rebuilding a seed set, chasing a reference
        list, filling a gap). Metadata comes from Crossref so the entry is the
        same shape as a journal-walk entry."""
        stats = {"requested": len(dois), "indexed": 0, "pdfs": 0, "failed": []}
        for raw in dois:
            doi = _clean_doi(raw)
            if not doi:
                continue
            prior = self.entries.get(doi)
            if prior and prior.pdf_path:
                stats["indexed"] += 1
                stats["pdfs"] += 1
                continue
            r = self._get(f"https://api.crossref.org/works/{doi}", {"mailto": self.email})
            if r is None or r.status_code != 200:
                stats["failed"].append(doi)
                continue
            try:
                work = r.json()["message"]
            except Exception:
                stats["failed"].append(doi)
                continue
            entry = self.add_work(work)
            if entry:
                stats["indexed"] += 1
                if entry.pdf_path:
                    stats["pdfs"] += 1
                    logger.info(f"[Library]   PDF via {entry.pdf_source}: {entry.title[:55]}")
                else:
                    logger.info(f"[Library]   paywalled: {entry.title[:55]}")
            self.save()
        self.save()
        logger.info(f"[Library] DOI harvest: {stats['pdfs']}/{stats['requested']} full texts")
        return stats

    def harvest_journals(
        self, journal_names: List[str], from_year: int = 2018,
        topic: Optional[str] = None, per_journal: int = 25,
        scan_cap: int = 400,
    ) -> dict:
        """Walk a list of journals and harvest their legally available articles.

        per_journal caps PDFs kept per journal; scan_cap caps works examined per
        journal so one paywalled journal cannot consume the whole crawl.
        """
        stats = {"journals": 0, "unresolved": [], "scanned": 0, "pdfs": 0}
        for name in journal_names:
            j = self.resolve_journal(name)
            if not j or not j.get("ISSN"):
                stats["unresolved"].append(name)
                logger.warning(f"[Library] could not resolve journal: {name}")
                continue
            stats["journals"] += 1
            issn = j["ISSN"][-1]
            resolved = j.get("title", name)
            got, scanned = 0, 0
            logger.info(f"[Library] harvesting '{resolved}' (ISSN {issn}, from {from_year})")
            for work in self.iter_journal_works(issn, from_year, topic=topic):
                scanned += 1
                stats["scanned"] += 1
                entry = self.add_work(work)
                if entry and entry.pdf_path:
                    got += 1
                    stats["pdfs"] += 1
                if got >= per_journal or scanned >= scan_cap:
                    break
            logger.info(f"[Library]   {resolved}: {got} PDFs from {scanned} works scanned")
            self.save()
        self.save()
        logger.info(
            f"[Library] DONE — {len(self.entries)} works indexed, "
            f"{self.n_with_pdf} full texts on disk, "
            f"{len(self.paywalled)} queued for library access"
        )
        return stats

    def close(self):
        self.client.close()
