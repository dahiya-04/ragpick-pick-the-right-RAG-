"""Turn a folder of PDFs / text / HTML into clean documents with metadata.

Every problem the parser notices (repeated headers, Hindi lines, empty pages that are
probably scanned, tables) is recorded per document and written to parse_report.csv.
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from .textutil import split_sentences, stable_hash


@dataclass
class Document:
    doc_id: str
    path: str
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)
    issues: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- file readers

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LETTER = re.compile(r"[A-Za-zऀ-ॿ]")


def _read_pdf(path: Path, opts: dict) -> tuple[list[str], dict, str]:
    import pymupdf

    issues = {"tables": 0, "empty_pages": 0}
    pages: list[str] = []
    with pymupdf.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or ""
        for page in pdf:
            tables = []
            if opts.get("tables", True):
                try:
                    tables = page.find_tables().tables
                except Exception:  # table detection is best effort
                    tables = []
            if tables:
                rects = [pymupdf.Rect(t.bbox) for t in tables]
                items = []
                for b in page.get_text("blocks", sort=True):
                    if b[6] != 0:  # image block
                        continue
                    r = pymupdf.Rect(b[:4])
                    centre = pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
                    if any(tr.contains(centre) for tr in rects):
                        continue
                    items.append((r.y0, b[4]))
                for t, tr in zip(tables, rects):
                    try:
                        md = t.to_markdown()
                    except Exception:
                        md = ""
                    if md.strip():
                        items.append((tr.y0, "\n" + md.strip() + "\n"))
                        issues["tables"] += 1
                items.sort(key=lambda x: x[0])
                text = "\n".join(s for _, s in items)
            else:
                text = page.get_text("text", sort=True)
            if len(text.strip()) < 20:
                issues["empty_pages"] += 1
            pages.append(text)
    issues["pages"] = len(pages)
    return pages, issues, title


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag in ("p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"):
            self.parts.append("\n")
        if tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _read_text(path: Path) -> tuple[list[str], dict, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        ex = _TextExtractor()
        ex.feed(raw)
        raw = html.unescape("".join(ex.parts))
    pages = raw.split("\f")  # form feed = page break in exported text
    return pages, {"pages": len(pages), "tables": 0, "empty_pages": 0}, ""


# --------------------------------------------------------------------------- cleaning


def _edge_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_repeating_lines(pages: list[str], min_frac: float = 0.5, edge: int = 3) -> tuple[list[str], int]:
    """Remove header/footer lines that repeat at the top or bottom of most pages."""
    if len(pages) < 3:
        return pages, 0
    counts: Counter = Counter()
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip()]
        counts.update({_edge_key(l) for l in lines[:edge] + lines[-edge:]})
    threshold = max(2, math.ceil(min_frac * len(pages)))
    repeating = {k for k, c in counts.items() if c >= threshold and k}
    removed, out = 0, []
    for p in pages:
        lines = p.splitlines()
        idx = [i for i, l in enumerate(lines) if l.strip()]
        edge_idx = set(idx[:edge] + idx[-edge:])
        kept = []
        for i, line in enumerate(lines):
            if i in edge_idx and _edge_key(line) in repeating:
                removed += 1
                continue
            kept.append(line)
        out.append("\n".join(kept))
    return out, removed


def _devanagari_fraction(line: str) -> float:
    letters = _LETTER.findall(line)
    if not letters:
        return 0.0
    return len(_DEVANAGARI.findall(line)) / len(letters)


def clean_text(text: str, drop_hindi: bool) -> tuple[str, dict]:
    issues = {"hindi_lines": 0}
    lines = []
    for line in text.splitlines():
        if _devanagari_fraction(line) > 0.3:
            issues["hindi_lines"] += 1
            if drop_hindi:
                continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), issues


# --------------------------------------------------------------------------- metadata

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

PRESETS: dict[str, dict[str, str]] = {
    "rbi": {
        "circular_no": r"RBI/(?:[A-Z]+/)?\d{4}-\d{2,4}/\d+",
        "ref_no": r"\b[A-Z]{2,}(?:\.[A-Z]{2,})*\.(?:[A-Za-z]+\.)*No\.\s?[A-Za-z0-9./()-]*\d[A-Za-z0-9./()-]*",
        "date": rf"(?:{_MONTHS})\s+\d{{1,2}},\s*\d{{4}}|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS}),?\s+\d{{4}}|\b\d{{2}}[./-]\d{{2}}[./-]\d{{4}}\b",
    },
    "none": {},
}

_SUPERSEDE_WORDS = re.compile(
    r"supersed|in supersession|repeal|stands? withdrawn|hereby withdrawn|rescind|replaces? the (?:earlier|previous)",
    re.I,
)
_REPEAL_SECTION = re.compile(
    r"^[^\n]*(?:circulars?|directions?|instructions?)[^\n]{0,40}repealed[^\n]*$|^[^\n]*repealed (?:circulars?|instructions?)[^\n]*$",
    re.I | re.M,
)
_TITLE_SKIP = re.compile(
    r"reserve bank|^dear|^madam|^sir|^all |department|mumbai|www\.|e-?mail|tel|fax|notification|^to,?$|"
    r"RBI/|No\.|^\d|^page",
    re.I,
)


def _parse_date(s: str) -> str:
    s = re.sub(r"(\d)(st|nd|rd|th)", r"\1", s.strip()).replace(",", ", ")
    s = re.sub(r"\s+", " ", s)
    for fmt in ("%B %d, %Y", "%d %B %Y", "%d %B, %Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


def _guess_title(text: str, pdf_title: str, fallback: str) -> str:
    if pdf_title and len(pdf_title) > 10 and not re.search(r"microsoft|untitled|\.docx?$", pdf_title, re.I):
        return pdf_title.strip()
    for line in text.splitlines()[:60]:
        line = line.strip()
        if 15 <= len(line) <= 200 and line[0].isupper() and not _TITLE_SKIP.search(line):
            return line
    return fallback


def extract_metadata(doc: Document, patterns: dict[str, str]) -> None:
    head = doc.text[:4000]
    for fld, pat in patterns.items():
        if fld == "date":
            # take the first match that is a real date (reference numbers like 13.03.00/2019 look date-like)
            for m in re.finditer(pat, head):
                iso = _parse_date(m.group(0).rstrip(".,;"))
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso):
                    doc.meta[fld] = iso
                    break
            continue
        m = re.search(pat, head)
        if m:
            doc.meta[fld] = m.group(0).rstrip(".,;")
    if "ref_no" in doc.meta:
        doc.meta.setdefault("department", doc.meta["ref_no"].split(".")[0])

    # References to older circulars that this document supersedes.
    ref_pats = [re.compile(patterns[k]) for k in ("circular_no", "ref_no") if k in patterns]
    if not ref_pats:
        return
    refs: set[str] = set()
    for sent in split_sentences(doc.text):
        if _SUPERSEDE_WORDS.search(sent):
            for rp in ref_pats:
                refs.update(m.group(0).rstrip(".,;") for m in rp.finditer(sent))
    sec = _REPEAL_SECTION.search(doc.text)
    if sec:
        tail = doc.text[sec.start():]
        for rp in ref_pats:
            refs.update(m.group(0).rstrip(".,;") for m in rp.finditer(tail))
    own = {doc.meta.get("circular_no"), doc.meta.get("ref_no")}
    doc.meta["supersedes"] = sorted(r for r in refs if r not in own)


def resolve_supersession(docs: list[Document]) -> None:
    by_ref: dict[str, Document] = {}
    for d in docs:
        for k in ("circular_no", "ref_no"):
            if d.meta.get(k):
                by_ref[d.meta[k]] = d
    for d in docs:
        for ref in d.meta.get("supersedes", []):
            old = by_ref.get(ref)
            if old is not None and old is not d and not old.meta.get("superseded_by"):
                old.meta["superseded"] = True
                old.meta["superseded_by"] = d.doc_id
    for d in docs:
        d.meta.setdefault("superseded", False)


def _to_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def apply_metadata_csv(docs: list[Document], csv_path: Path) -> None:
    """Manual overrides: one row per file. Column `file` matches a doc_id or file name."""
    index = {}
    for d in docs:
        index[d.doc_id] = d
        index[Path(d.path).name] = d
        index[Path(d.path).stem] = d
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = index.get((row.pop("file", "") or "").strip())
            if d is None:
                continue
            for k, v in row.items():
                if v is None or v == "":
                    continue
                if k == "superseded":
                    d.meta[k] = _to_bool(v)
                elif k == "title":
                    d.title = v
                else:
                    d.meta[k] = v
            if d.meta.get("superseded_by"):
                d.meta["superseded"] = True


# --------------------------------------------------------------------------- corpus

SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm"}


def parse_file(path: Path, root: Path, opts: dict) -> Document:
    if path.suffix.lower() == ".pdf":
        pages, issues, pdf_title = _read_pdf(path, opts)
    else:
        pages, issues, pdf_title = _read_text(path)
    removed = 0
    if opts.get("remove_headers", True):
        pages, removed = strip_repeating_lines(pages)
    text, clean_issues = clean_text("\n\n".join(pages), opts.get("drop_hindi", True))
    issues.update(clean_issues)
    issues["header_lines_removed"] = removed
    issues["chars"] = len(text)
    flags = []
    if issues["pages"] and issues["empty_pages"] / issues["pages"] > 0.5:
        flags.append("likely_scanned_needs_ocr")
    if issues["hindi_lines"]:
        flags.append("bilingual")
    if issues["tables"]:
        flags.append("has_tables")
    if issues["chars"] < 200:
        flags.append("very_short")
    issues["flags"] = ";".join(flags)
    rel = path.relative_to(root).with_suffix("")
    doc_id = rel.as_posix()
    return Document(doc_id=doc_id, path=str(path), text=text, title=pdf_title, issues=issues)


def parse_corpus(corpus_cfg: dict, base_dir: Path, cache_dir: Path, log=print) -> list[Document]:
    root = (base_dir / corpus_cfg["path"]).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Corpus folder not found: {root}")
    pattern = corpus_cfg.get("glob", "**/*")
    files = sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No supported files ({', '.join(sorted(SUPPORTED))}) in {root} matching {pattern}")

    opts = corpus_cfg.get("parse", {})
    meta_cfg = corpus_cfg.get("metadata", {})
    csv_path = (base_dir / meta_cfg["csv"]).resolve() if meta_cfg.get("csv") else None
    fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime) for p in files]
    if csv_path and csv_path.exists():
        fingerprint.append((str(csv_path), csv_path.stat().st_mtime))
    key = stable_hash({"files": fingerprint, "opts": opts, "meta": meta_cfg, "v": 2})
    cache_file = cache_dir / "parsed" / f"{key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        log(f"[parse] {len(data)} documents loaded from cache")
        return [Document(**d) for d in data]

    log(f"[parse] parsing {len(files)} files from {root}")
    docs = []
    for i, p in enumerate(files, 1):
        try:
            docs.append(parse_file(p, root, opts))
        except Exception as e:  # keep going; report the failure
            log(f"  ! failed to parse {p.name}: {e}")
        if i % 25 == 0:
            log(f"  parsed {i}/{len(files)}")

    preset = meta_cfg.get("extractor", "rbi")
    patterns = dict(PRESETS.get(preset, {}))
    patterns.update(meta_cfg.get("patterns", {}) or {})
    for d in docs:
        extract_metadata(d, patterns)
        d.title = _guess_title(d.text, d.title, Path(d.path).stem)
    resolve_supersession(docs)
    if csv_path:
        apply_metadata_csv(docs, csv_path)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps([asdict(d) for d in docs], ensure_ascii=False), encoding="utf-8")
    return docs


def write_parse_report(docs: list[Document], path: Path) -> None:
    cols = ["doc_id", "title", "circular_no", "ref_no", "date", "superseded", "superseded_by",
            "pages", "empty_pages", "tables", "header_lines_removed", "hindi_lines", "chars", "flags"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in docs:
            row = {"doc_id": d.doc_id, "title": d.title}
            row.update({k: d.meta.get(k, "") for k in ("circular_no", "ref_no", "date", "superseded", "superseded_by")})
            row.update({k: d.issues.get(k, "") for k in cols if k in d.issues})
            w.writerow(row)
