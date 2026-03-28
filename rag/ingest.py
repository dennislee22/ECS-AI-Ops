import hashlib, re
from pathlib import Path
import config.config as config
from rag.store import embed_text, _get_lancedb

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            pb = text.rfind("\n\n", start, end)
            if pb > start + CHUNK_SIZE // 2:
                end = pb
            else:
                sb = max(text.rfind(". ", start, end), text.rfind(".\n", start, end))
                if sb > start + CHUNK_SIZE // 2:
                    end = sb + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def _doc_type(filename: str) -> str:
    n = filename.lower()
    if any(k in n for k in ["known", "issue", "bug", "error"]):    return "known_issue"
    if any(k in n for k in ["runbook", "playbook", "procedure"]):  return "runbook"
    if any(k in n for k in ["dos", "donts", "guidelines"]):        return "dos_donts"
    return "general"


# ── Plain-text / PDF / Markdown ingest ───────────────────────────────────────

def ingest_file(file_path: str, force: bool = False) -> dict:
    path  = Path(file_path)
    fhash = hashlib.md5(path.read_bytes()).hexdigest()
    _, docs_tbl, _ = _get_lancedb()

    if not force:
        try:
            existing = docs_tbl.search().where(
                f"source = '{str(path)}' AND file_hash = '{fhash}'"
            ).limit(1).to_list()
            if existing:
                config._log_rag.info(f"[RAG] Skip (unchanged): {path.name}")
                return {"file": path.name, "status": "skipped", "chunks": 0}
        except Exception:
            pass

    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        elif suffix == ".md":
            from markdown_it import MarkdownIt
            html = MarkdownIt().render(path.read_text(encoding="utf-8"))
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        else:
            text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"file": path.name, "status": "error", "chunks": 0, "error": str(e)}

    if not text.strip():
        return {"file": path.name, "status": "empty", "chunks": 0}

    chunks   = chunk_text(text)
    doc_type = _doc_type(path.name)
    config._log_rag.info(f"[RAG] {path.name}: {len(chunks)} chunks  type={doc_type}")

    try:
        docs_tbl.delete(f"source = '{str(path)}'")
    except Exception:
        pass

    rows = []
    for i, ch in enumerate(chunks):
        rows.append({
            "id":          f"{fhash}_{i}",
            "vector":      embed_text(ch),
            "text":        ch,
            "source":      str(path),
            "doc_type":    doc_type,
            "chunk_index": i,
            "file_hash":   fhash,
        })
    docs_tbl.add(rows)
    return {"file": path.name, "status": "ingested", "chunks": len(chunks), "doc_type": doc_type}


# ── Excel sheet role mapping ───────────────────────────────────────────────────
# Maps each sheet to column roles via hint substrings matched against headers.
# More specific hints listed first to avoid false matches.
# Prefix hint with "^...$" for exact column name match.

_SHEET_ROLES: dict[str, dict[str, list[str]]] = {

    "Incident": {
        # Columns: #, Version, Incident, Potential Cause,
        #          Potential Resolution, Risk, External Documentation
        "incident":              ["incident"],
        "potential_cause":       ["potential cause", "cause"],
        "potential_resolution":  ["potential resolution", "resolution"],
        "risk":                  ["risk"],
        "ext_doc":               ["external documentation", "documentation", "doc"],
        "version":               ["version"],
    },

    "Dos and Donts": {
        # Columns: #, Category, Version, ✅ DO, ❌ DON'T, Rationale
        "do_text":   ["✅", "do"],
        "dont_text": ["❌", "don"],   # "don" matches "DON'T" before "do"
        "rationale": ["rationale", "reason", "why"],
        "category":  ["category", "type", "area"],
        "version":   ["version"],
    },

    "Known Issues": {
        # Columns: Issue ID, Version, Category, Problem, Root Cause,
        #          Resolution, Resolution (Risk), Remediation Steps, Severity, Jira
        "issue_id":           ["issue id"],
        "problem":            ["problem", "summary", "title"],
        "root_cause":         ["root cause", "cause"],
        "resolution":         ["^resolution$"],          # exact — avoids "resolution (risk)"
        "resolution_risk":    ["resolution (risk)"],
        "remediation_steps":  ["remediation steps", "remediation", "steps"],
        "severity":           ["severity", "priority"],
        "jira":               ["jira", "link", "url"],
        "category":           ["category", "type", "component"],
        "version":            ["version"],
    },
}

_INDEX_HINTS = {"#", "no", "num", "index", "row", "sl no", "s.no"}
_NULL_VALUES = {"", "none", "nan", "n/a", "-", "nat"}


def _clean(v) -> str:
    s = str(v).strip() if v is not None else ""
    return "" if s.lower() in _NULL_VALUES else s


def _best_col(row: dict, hints: list[str], cols: list[str]) -> str:
    """Return first non-empty cell whose column header matches any hint."""
    for hint in hints:
        if hint.startswith("^") and hint.endswith("$"):
            # Exact column name match
            exact = hint[1:-1].lower()
            for col in cols:
                if col.lower().strip() == exact:
                    v = _clean(row.get(col, ""))
                    if v:
                        return v
        else:
            h = hint.lower()
            for col in cols:
                if h in col.lower():
                    v = _clean(row.get(col, ""))
                    if v:
                        return v
    return ""


def _map_row(row: dict, sheet_type: str, cols: list[str]) -> tuple[dict, str]:
    """
    Resolve column roles and build the search anchor text.
    Returns (resolved_fields_dict, search_text).
    """
    roles    = _SHEET_ROLES.get(sheet_type, {})
    resolved = {role: _best_col(row, hints, cols) for role, hints in roles.items()}

    if sheet_type == "Incident":
        search_text = " / ".join(
            t for t in [resolved.get("incident", ""), resolved.get("potential_cause", "")]
            if t
        ).strip(" /")

    elif sheet_type == "Dos and Donts":
        search_text = " / ".join(
            t for t in [resolved.get("do_text", ""), resolved.get("dont_text", "")]
            if t
        ).strip(" /")

    elif sheet_type == "Known Issues":
        search_text = " / ".join(
            t for t in [resolved.get("problem", ""), resolved.get("root_cause", "")]
            if t
        ).strip(" /")

    else:
        parts = [
            _clean(row.get(col, ""))
            for col in cols
            if col.lower().strip() not in _INDEX_HINTS
        ]
        search_text = " / ".join(p for p in parts if p)

    return resolved, search_text


def _build_excel_row(fhash: str, idx: int, path: Path,
                     sheet_type: str, resolved: dict,
                     search_text: str, id_prefix: str) -> dict:
    """Build a fully-typed dict matching the excel_issues PyArrow schema."""
    return {
        "id":                    f"{id_prefix}-{fhash}-{idx}",
        "vector":                embed_text(search_text),
        "source_file":           path.name,
        "file_hash":             fhash,
        "sheet":                 sheet_type,
        "symptom":               search_text,
        # shared
        "version":               resolved.get("version",              ""),
        "category":              resolved.get("category",             ""),
        # Incident
        "incident":              resolved.get("incident",             ""),
        "potential_cause":       resolved.get("potential_cause",      ""),
        "potential_resolution":  resolved.get("potential_resolution", ""),
        "risk":                  resolved.get("risk",                 ""),
        "ext_doc":               resolved.get("ext_doc",              ""),
        # Known Issues
        "issue_id":              resolved.get("issue_id",             ""),
        "problem":               resolved.get("problem",              ""),
        "root_cause":            resolved.get("root_cause",           ""),
        "resolution":            resolved.get("resolution",           ""),
        "resolution_risk":       resolved.get("resolution_risk",      ""),
        "remediation_steps":     resolved.get("remediation_steps",    ""),
        "severity":              resolved.get("severity",             ""),
        "jira":                  resolved.get("jira",                 ""),
        # Dos and Donts
        "do_text":               resolved.get("do_text",              ""),
        "dont_text":             resolved.get("dont_text",            ""),
        "rationale":             resolved.get("rationale",            ""),
    }


# ── Excel ingest ──────────────────────────────────────────────────────────────

def ingest_excel(file_path: str, force: bool = False) -> dict:
    try:
        import pandas as pd
    except ImportError:
        return {"file": Path(file_path).name, "status": "error", "chunks": 0,
                "error": "pandas not installed — pip install pandas openpyxl"}

    path  = Path(file_path)
    fhash = hashlib.md5(path.read_bytes()).hexdigest()
    _, _, excel_tbl = _get_lancedb()

    if not force:
        try:
            existing = excel_tbl.search().where(
                f"source_file = '{path.name}' AND file_hash = '{fhash}'"
            ).limit(1).to_list()
            if existing:
                config._log_rag.info(f"[RAG/Excel] Skip (unchanged): {path.name}")
                return {"file": path.name, "status": "skipped", "chunks": 0}
        except Exception:
            pass

    try:
        xl = pd.read_excel(str(path), sheet_name=None, dtype=str)
    except Exception as e:
        return {"file": path.name, "status": "error", "chunks": 0, "error": str(e)}

    rows  = []
    total = 0

    for sheet_name, df in xl.items():
        sn   = sheet_name.strip()
        sn_l = sn.lower()

        if "known" in sn_l or ("issue" in sn_l and "incident" not in sn_l):
            sheet_type, id_prefix = "Known Issues",  "ki"
        elif "dos" in sn_l or "don" in sn_l or "practice" in sn_l:
            sheet_type, id_prefix = "Dos and Donts", "dd"
        elif "incident" in sn_l or "learn" in sn_l or "past" in sn_l or "postmortem" in sn_l:
            sheet_type, id_prefix = "Incident",      "ic"
        else:
            config._log_rag.info(f"[RAG/Excel] Skipping unrecognised sheet '{sn}'")
            continue

        df.columns = [c.strip() for c in df.columns]
        cols = list(df.columns)
        config._log_rag.info(f"[RAG/Excel] Sheet '{sn}' → {sheet_type} ({len(df)} rows)")

        # Warn about columns not matched by any role hint
        role_hints_flat = [
            h.lstrip("^").rstrip("$")
            for hints in _SHEET_ROLES.get(sheet_type, {}).values()
            for h in hints
        ]
        unmatched = [
            c for c in cols
            if c.lower().strip() not in _INDEX_HINTS
            and not any(h in c.lower() for h in role_hints_flat)
        ]
        if unmatched:
            config._log_rag.warning(
                f"[RAG/Excel] Sheet '{sn}' — unrecognised columns: {unmatched}"
            )

        for _, raw_row in df.iterrows():
            row = {col: _clean(v) for col, v in raw_row.items()}
            resolved, search_text = _map_row(row, sheet_type, cols)
            if not search_text:
                continue

            rows.append(_build_excel_row(fhash, total, path, sheet_type,
                                          resolved, search_text, id_prefix))
            total += 1

    if not rows:
        return {"file": path.name, "status": "empty", "chunks": 0}

    try:
        excel_tbl.delete(
            f"id LIKE 'ki-{fhash}%' OR id LIKE 'dd-{fhash}%' OR id LIKE 'ic-{fhash}%'"
        )
    except Exception:
        pass

    excel_tbl.add(rows)
    config._log_rag.info(f"[RAG/Excel] {path.name}: {total} rows ingested")
    return {"file": path.name, "status": "ingested", "chunks": total, "doc_type": "excel"}


def ingest_directory(docs_dir: str, force: bool = False) -> list:
    p = Path(docs_dir)
    results = []
    for f in (sorted(p.glob("**/*.md")) + sorted(p.glob("**/*.pdf"))
              + sorted(p.glob("**/*.txt"))):
        results.append(ingest_file(str(f), force=force))
    for f in sorted(p.glob("**/*.xlsx")) + sorted(p.glob("**/*.xls")):
        results.append(ingest_excel(str(f), force=force))
    return results
