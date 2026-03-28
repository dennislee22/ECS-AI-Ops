from typing import Optional
from pathlib import Path
import config.config as config
from rag.store import embed_text, _get_lancedb

TOP_K = 10

# ── Sheet aliases ─────────────────────────────────────────────────────────────

_SHEET_ALIASES: dict[str, str] = {
    "dos":            "Dos and Donts",
    "donts":          "Dos and Donts",
    "dos and donts":  "Dos and Donts",
    "dos & donts":    "Dos and Donts",
    "known issues":   "Known Issues",
    "known":          "Known Issues",
    "issues":         "Known Issues",
    "incident":       "Incident",
    "incidents":      "Incident",
    "past learnings": "Incident",   # backward compat alias
    "learnings":      "Incident",
    "past":           "Incident",
}


# ── Result formatters ─────────────────────────────────────────────────────────

def _fmt_known_issue(hit: dict, sim: float) -> str:
    jira  = hit.get("jira", "")
    lines = [
        f"[Known Issues] {hit.get('issue_id', '')} | {hit.get('severity', '')} | relevance:{sim}"
        + (f" | {jira}" if jira else ""),
        f"  Problem          : {hit.get('problem', '')}",
        f"  Root Cause       : {hit.get('root_cause', '')}",
        f"  Resolution       : {hit.get('resolution', '')}",
    ]
    if hit.get("resolution_risk"):
        lines.append(f"  Resolution Risk  : {hit['resolution_risk']}")
    if hit.get("remediation_steps"):
        lines.append(f"  Remediation Steps: {hit['remediation_steps']}")
    return "\n".join(lines)


def _fmt_incident(hit: dict, sim: float) -> str:
    lines = [
        f"[Incident] {hit.get('version', '')} | relevance:{sim}",
        f"  Incident             : {hit.get('incident', '')}",
        f"  Potential Cause      : {hit.get('potential_cause', '')}",
        f"  Potential Resolution : {hit.get('potential_resolution', '')}",
    ]
    if hit.get("risk"):
        lines.append(f"  Risk                 : {hit['risk']}")
    if hit.get("ext_doc"):
        lines.append(f"  Ref                  : {hit['ext_doc']}")
    return "\n".join(lines)


def _fmt_dos_donts(hit: dict, sim: float) -> str:
    return "\n".join([
        f"[Dos and Donts] {hit.get('category', '')} | relevance:{sim}",
        f"  ✅ DO   : {hit.get('do_text', '')}",
        f"  ❌ DON'T: {hit.get('dont_text', '')}",
        f"  Why     : {hit.get('rationale', '')}",
    ])


def _fmt_generic(hit: dict, sim: float) -> str:
    return f"[{hit.get('sheet', 'Unknown')}] relevance:{sim}\n  {hit.get('symptom', '')}"


_FORMATTERS = {
    "Known Issues":  _fmt_known_issue,
    "Incident":      _fmt_incident,
    "Dos and Donts": _fmt_dos_donts,
}


# ── Main retrieval ────────────────────────────────────────────────────────────

def rag_retrieve(query: str, top_k: int = TOP_K,
                 doc_type: Optional[str] = None,
                 sheet: Optional[str] = None) -> str:
    _, docs_tbl, excel_tbl = _get_lancedb()
    sections = []

    if sheet:
        sheet = _SHEET_ALIASES.get(sheet.lower().strip(), sheet)

    # ── Excel KB search ───────────────────────────────────────────────────────
    try:
        excel_count = excel_tbl.count_rows()
    except Exception:
        excel_count = 0

    if excel_count > 0:
        try:
            qvec = embed_text(query)
            aq   = excel_tbl.search(qvec, vector_column_name="vector")
            if sheet:
                aq = aq.where(f"sheet = '{sheet}'")
            all_hits = aq.limit(top_k * 2).to_list()

            seen, merged = set(), []
            for r in all_hits:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    merged.append(r)
            merged = merged[:top_k]

            if merged:
                lines = [f"📋 Knowledge Base ({len(merged)} match(es)):\n"]
                for hit in merged:
                    sim = round(1 - hit.get("_distance", 1.0), 3)
                    fmt = _FORMATTERS.get(hit.get("sheet", ""), _fmt_generic)
                    lines.append(fmt(hit, sim))
                sections.append("\n".join(lines))

        except Exception as e:
            config._log_rag.warning(f"[RAG/Excel] Search failed: {e}")

    # ── Docs search ───────────────────────────────────────────────────────────
    try:
        docs_count = docs_tbl.count_rows()
    except Exception:
        docs_count = 0

    if docs_count > 0:
        try:
            qvec = embed_text(query)
            srch = docs_tbl.search(qvec, vector_column_name="vector")
            if doc_type:
                srch = srch.where(f"doc_type = '{doc_type}'")
            hits = srch.limit(top_k).to_list()
            if hits:
                lines = [f"📄 Documentation ({len(hits)} chunk(s)):\n"]
                for hit in hits:
                    sim = round(1 - hit.get("_distance", 1.0), 3)
                    src = Path(hit.get("source", "?")).name
                    lines.append(f"[{src}] relevance:{sim}\n{hit.get('text', '')}\n")
                sections.append("\n".join(lines))
        except Exception as e:
            config._log_rag.warning(f"[RAG/Docs] Search failed: {e}")

    if sections:
        return "\n\n---\n\n".join(sections)
    if excel_count == 0 and docs_count == 0:
        return "KB_EMPTY: No documents have been ingested into the knowledge base."
    return "No relevant documentation found."


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_doc_stats() -> dict:
    try:
        _, docs_tbl, excel_tbl = _get_lancedb()
        docs_count  = docs_tbl.count_rows()
        excel_count = excel_tbl.count_rows()

        excel_by_sheet: dict = {}
        if excel_count > 0:
            try:
                from collections import Counter
                rows = excel_tbl.search().limit(excel_count + 1).to_list()
                excel_by_sheet = dict(Counter(r.get("sheet", "unknown") for r in rows))
            except Exception:
                pass

        docs_by_type: dict = {}
        if docs_count > 0:
            try:
                from collections import Counter
                rows = docs_tbl.search().limit(docs_count + 1).to_list()
                docs_by_type = dict(Counter(r.get("doc_type", "general") for r in rows))
            except Exception:
                pass

        return {
            "total_chunks":   docs_count + excel_count,
            "docs_chunks":    docs_count,
            "excel_rows":     excel_count,
            "docs_by_type":   docs_by_type,
            "excel_by_sheet": excel_by_sheet,
        }
    except Exception as e:
        return {"total_chunks": 0, "docs_chunks": 0, "excel_rows": 0,
                "docs_by_type": {}, "excel_by_sheet": {}, "error": str(e)}


# ── KB topic detection ────────────────────────────────────────────────────────

_KB_TOPIC_KEYWORDS = [
    "longhorn", "ecs", "cdp", "cloudera", "rancher", "vault", "prometheus",
    "grafana", "cert-manager", "coredns", "ingress", "pvc", "pv", "storageclass",
    "known issue", "known problem", "list issue", "unresolved", "open issue",
    "dos and don", "best practice",
    "incident", "show incident", "list incident",
    "runbook", "playbook", "troubleshoot", "fix", "remediation",
    "crashloop", "oomkill", "imagepull", "pending", "evicted",
    "not running", "not ready", "node pressure",
    "1.5", "1.6", "sp1", "sp2", "upgrade",
]


def _is_kb_topic(question: str) -> bool:
    ql = question.lower()
    return any(k in ql for k in _KB_TOPIC_KEYWORDS)


_MSG_NO_INGEST = (
    "No results found. The knowledge base appears to be empty — "
    "no documents have been ingested yet. "
    "Please go to ⚙ Settings → RAG Documents to upload and ingest your knowledge base files."
)


# ── Tool registration ─────────────────────────────────────────────────────────

RAG_TOOLS = {
    "rag_search": {
        "fn": rag_retrieve,
        "description": (
            "Search the internal knowledge base for known issues, incidents, troubleshooting guides, "
            "dos and don'ts, and operational best practices. "
            "Call this tool in two situations: "
            "(1) AFTER get_unhealthy_pods_detail when a pod shows OOMKilled, CrashLoopBackOff, "
            "ImagePullBackOff, Pending, or any error — use the specific error and component as the query. "
            "(2) DIRECTLY when the user asks about: known issues, incidents, documentation, runbooks, "
            "best practices, dos and don'ts, or WHY something might be happening. "
            "Call rag_search BEFORE live tools when query contains 'issues', 'problems', "
            "'known issues', 'what could cause', 'best practice', 'how to fix'. "
            "Call live tools BEFORE rag_search when query is about current cluster state "
            "('is X healthy', 'how many pods', 'list pvcs'). "
            "Pass sheet= ONLY when the user explicitly asks for a specific category. "
            "Leave sheet= empty (default) to search ALL sheets. "
            "Valid sheet values: 'Known Issues', 'Dos and Donts', 'Incident'. "
            "Examples: "
            "rag_search(query='CrashLoopBackOff cdp-cadence') — no sheet "
            "rag_search(query='longhorn storage issues') — no sheet "
            "rag_search(query='what are the dos and donts', sheet='Dos and Donts') "
            "rag_search(query='vault incident', sheet='Incident') "
            "IMPORTANT: if the result starts with KB_EMPTY: relay the message as-is — "
            "do NOT answer from training data."
        ),
        "parameters": {
            "query":    {"type": "string",
                         "description": "Search query — use specific error names, component names, or symptoms."},
            "top_k":    {"type": "integer", "default": 10},
            "doc_type": {"type": "string",  "default": None},
            "sheet":    {"type": "string",  "default": None,
                         "description": (
                             "Optional sheet filter. Valid values: "
                             "'Known Issues', 'Dos and Donts', 'Incident'. "
                             "Only pass when user explicitly requests a specific category."
                         )},
        },
    },
}
