from pathlib import Path
import config.config as config

_embedder_fn  = None
_lancedb_conn = None
_docs_table   = None
_excel_table  = None
_EMBED_DIM    = 768

# ── Embedder ──────────────────────────────────────────────────────────────────

def _get_embedder():
    global _embedder_fn
    if _embedder_fn is not None:
        return _embedder_fn

    config._log_rag.info(f"[Embed] Loading SentenceTransformer: {config.EMBED_MODEL}")
    from sentence_transformers import SentenceTransformer
    import transformers as _tf
    _tf.logging.set_verbosity_error()

    if config.NUM_GPU > 0:
        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                config._log_rag.warning(
                    "[Embed] NUM_GPU=%d but torch.cuda.is_available()=False "
                    "(CUDA runtime issue?) — falling back to CPU", config.NUM_GPU
                )
                device = "cpu"
        except ImportError:
            pass
    else:
        device = "cpu"

    config._log_rag.info(f"[Embed] device={device} (NUM_GPU={config.NUM_GPU})")
    _st = SentenceTransformer(config.EMBED_MODEL, device=device, trust_remote_code=True)

    def _local(text: str) -> list:
        return _st.encode(text, normalize_embeddings=True).tolist()

    _embedder_fn = _local
    return _embedder_fn


def embed_text(text: str) -> list:
    return _get_embedder()(text)


# ── LanceDB ───────────────────────────────────────────────────────────────────

def _get_lancedb():
    global _lancedb_conn, _docs_table, _excel_table
    if _lancedb_conn is not None:
        return _lancedb_conn, _docs_table, _excel_table

    import lancedb
    import pyarrow as pa

    Path(config.LANCEDB_DIR).mkdir(parents=True, exist_ok=True)
    config._log_rag.info(f"[LanceDB] Opening store: {config.LANCEDB_DIR}")
    _lancedb_conn = lancedb.connect(config.LANCEDB_DIR)

    # ── Docs table (md / pdf / txt chunks) ───────────────────────────────────
    docs_schema = pa.schema([
        pa.field("id",          pa.utf8()),
        pa.field("vector",      pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("text",        pa.utf8()),
        pa.field("source",      pa.utf8()),
        pa.field("doc_type",    pa.utf8()),
        pa.field("chunk_index", pa.int32()),
        pa.field("file_hash",   pa.utf8()),
    ])
    if "docs" in _lancedb_conn.table_names():
        _docs_table = _lancedb_conn.open_table("docs")
    else:
        _docs_table = _lancedb_conn.create_table("docs", schema=docs_schema)
        config._log_rag.info("[LanceDB] Created table: docs")

    # ── Excel table — aligned to Incident / Dos and Donts / Known Issues ─────
    # Sheet: Incident
    #   columns: #, Version, Incident, Potential Cause, Potential Resolution,
    #            Risk, External Documentation
    # Sheet: Dos and Donts
    #   columns: #, Category, Version, ✅ DO, ❌ DON'T, Rationale
    # Sheet: Known Issues
    #   columns: Issue ID, Version, Category, Problem, Root Cause,
    #            Resolution, Resolution (Risk), Remediation Steps, Severity, Jira
    excel_schema = pa.schema([
        pa.field("id",               pa.utf8()),
        pa.field("vector",           pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("source_file",      pa.utf8()),
        pa.field("file_hash",        pa.utf8()),
        pa.field("sheet",            pa.utf8()),  # Incident | Dos and Donts | Known Issues
        pa.field("symptom",          pa.utf8()),  # search anchor text (embedded)
        # ── Shared ───────────────────────────────────────────────────────────
        pa.field("version",          pa.utf8()),
        pa.field("category",         pa.utf8()),
        # ── Incident sheet ───────────────────────────────────────────────────
        pa.field("incident",         pa.utf8()),  # Incident column
        pa.field("potential_cause",  pa.utf8()),  # Potential Cause
        pa.field("potential_resolution", pa.utf8()),  # Potential Resolution
        pa.field("risk",             pa.utf8()),  # Risk
        pa.field("ext_doc",          pa.utf8()),  # External Documentation
        # ── Known Issues sheet ───────────────────────────────────────────────
        pa.field("issue_id",         pa.utf8()),  # Issue ID
        pa.field("problem",          pa.utf8()),  # Problem
        pa.field("root_cause",       pa.utf8()),  # Root Cause
        pa.field("resolution",       pa.utf8()),  # Resolution
        pa.field("resolution_risk",  pa.utf8()),  # Resolution (Risk)
        pa.field("remediation_steps",pa.utf8()),  # Remediation Steps
        pa.field("severity",         pa.utf8()),  # Severity
        pa.field("jira",             pa.utf8()),  # Jira
        # ── Dos and Donts sheet ──────────────────────────────────────────────
        pa.field("do_text",          pa.utf8()),  # ✅ DO
        pa.field("dont_text",        pa.utf8()),  # ❌ DON'T
        pa.field("rationale",        pa.utf8()),  # Rationale
    ])
    # Expected field names in the current excel schema
    _EXCEL_REQUIRED_FIELDS = {"version", "incident", "potential_cause",
                               "resolution_risk", "remediation_steps", "ext_doc"}

    if "excel_issues" in _lancedb_conn.table_names():
        _excel_table = _lancedb_conn.open_table("excel_issues")
        # Check if schema is stale (missing fields from current version)
        existing_fields = {f.name for f in _excel_table.schema}
        if not _EXCEL_REQUIRED_FIELDS.issubset(existing_fields):
            config._log_rag.warning(
                "[LanceDB] excel_issues schema is stale — dropping and recreating. "
                "Re-ingest your Excel files after restart."
            )
            _lancedb_conn.drop_table("excel_issues")
            _excel_table = _lancedb_conn.create_table("excel_issues", schema=excel_schema)
            config._log_rag.info("[LanceDB] Recreated table: excel_issues (schema updated)")
    else:
        _excel_table = _lancedb_conn.create_table("excel_issues", schema=excel_schema)
        config._log_rag.info("[LanceDB] Created table: excel_issues")

    return _lancedb_conn, _docs_table, _excel_table


def init_db():
    _get_lancedb()
    _get_embedder()
