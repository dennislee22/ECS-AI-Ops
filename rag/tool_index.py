from __future__ import annotations

import json
import logging
from typing import Any
import numpy as np
import os

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_ALWAYS_INCLUDE = {
    "kubectl_exec",    # catch-all kubectl fallback
    "find_resource",   # primary named-resource dispatcher
    "rag_search",      # knowledge base
}

_TABLE_NAME = "tool_index"
_CONFIDENCE_THRESHOLD = 0.65
_MIN_CONFIDENT = 1  # <-- FIXED: Allow even a single highly confident match to succeed

# ── Embedding helper ─────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer
import config.config as _cfg

_embedder = None  # module-level singleton — loaded once on first use

def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    model_name = getattr(_cfg, "EMBED_MODEL", None)
    if not model_name:
        raise ValueError("EMBED_MODEL must be set in config.config")

    model_name = os.path.expanduser(model_name)
    logger.info(f"[tool_index] Loading embedder from: {model_name}")

    try:
        _embedder = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"[tool_index] Failed to load embedder: {e}", exc_info=True)
        raise

    return _embedder

def _embed(text: str, is_query: bool = False) -> list[float]:
    embedder = _get_embedder()
    # <-- FIXED: Nomic-embed-text strictly expects a task prefix for user questions
    final_text = f"search_query: {text}" if is_query else text
    vec = embedder.encode(final_text, normalize_embeddings=True)
    return vec.tolist()


# ── Text to embed for each tool ───────────────────────────────────────────────

def _tool_text(name: str, cfg: dict) -> str:
    """
    Build the text that gets embedded for a tool.
    Prioritizes 'embed_keywords' if present for pure semantic signal.
    """
    # <-- FIXED: Decouple retrieval (keywords) from execution (instructions)
    if "embed_keywords" in cfg:
        return f"{name}: {cfg['embed_keywords']}"
        
    desc = cfg.get("description", "")
    # First 300 chars captures the intent; verbose guardrails are not needed
    # for retrieval — they're needed at execution time (full schema).
    short_desc = desc[:300].rsplit(" ", 1)[0]  # trim at word boundary
    return f"{name}: {short_desc}"


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_tools(all_tools: dict[str, dict]) -> None:
    """
    Drop and recreate the tool_index table in LanceDB with fresh embeddings.
    Called once at startup (always-fresh strategy).

    Parameters
    ----------
    all_tools : dict
        Combined tool registry — {**K8S_TOOL_METADATA, **RAG_TOOLS}
        Same dict you pass to build_agent().
    """
    try:
        import lancedb
        import config.config as _cfg

        db = lancedb.connect(_cfg.LANCEDB_DIR)

        # Always-fresh: drop existing table if present
        existing = db.table_names()
        if _TABLE_NAME in existing:
            db.drop_table(_TABLE_NAME)
            logger.info(f"[tool_index] Dropped existing '{_TABLE_NAME}' table")

        rows = []
        for name, cfg in all_tools.items():
            text = _tool_text(name, cfg)
            vector = _embed(text, is_query=False)
            rows.append({
                "tool_name": name,
                "text":      text,
                "vector":    vector,
            })

        db.create_table(_TABLE_NAME, data=rows)
        logger.info(f"[tool_index] Ingested {len(rows)} tools into '{_TABLE_NAME}'")

    except Exception as exc:
        logger.error(f"[tool_index] ingest_tools failed: {exc}", exc_info=True)
        # Non-fatal — app will fall back to all tools if table is missing


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve_tools(
    user_query: str,
    all_schemas: list[dict],
    top_k: int = 8,  # You can safely bump this up to 12 in app.py if needed!
) -> list[dict]:
    # Build a quick lookup: tool_name → full schema
    schema_map = {s["function"]["name"]: s for s in all_schemas}

    # ── Always-include tools (always present regardless of query) ─────────────
    always_schemas = [schema_map[n] for n in _ALWAYS_INCLUDE if n in schema_map]
    always_names   = {s["function"]["name"] for s in always_schemas}

    try:
        import lancedb
        import config.config as _cfg

        db    = lancedb.connect(_cfg.LANCEDB_DIR)

        # If the table wasn't created (ingest failed at startup), fall back
        if _TABLE_NAME not in db.table_names():
            logger.warning("[tool_index] tool_index table not found — falling back to all tools")
            return all_schemas

        table = db.open_table(_TABLE_NAME)

        query_vec = _embed(user_query, is_query=True)

        # Retrieve top_k + buffer so we have room after filtering
        results = (
            table.search(query_vec)
                 .metric("cosine")  # <-- FIXED: Tell LanceDB to use Cosine distance
                 .limit(top_k + len(_ALWAYS_INCLUDE))
                 .to_pandas()
        )

        # ── Confidence check ──────────────────────────────────────────────────
        # _distance is cosine distance (0 = identical, 2 = opposite)
        confident = results[results["_distance"] < _CONFIDENCE_THRESHOLD]

        if len(confident) < _MIN_CONFIDENT:
            logger.info(
                f"[tool_index] Low confidence ({len(confident)} results under threshold "
                f"{_CONFIDENCE_THRESHOLD}) for query={user_query[:60]!r} — "
                f"falling back to all {len(all_schemas)} tools"
            )
            return all_schemas

        # ── Build final schema list ───────────────────────────────────────────
        selected_names  = set(confident["tool_name"].tolist())
        selected_names |= always_names   # union with always-include set

        selected_schemas = [schema_map[n] for n in selected_names if n in schema_map]

        logger.info(
            f"[tool_index] query={user_query[:60]!r} → "
            f"{len(selected_schemas)} tools selected "
            f"(semantic={len(confident)}, always={len(always_names)}): "
            f"{sorted(selected_names)}"
        )

        return selected_schemas

    except Exception as exc:
        logger.error(f"[tool_index] retrieve_tools failed: {exc} — falling back to all tools", exc_info=True)
        return all_schemas