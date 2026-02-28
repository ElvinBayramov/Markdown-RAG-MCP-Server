"""
Markdown RAG MCP Server v3.0 (Windows-compatible)

Features:
- Heading-aware chunking with sentence overlap
- Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- Hybrid Search: ChromaDB vector + BM25 keyword scoring
- Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
- Metadata filters (category, filename)
- Clean breadcrumb output
"""

import os
import re
import glob
import warnings
import logging
import hashlib
from pathlib import Path
from typing import Optional

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
_SERVER_DIR = Path(__file__).parent
DOCS_PATH = os.getenv(
    "RAG_DOCS_PATH",
    str(_SERVER_DIR.parent)
)
DB_PATH = os.getenv(
    "RAG_DB_PATH",
    str(_SERVER_DIR / "chroma_db")
)
COLLECTION_NAME = "docs_v3"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ──────────────────────────────────────────────
# ChromaDB + Models setup
# ──────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

# Lazy load cross-encoder (downloads ~80MB on first use)
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder

# BM25 index (built after indexing)
_bm25_index = None
_bm25_corpus = None   # list of (chunk_id, text)

def _build_bm25(ids: list[str], texts: list[str]):
    """Build BM25 index from indexed documents."""
    global _bm25_index, _bm25_corpus
    if not ids:
        _bm25_index = None
        _bm25_corpus = None
        return
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(t) for t in texts]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = list(zip(ids, texts))

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 2]

mcp = FastMCP("Markdown RAG")


# ──────────────────────────────────────────────
# Heading-Aware Chunking with Overlap
# ──────────────────────────────────────────────
def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for overlap."""
    pieces = re.split(r"(?<=[.!?。])\s+|\n{2,}", text)
    return [s.strip() for s in pieces if s.strip()]

def _extract_sections(text: str, filepath: str) -> list[dict]:
    """Split markdown by headings (## and ###) into semantic chunks
    with 2-sentence overlap between consecutive chunks."""

    text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)

    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    raw_sections = []
    current_heading = "Introduction"
    current_h1 = Path(filepath).stem

    for part in parts:
        part = part.strip()
        if not part:
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)$", part)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            if level == 1:
                current_h1 = title
            current_heading = title
            continue

        clean = re.sub(r"[|\-\s#*`>]", "", part)
        if len(clean) < 30:
            continue

        # Split long sections into sub-chunks (~250 words)
        words = part.split()
        if len(words) > 300:
            sub_chunks = []
            for i in range(0, len(words), 250):
                sub_chunk = " ".join(words[i:i + 250])
                sub_chunks.append(sub_chunk)
        else:
            sub_chunks = [part]

        for idx, chunk_text in enumerate(sub_chunks):
            raw_sections.append({
                "text": chunk_text,
                "heading": current_heading,
                "parent_heading": current_h1,
                "source": filepath,
                "filename": Path(filepath).name,
                "sub_index": idx,
                "word_count": len(chunk_text.split())
            })

    # Add 2-sentence overlap between consecutive chunks from same file
    for i in range(1, len(raw_sections)):
        if raw_sections[i]["source"] == raw_sections[i-1]["source"]:
            prev_sentences = _split_into_sentences(raw_sections[i-1]["text"])
            overlap = " ".join(prev_sentences[-2:]) if len(prev_sentences) >= 2 else ""
            if overlap:
                raw_sections[i]["text"] = f"[...] {overlap}\n\n{raw_sections[i]['text']}"

    return raw_sections


def _categorize_file(filepath: str, content: str) -> str:
    """
    Auto-categorize file — three-level priority:
    1. YAML Frontmatter `category:` key (explicit user override)
    2. First H1 heading `# Title` in the document (dynamic, zero-effort)
    3. Filename stem as last resort
    """
    # Priority 1: YAML Frontmatter
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, flags=re.DOTALL)
    if frontmatter_match:
        for line in frontmatter_match.group(1).split('\n'):
            if line.strip().lower().startswith('category:'):
                return line.split(':', 1)[1].strip().lower()

    # Priority 2: First H1 heading — "# My Document Title" → "my document title"
    h1_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if h1_match:
        raw_title = h1_match.group(1).strip()
        # Normalize: lowercase, strip markdown emphasis, keep letters/numbers/spaces
        category = re.sub(r"[*_`]", "", raw_title).lower()
        category = re.sub(r"[^\w\s-]", "", category).strip()
        if category:
            return category

    # Priority 3: Filename stem
    return Path(filepath).stem.lower().replace("_", " ").replace("-", " ")


def _format_result(doc: str, meta: dict, score: float, rank: int) -> str:
    """Format a single search result cleanly."""
    heading = meta.get("heading", "")
    parent = meta.get("parent_heading", "")
    filename = meta.get("filename", "unknown")
    category = meta.get("category", "")

    breadcrumb = filename
    if parent and parent != Path(filename).stem:
        breadcrumb += f" > {parent}"
    if heading and heading != parent:
        breadcrumb += f" > {heading}"

    content = doc.strip()
    # Remove overlap prefix markers
    content = re.sub(r"^\[\.\.\.\]\s*", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    if len(content) > 600:
        cut = content[:600].rfind(". ")
        if cut > 300:
            content = content[:cut + 1]
        else:
            content = content[:600] + "..."

    return (
        f"### [{rank}] {breadcrumb}\n"
        f"**Relevance:** {score:.0%} | **Category:** {category}\n\n"
        f"{content}\n"
    )


# ──────────────────────────────────────────────
# MCP Tools
# ──────────────────────────────────────────────
@mcp.tool()
def index_documents(docs_path: str = DOCS_PATH) -> str:
    """
    Index all Markdown files in the given directory into ChromaDB.
    Run once (or when docs change). Downloads embedding model on first use (~50MB).
    """
    md_files = glob.glob(os.path.join(docs_path, "**", "*.md"), recursive=True)
    if not md_files:
        return f"No .md files found in {docs_path}"

    ids, texts, metas = [], [], []

    for filepath in md_files:
        try:
            raw = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        filename = Path(filepath).name
        category = _categorize_file(filepath, raw)
        sections = _extract_sections(raw, filepath)

        for sec in sections:
            chunk_hash = hashlib.md5(sec["text"].encode("utf-8")).hexdigest()[:10]
            chunk_id = f"{Path(filepath).stem}__{chunk_hash}"
            chunk_id = re.sub(r"[^a-zA-Z0-9_]", "_", chunk_id)

            ids.append(chunk_id)
            texts.append(sec["text"])
            metas.append({
                "source": sec["source"],
                "filename": sec["filename"],
                "heading": sec["heading"],
                "parent_heading": sec["parent_heading"],
                "category": category,
                "word_count": sec["word_count"]
            })

    # Clear old collection and re-index
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    global collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 50
    added = 0
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metas[i:batch_end],
        )
        added += batch_end - i

    # Build BM25 index
    _build_bm25(ids, texts)

    categories = {}
    for m in metas:
        cat = m["category"]
        categories[cat] = categories.get(cat, 0) + 1

    cat_summary = " | ".join(f"{k}: {v}" for k, v in sorted(categories.items()))

    return (
        f"Indexed {added} chunks from {len(md_files)} files\n"
        f"Path: {docs_path}\n"
        f"Categories: {cat_summary}\n"
        f"Model: {EMBED_MODEL} (multilingual)\n"
        f"Search: Hybrid (Vector + BM25) + Cross-Encoder Reranking\n"
        f"DB: {DB_PATH}"
    )


@mcp.tool()
def search_docs(
    query: str,
    n_results: int = 5,
    category: Optional[str] = None,
    filename: Optional[str] = None
) -> str:
    """
    Semantic search across indexed Markdown documentation.
    Returns the most relevant chunks with source file references.

    Args:
        query: Natural language query, e.g. "how does the authentication logic work?"
        n_results: Number of results to return (default 5)
        category: Optional filter by category (dynamically based on your root folder names)
        filename: Optional filter by specific file, e.g. "architecture.md"
    """
    count = collection.count()
    if count == 0:
        return "No documents indexed yet. Call index_documents() first."

    # ── Step 1: Vector search (ChromaDB) ──
    where_filter = None
    conditions = []
    if category:
        conditions.append({"category": category})
    if filename:
        conditions.append({"filename": filename})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    fetch_count = min(n_results * 4, count)  # fetch more for reranking

    try:
        vector_results = collection.query(
            query_texts=[query],
            n_results=fetch_count,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        return f"Search error: {str(e)}"

    if not vector_results["documents"][0]:
        return f"No results found for: '{query}'"

    # Build candidate pool from vector results
    candidates = {}
    for doc, meta, dist in zip(
        vector_results["documents"][0],
        vector_results["metadatas"][0],
        vector_results["distances"][0]
    ):
        vec_score = 1 - dist
        key = hashlib.md5(doc[:100].encode()).hexdigest()
        candidates[key] = {
            "doc": doc, "meta": meta,
            "vec_score": vec_score, "bm25_score": 0.0
        }

    # ── Step 2: BM25 keyword search ──
    if _bm25_index is not None and _bm25_corpus is not None:
        query_tokens = _tokenize(query)
        if query_tokens:
            bm25_scores = _bm25_index.get_scores(query_tokens)
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            # Match BM25 results to vector candidates
            for idx, (chunk_id, text) in enumerate(_bm25_corpus):
                key = hashlib.md5(text[:100].encode()).hexdigest()
                if key in candidates:
                    candidates[key]["bm25_score"] = bm25_scores[idx] / max_bm25

    # ── Step 3: Combine scores (RRF - Reciprocal Rank Fusion) ──
    # Sort by vector score to get vector rank
    vec_sorted = sorted(candidates.values(), key=lambda x: x["vec_score"], reverse=True)
    for rank, c in enumerate(vec_sorted):
        c["vec_rank"] = rank + 1

    # Sort by BM25 score to get BM25 rank
    bm25_sorted = sorted(candidates.values(), key=lambda x: x["bm25_score"], reverse=True)
    for rank, c in enumerate(bm25_sorted):
        c["bm25_rank"] = rank + 1

    # RRF: combined_score = 1/(k+vec_rank) + 1/(k+bm25_rank)
    k = 60  # standard RRF constant
    for c in candidates.values():
        c["hybrid_score"] = (1.0 / (k + c["vec_rank"])) + (1.0 / (k + c["bm25_rank"]))

    # Sort by hybrid score
    hybrid_sorted = sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)

    # Take top candidates for reranking
    top_candidates = hybrid_sorted[:min(n_results * 2, len(hybrid_sorted))]

    # ── Step 4: Cross-Encoder Reranking ──
    try:
        cross_encoder = _get_cross_encoder()
        pairs = [(query, c["doc"][:512]) for c in top_candidates]
        ce_scores = cross_encoder.predict(pairs)

        # Normalize CE scores to 0-1
        min_ce = min(ce_scores)
        max_ce = max(ce_scores) if max(ce_scores) != min(ce_scores) else min(ce_scores) + 1
        for i, c in enumerate(top_candidates):
            c["ce_score"] = (ce_scores[i] - min_ce) / (max_ce - min_ce)

        # Final score: 40% hybrid + 60% cross-encoder
        for c in top_candidates:
            # Normalize hybrid score to 0-1 range
            max_hybrid = max(x["hybrid_score"] for x in top_candidates)
            min_hybrid = min(x["hybrid_score"] for x in top_candidates)
            hybrid_range = max_hybrid - min_hybrid if max_hybrid != min_hybrid else 1.0
            norm_hybrid = (c["hybrid_score"] - min_hybrid) / hybrid_range
            c["final_score"] = 0.4 * norm_hybrid + 0.6 * c["ce_score"]

    except Exception:
        # Fallback: use hybrid score directly
        max_h = max(c["hybrid_score"] for c in top_candidates)
        for c in top_candidates:
            c["final_score"] = c["hybrid_score"] / max_h if max_h > 0 else 0

    # Sort by final score
    final_sorted = sorted(top_candidates, key=lambda x: x["final_score"], reverse=True)
    final_results = final_sorted[:n_results]

    # ── Step 5: Format output ──
    output_parts = [f"## Results for: \"{query}\"\n"]

    if category or filename:
        filters = []
        if category:
            filters.append(f"category={category}")
        if filename:
            filters.append(f"file={filename}")
        output_parts.append(f"**Filters:** {', '.join(filters)}\n")

    output_parts.append(f"**Method:** Hybrid (Vector + BM25) + Cross-Encoder Reranking\n")

    for rank, c in enumerate(final_results, 1):
        output_parts.append(_format_result(c["doc"], c["meta"], c["final_score"], rank))

    return "\n".join(output_parts)


@mcp.tool()
def rag_status() -> str:
    """Show how many chunks are indexed and from which files."""
    count = collection.count()
    if count == 0:
        return "No documents indexed. Run index_documents() to start."

    all_data = collection.get(include=["metadatas"])["metadatas"]

    files = {}
    categories = {}
    total_words = 0
    for m in all_data:
        fname = m.get("filename", "unknown")
        cat = m.get("category", "other")
        wc = m.get("word_count", 0)
        files[fname] = files.get(fname, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
        total_words += wc

    file_list = "\n".join(
        f"  - {name} ({chunks} chunks)"
        for name, chunks in sorted(files.items())
    )
    cat_list = "\n".join(
        f"  - {cat}: {n} chunks"
        for cat, n in sorted(categories.items())
    )

    bm25_status = "Active" if _bm25_index is not None else "Not built (re-index needed)"
    ce_status = "Loaded" if _cross_encoder is not None else "Lazy (loads on first search)"

    return (
        f"## RAG Index Status\n\n"
        f"**Total:** {count} chunks | ~{total_words:,} words\n"
        f"**Embedding:** {EMBED_MODEL}\n"
        f"**BM25:** {bm25_status}\n"
        f"**Cross-Encoder:** {ce_status} ({RERANK_MODEL})\n"
        f"**DB:** {DB_PATH}\n\n"
        f"### Files ({len(files)}):\n{file_list}\n\n"
        f"### Categories:\n{cat_list}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
