"""
functions/coco_index.py  –  ErrorX Code Corpus Indexer (CocoIndex)
===================================================================
LangGraph node that walks a cloned repository, chunks every supported
source file with Tree-sitter (falls back to line-window chunking when
Tree-sitter has no grammar), embeds each chunk, and persists a FAISS
index + metadata sidecar to:

    <workspace_path>/index/coco.faiss       (when faiss-cpu installed)
    <workspace_path>/index/coco_vectors.npy (numpy fallback)
    <workspace_path>/index/coco_meta.json

Embedding priority
------------------
1. sentence-transformers  (best quality, optional)
2. numpy TF-IDF           (always available, zero extra deps)

Position in the graph
---------------------
    build_tree  →  coco_index  →  analyze_surface  →  inject_bugs

IMPORTANT – node return contract
---------------------------------
This node returns ONLY the keys it owns.  It does NOT spread **state.
logs/errors are declared Annotated[List, operator.add] in models.py so
LangGraph merges them automatically — the node just returns new entries.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Optional imports (graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────

def _try_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def _try_st():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None


def _try_treesitter():
    try:
        import tree_sitter
        from tree_sitter import Language
        return tree_sitter, Language
    except ImportError:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".rb", ".rs", ".cpp", ".c", ".h",
    ".cs", ".php", ".swift", ".kt", ".scala",
)

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt",
}

EMBED_MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_CHUNK_LINES   = 40
FALLBACK_OVERLAP_LINES = 8
MAX_CHUNK_CHARS        = 2_000
TFIDF_VOCAB_SIZE       = 4_096

SEP  = "=" * 72
DASH = "-" * 72


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(repo_root: Path) -> List[Path]:
    results: List[Path] = []
    for path in repo_root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            results.append(path)
    return sorted(results)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_with_treesitter(
    source: str,
    filepath: Path,
    ts_module,
    Language_cls,
) -> Optional[List[Dict[str, Any]]]:
    ext_to_lang = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "tsx", ".jsx": "javascript", ".java": "java",
        ".go": "go", ".rb": "ruby", ".rs": "rust",
        ".cpp": "cpp", ".c": "c", ".cs": "c_sharp",
    }
    lang_name = ext_to_lang.get(filepath.suffix)
    if not lang_name:
        return None

    try:
        grammar_module = __import__(f"tree_sitter_{lang_name}")
        language       = Language_cls(grammar_module.language())
        parser         = ts_module.Parser(language)
    except Exception:
        return None

    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return None

    lines  = source.splitlines()
    chunks: List[Dict[str, Any]] = []

    DECL_TYPES = {
        "function_definition", "function_declaration",
        "method_definition",   "method_declaration",
        "class_definition",    "class_declaration",
        "decorated_definition", "arrow_function",
    }

    def _name(node) -> str:
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte]
        return node.type

    def _visit(node) -> None:
        if node.type in DECL_TYPES:
            start = node.start_point[0]
            end   = node.end_point[0]
            text  = "\n".join(lines[start:end + 1])[:MAX_CHUNK_CHARS]
            chunks.append({
                "text":       text,
                "start_line": start + 1,
                "end_line":   end + 1,
                "node_type":  node.type,
                "name":       _name(node),
            })
        else:
            for child in node.children:
                _visit(child)

    _visit(tree.root_node)
    return chunks if chunks else None


def _chunk_by_lines(source: str) -> List[Dict[str, Any]]:
    lines  = source.splitlines()
    chunks: List[Dict[str, Any]] = []
    step   = max(FALLBACK_CHUNK_LINES - FALLBACK_OVERLAP_LINES, 1)

    for i in range(0, len(lines), step):
        window = lines[i:i + FALLBACK_CHUNK_LINES]
        text   = "\n".join(window)[:MAX_CHUNK_CHARS]
        chunks.append({
            "text":       text,
            "start_line": i + 1,
            "end_line":   i + len(window),
            "node_type":  "line_window",
            "name":       f"lines_{i + 1}_{i + len(window)}",
        })

    return chunks


def _chunk_file(
    filepath: Path,
    repo_root: Path,
    ts_module,
    Language_cls,
) -> List[Dict[str, Any]]:
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rel_path = str(filepath.relative_to(repo_root))
    lang     = filepath.suffix.lstrip(".")

    raw: Optional[List[Dict]] = None
    if ts_module is not None:
        raw = _chunk_with_treesitter(source, filepath, ts_module, Language_cls)

    raw = raw or _chunk_by_lines(source)

    return [
        {**c, "rel_path": rel_path, "lang": lang}
        for c in raw
        if c["text"].strip()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF fallback embedder (pure numpy)
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class _TFIDFEmbedder:
    """
    Lightweight TF-IDF vectoriser over code tokens.
    Produces L2-normalised float32 vectors suitable for cosine similarity.
    """

    def __init__(self, vocab_size: int = TFIDF_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.vocab:  Dict[str, int] = {}
        self.idf:    np.ndarray     = np.array([])

    def fit(self, texts: List[str]) -> "_TFIDFEmbedder":
        N  = len(texts)
        df: Counter = Counter()
        tokenised   = [_tokenize(t) for t in texts]
        for toks in tokenised:
            df.update(set(toks))

        top = [tok for tok, _ in df.most_common(self.vocab_size)]
        self.vocab = {tok: i for i, tok in enumerate(top)}
        self.idf   = np.array(
            [math.log((N + 1) / (df[tok] + 1)) + 1.0 for tok in top],
            dtype=np.float32,
        )
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        V      = len(self.vocab)
        matrix = np.zeros((len(texts), V), dtype=np.float32)
        for row, text in enumerate(texts):
            toks  = _tokenize(text)
            total = len(toks) or 1
            tf    = Counter(toks)
            for tok, cnt in tf.items():
                if tok in self.vocab:
                    matrix[row, self.vocab[tok]] = (cnt / total) * self.idf[self.vocab[tok]]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Unified embed
# ─────────────────────────────────────────────────────────────────────────────

def _embed_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 64,
) -> Tuple[np.ndarray, str]:
    """
    Returns (vectors float32 ndarray, backend_name str).
    Tries sentence-transformers first; falls back to TF-IDF.
    """
    texts = [
        f"[{c['rel_path']}] {c['name']}\n{c['text']}"
        for c in chunks
    ]

    ST = _try_st()
    if ST is not None:
        print(f"  Embedding {len(texts)} chunks with sentence-transformers …")
        model   = ST(EMBED_MODEL_NAME)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        return vectors, "sentence-transformers"

    print(f"  sentence-transformers not available — using TF-IDF fallback")
    print(f"  Embedding {len(texts)} chunks with TF-IDF …")
    vectors = _TFIDFEmbedder().fit_transform(texts)
    return vectors, "tfidf"


# ─────────────────────────────────────────────────────────────────────────────
# Index persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_index(
    vectors:   np.ndarray,
    meta:      List[Dict[str, Any]],
    index_dir: Path,
) -> Tuple[Path, Path]:
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "coco_meta.json"

    slim = [{k: v for k, v in c.items() if k != "text"} for c in meta]
    meta_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")

    faiss = _try_faiss()
    if faiss is not None:
        index_path = index_dir / "coco.faiss"
        idx        = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(vectors)
        faiss.write_index(idx, str(index_path))
    else:
        index_path = index_dir / "coco_vectors.npy"
        np.save(str(index_path), vectors)

    return index_path, meta_path


# ─────────────────────────────────────────────────────────────────────────────
# Public query helper
# ─────────────────────────────────────────────────────────────────────────────

def query_index(
    query:      str,
    index_path: str | Path,
    meta_path:  str | Path,
    top_k:      int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks for a natural-language query.
    Works with both the FAISS and numpy backends.
    Returns metadata dicts with an extra 'score' field.
    """
    index_path = Path(index_path)
    meta       = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    ST = _try_st()
    if ST is not None:
        model  = ST(EMBED_MODEL_NAME)
        vector = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
    else:
        vector = _TFIDFEmbedder().fit_transform([query])

    faiss = _try_faiss()
    if faiss is not None and index_path.suffix == ".faiss":
        idx            = faiss.read_index(str(index_path))
        scores, idxs   = idx.search(vector, top_k)
        pairs          = list(zip(scores[0], idxs[0]))
    else:
        stored = np.load(str(index_path))
        sims   = (stored @ vector.T).squeeze()
        top    = np.argsort(sims)[::-1][:top_k]
        pairs  = [(float(sims[i]), i) for i in top]

    return [
        {**meta[int(i)], "score": float(s)}
        for s, i in pairs
        if 0 <= int(i) < len(meta)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def indexer_node(state: dict) -> dict:
    """
    ErrorX LangGraph node — builds the CocoIndex for a cloned repository.

    Returns ONLY owned keys (no **state spread).
    logs/errors are new entries only; LangGraph merges via operator.add.
    """
    repo_path:      str = state["repo_path"]
    workspace_path: str = state.get("workspace_path") or repo_path

    new_logs:   List[str] = []
    new_errors: List[str] = []

    repo_root = Path(repo_path)
    index_dir = Path(workspace_path) / "index"

    print(SEP)
    print("  ErrorX – CocoIndex (Code Corpus Indexer)")
    print(f"  Repo      : {repo_path}")
    print(f"  Index dir : {index_dir}")
    print(SEP)

    ts_module, Language_cls = _try_treesitter()
    if ts_module is None:
        print("  [WARN] tree-sitter not found – using line-window chunker")
        new_logs.append("[coco_index] tree-sitter unavailable; using fallback chunker")

    # ── discover ──────────────────────────────────────────────────────────────
    files = _discover_files(repo_root)
    if not files:
        msg = f"[coco_index] No indexable source files found under {repo_path}"
        new_errors.append(msg)
        print(f"  [ERROR] {msg}")
        return {"logs": new_logs, "errors": new_errors}

    print(f"  Discovered {len(files)} source file(s)")

    # ── chunk ─────────────────────────────────────────────────────────────────
    all_chunks: List[Dict[str, Any]] = []
    for fp in files:
        all_chunks.extend(_chunk_file(fp, repo_root, ts_module, Language_cls))

    if not all_chunks:
        msg = "[coco_index] Zero chunks produced – index not built"
        new_errors.append(msg)
        print(f"  [ERROR] {msg}")
        return {"logs": new_logs, "errors": new_errors}

    print(f"  Total chunks : {len(all_chunks)}")

    # ── embed ─────────────────────────────────────────────────────────────────
    try:
        vectors, backend = _embed_chunks(all_chunks)
    except Exception as exc:
        msg = f"[coco_index] Embedding failed: {exc}"
        new_errors.append(msg)
        print(f"  [ERROR] {msg}")
        return {"logs": new_logs, "errors": new_errors}

    # ── persist ───────────────────────────────────────────────────────────────
    try:
        index_path, meta_path = _save_index(vectors, all_chunks, index_dir)
    except Exception as exc:
        msg = f"[coco_index] Index save failed: {exc}"
        new_errors.append(msg)
        print(f"  [ERROR] {msg}")
        return {"logs": new_logs, "errors": new_errors}

    print(DASH)
    print(f"  Backend : {backend}")
    print(f"  Index   → {index_path}")
    print(f"  Meta    → {meta_path}")
    print(f"  Chunks  : {len(all_chunks)}  |  Vectors : {vectors.shape}")
    print(SEP)

    new_logs.append(
        f"[coco_index] ✓ {len(files)} files → {len(all_chunks)} chunks "
        f"[{backend}]  ({index_path})"
    )

    return {
        "index_path":      str(index_path),
        "index_meta_path": str(meta_path),
        "logs":            new_logs,
        "errors":          new_errors,
    }