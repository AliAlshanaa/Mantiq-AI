import itertools
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class IngestedSnippet:
    source: str
    content: str
    citation: str


def _tokenize_query(query: str) -> List[str]:
    query = (query or "").strip().lower()
    tokens = re.findall(r"[\w؀-ۿ]+", query, flags=re.UNICODE)
    return [t for t in tokens if len(t) >= 3]


def _score_text(text: str, tokens: Iterable[str]) -> int:
    t = (text or "").lower()
    return sum(1 for tok in tokens if tok in t)


def _qi(name: str) -> str:
    """Double-quote a SQLite identifier, escaping any embedded double-quotes."""
    return '"' + name.replace('"', '""') + '"'


def ingest_pdfs(
    query: str,
    pdf_dir: str,
    *,
    max_files: int = 10,
    max_pages_per_file: int = 4,
    max_snippets: int = 3,
    max_content_chars: int = 2000,
) -> List[IngestedSnippet]:
    """
    Lightweight PDF scanner (non-vector) to complement the vector store.
    Intended to catch PDFs that aren't indexed yet or to provide page-level citations.
    """
    tokens = _tokenize_query(query)
    if not tokens or not os.path.isdir(pdf_dir):
        return []

    try:
        # Local import so the app can still run without PDF deps.
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
    except Exception:
        return []

    # Sort for deterministic selection when the directory has more than max_files entries.
    pdf_files = sorted(
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )[:max_files]

    scored: List[Tuple[int, IngestedSnippet]] = []
    for path in pdf_files:
        try:
            loader = PyPDFLoader(path)
            # lazy_load() yields pages one at a time; islice caps memory to max_pages_per_file.
            pages = list(itertools.islice(loader.lazy_load(), max_pages_per_file))
        except Exception:
            continue

        for page in pages:
            text = (page.page_content or "").strip()
            score = _score_text(text, tokens)
            if score <= 0:
                continue

            page_no = page.metadata.get("page", None)
            base = os.path.basename(path)
            citation = f"{base} (PDF, page {page_no})" if page_no is not None else f"{base} (PDF)"
            snippet = IngestedSnippet(
                source=f"PDF_SCAN:{base}",
                content=text[:max_content_chars],
                citation=citation,
            )
            scored.append((score, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_snippets]]


def ingest_excels(
    query: str,
    excel_dir: str,
    *,
    max_files: int = 10,
    max_rows_per_sheet: int = 200,
    max_snippets: int = 3,
    max_content_chars: int = 2000,
) -> List[IngestedSnippet]:
    """
    Excel and CSV ingestion via pandas. Produces searchable table snippets.
    """
    tokens = _tokenize_query(query)
    if not tokens or not os.path.isdir(excel_dir):
        return []

    try:
        import pandas as pd  # type: ignore
    except Exception:
        print("--- ⚠️ Skipping Excel/CSV: pandas not installed ---")
        return []

    # Sort for deterministic selection when there are more files than max_files.
    files = sorted(
        os.path.join(excel_dir, f)
        for f in os.listdir(excel_dir)
        if f.lower().endswith((".xlsx", ".xls", ".xlsm", ".csv"))
    )[:max_files]

    scored: List[Tuple[int, IngestedSnippet]] = []
    for path in files:
        base = os.path.basename(path)
        is_csv = path.lower().endswith(".csv")

        try:
            if is_csv:
                sheet_names = ["Sheet1"]
                xls = None
            else:
                xls = pd.ExcelFile(path)
                sheet_names = xls.sheet_names[:5]
        except Exception as e:
            print(f"--- ⚠️ Error opening {base}: {e} ---")
            continue

        # Parse each sheet lazily so irrelevant sheets don't pay full parse cost.
        for sheet in sheet_names:
            try:
                df = (pd.read_csv(path) if is_csv else xls.parse(sheet)).head(max_rows_per_sheet)
            except Exception:
                continue

            text = df.to_string(index=False)
            score = _score_text(text, tokens)
            if score <= 0:
                continue

            citation = f"{base} ({'CSV' if is_csv else 'Excel'}, sheet: {sheet})"
            snippet = IngestedSnippet(
                source=f"DATA_FILE:{base}#{sheet}",
                content=text[:max_content_chars],
                citation=citation,
            )
            scored.append((score, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_snippets]]


def ingest_sqlite(
    query: str,
    db_path: str,
    *,
    max_tables: int = 12,
    max_rows_per_table: int = 8,
    max_snippets: int = 3,
    max_content_chars: int = 2000,
    deny_tables: Optional[Iterable[str]] = None,
) -> List[IngestedSnippet]:
    """
    Searches an internal SQLite database using LIKE across TEXT-like columns.
    This is a pragmatic default for "internal databases" without requiring schema changes.
    """
    tokens = _tokenize_query(query)
    raw_query = query.strip()
    # Accept the call even when tokens is empty (e.g. short words like "AI") — fall back to raw LIKE.
    if not raw_query or not db_path or not os.path.exists(db_path):
        return []

    deny = {t.lower() for t in (deny_tables or [])}

    snippets: List[IngestedSnippet] = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        tables = [
            r["name"]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        tables = [t for t in tables if t.lower() not in deny][:max_tables]

        for table in tables:
            try:
                cols = cur.execute(f"PRAGMA table_info({_qi(table)})").fetchall()
            except Exception:
                continue

            # Heuristic: only search columns that look like they might contain text.
            text_cols = [
                c[1] for c in cols
                if any(x in (c[2] or "").lower() for x in ["char", "text", "str", "blob"])
                or (c[2] or "") == ""
            ]

            if not text_cols:
                continue

            quoted_cols = [_qi(c) for c in text_cols]

            if tokens:
                # Each token must appear in at least one text column (AND across tokens, OR across cols).
                where_clauses = []
                params: List[str] = []
                for token in tokens:
                    token_clause = " OR ".join([f"{qc} LIKE ?" for qc in quoted_cols])
                    where_clauses.append(f"({token_clause})")
                    params.extend([f"%{token}%"] * len(quoted_cols))
                where = " AND ".join(where_clauses)
            else:
                # Fallback for short queries where tokenization yields nothing.
                col_clause = " OR ".join([f"{qc} LIKE ?" for qc in quoted_cols])
                where = f"({col_clause})"
                params = [f"%{raw_query}%"] * len(quoted_cols)

            try:
                rows = cur.execute(
                    f"SELECT * FROM {_qi(table)} WHERE {where} LIMIT {max_rows_per_table}",
                    params,
                ).fetchall()
            except Exception:
                continue

            if not rows:
                continue

            content = f"TABLE: {table}\nROWS: {[dict(r) for r in rows]}"
            effective_tokens = tokens if tokens else [raw_query]
            score = _score_text(content, effective_tokens)
            if score <= 0:
                continue

            citation = f"{os.path.basename(db_path)} (SQLite table: {table})"
            snippets.append(
                IngestedSnippet(
                    source=f"SQLITE:{table}",
                    content=content[:max_content_chars],
                    citation=citation,
                )
            )

            if len(snippets) >= max_snippets:
                break

    except Exception:
        return []
    finally:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass

    effective_tokens = tokens if tokens else [raw_query]
    snippets.sort(key=lambda s: _score_text(s.content, effective_tokens), reverse=True)
    return snippets[:max_snippets]
