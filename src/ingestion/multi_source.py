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
    tokens = re.findall(r"[\w\u0600-\u06FF]+", query, flags=re.UNICODE)
    return [t for t in tokens if len(t) >= 3]


def _score_text(text: str, tokens: Iterable[str]) -> int:
    t = (text or "").lower()
    return sum(1 for tok in tokens if tok in t)


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

    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ][:max_files]

    scored: List[Tuple[int, IngestedSnippet]] = []
    for path in pdf_files:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()[:max_pages_per_file]
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

    # Include CSVs in the scan
    files = [
        os.path.join(excel_dir, f)
        for f in os.listdir(excel_dir)
        if f.lower().endswith((".xlsx", ".xls", ".xlsm", ".csv"))
    ][:max_files]

    scored: List[Tuple[int, IngestedSnippet]] = []
    for path in files:
        base = os.path.basename(path)
        is_csv = path.lower().endswith(".csv")
        
        try:
            if is_csv:
                # Handle CSV
                df_map = {"Sheet1": pd.read_csv(path).head(max_rows_per_sheet)}
            else:
                # Handle Excel
                xls = pd.ExcelFile(path)
                df_map = {sheet: xls.parse(sheet).head(max_rows_per_sheet) for sheet in xls.sheet_names[:5]}
        except Exception as e:
            print(f"--- ⚠️ Error reading {base}: {e} ---")
            continue

        for sheet, df in df_map.items():
            # Turn the small table into text and score it.
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
    if not tokens or not db_path or not os.path.exists(db_path):
        return []

    deny = {t.lower() for t in (deny_tables or [])}
    like_query = f"%{query.strip()}%"

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
                cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
            except Exception:
                continue

                        # Heuristic: only try columns that look like they might contain text.
            text_cols = []
            for c in cols:
                col_name = c[1]
                col_type = (c[2] or "").lower()
                # Expanded type check
                if any(t in col_type for t in ["char", "text", "str", "blob"]) or col_type == "":
                    text_cols.append(col_name)

            if not text_cols:
                continue

            # Multi-word query support for SQLite
            where_clauses = []
            params = []
            for token in tokens:
                token_clause = " OR ".join([f"{c} LIKE ?" for c in text_cols])
                where_clauses.append(f"({token_clause})")
                params.extend([f"%{token}%"] * len(text_cols))
            
            where = " AND ".join(where_clauses)

            try:
                rows = cur.execute(
                    f"SELECT * FROM {table} WHERE {where} LIMIT {max_rows_per_table}",
                    params,
                ).fetchall()
            except Exception:
                continue


            if not rows:
                continue

            # Flatten matched rows into a compact string.
            rendered_rows = []
            for r in rows:
                d = dict(r)
                rendered_rows.append(d)
            content = f"TABLE: {table}\nROWS: {rendered_rows}"
            score = _score_text(content, tokens)
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

    snippets.sort(key=lambda s: _score_text(s.content, tokens), reverse=True)
    return snippets[:max_snippets]

