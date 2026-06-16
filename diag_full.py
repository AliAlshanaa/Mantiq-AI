import time

def step(name, fn):
    t0 = time.time()
    print(f"START {name}", flush=True)
    try:
        result = fn()
        print(f"DONE  {name}: {time.time()-t0:.2f}s -> {result}", flush=True)
        return result
    except Exception as e:
        print(f"ERROR {name}: {time.time()-t0:.2f}s -> {e}", flush=True)
        return None

task = "  35من هو علي الشناعه "

from src.database.vector_store import get_retriever
retriever = step("get_retriever", lambda: type(get_retriever()).__name__)
retriever = get_retriever()

step("retriever.invoke", lambda: len(retriever.invoke(task)))

from src.ingestion.multi_source import ingest_pdfs, ingest_sqlite
from src.database.db_manager import DB_PATH

step("ingest_pdfs", lambda: len(ingest_pdfs(task, "./data/documents", max_snippets=3, max_content_chars=2000)))
step("ingest_sqlite", lambda: len(ingest_sqlite(task, DB_PATH, max_snippets=3, max_content_chars=2000, deny_tables=["user_profile","task_history"])))

from tavily import TavilyClient
search_tool = TavilyClient()
step("tavily.search", lambda: len(search_tool.search(query=task, max_results=2).get("results", [])))

print("ALL DONE", flush=True)
