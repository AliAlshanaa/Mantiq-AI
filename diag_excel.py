import time
import sys

print("START", flush=True)
t0 = time.time()

from src.ingestion.multi_source import ingest_excels

print(f"import done: {time.time()-t0:.2f}s", flush=True)

t1 = time.time()
snippets = ingest_excels("  35من هو علي الشناعه ", "./data/spreadsheets", max_snippets=3, max_content_chars=2000)
print(f"ingest_excels done: {time.time()-t1:.2f}s -> {len(snippets)} snippets", flush=True)
for s in snippets:
    print(s.citation, flush=True)

print("DONE", flush=True)
