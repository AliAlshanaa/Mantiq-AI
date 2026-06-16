import time
from dotenv import load_dotenv
load_dotenv()

from src.core.factory import create_llm

print("creating llm...", flush=True)
t0 = time.time()
llm = create_llm("llama", temperature=0.3)
print(f"create_llm done: {time.time()-t0:.2f}s", flush=True)

print("invoking llm...", flush=True)
t1 = time.time()
resp = llm.invoke("Say OK in one word.")
print(f"invoke done: {time.time()-t1:.2f}s -> {resp.content!r}", flush=True)
