import time, os
from dotenv import load_dotenv
load_dotenv()

print("env TAVILY_API_KEY set:", bool(os.getenv("TAVILY_API_KEY")), flush=True)

t0 = time.time()
print("importing tavily...", flush=True)
from tavily import TavilyClient
print(f"import done: {time.time()-t0:.2f}s", flush=True)

t1 = time.time()
print("instantiating TavilyClient...", flush=True)
search_tool = TavilyClient()
print(f"instantiate done: {time.time()-t1:.2f}s", flush=True)

t2 = time.time()
print("calling search...", flush=True)
resp = search_tool.search(query="  35من هو علي الشناعه ", max_results=2)
print(f"search done: {time.time()-t2:.2f}s -> {len(resp.get('results', []))} results", flush=True)
