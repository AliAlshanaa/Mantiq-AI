from langgraph.graph import StateGraph, END
from src.core.state import AgentState
from src.agents.researcher import researcher_node

# 1. Initialize the Graph with our custom State
workflow = StateGraph(AgentState)

# 2. Add our first node (The Researcher)
workflow.add_node("researcher", researcher_node)

# 3. Define the Flow
# We start with research
workflow.set_entry_point("researcher")

# For now, since we haven't built the Writer yet, 
# we will point the researcher directly to END to test it.
workflow.add_edge("researcher", END)

# 4. Compile the graph
app = workflow.compile()